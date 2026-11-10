import logging
from pathlib import Path
from typing import Generator, Type

import joblib
import numpy as np
import torch
from torch import optim
from torch.nn import MSELoss
from torch.nn.functional import smooth_l1_loss, cross_entropy, softmax, softplus
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from desimate.charencoders import CharacterEncoder
from desimate.distances import DistanceMetric
from desimate.distances.levenshtein import (
    NormalizedLevenshteinDistance,
    LevenshteinDistance,
)
from desimate.encoders import VectorizerTrainer, Vectorizer, StringDataset
from desimate.encoders.cnned import ComputeAllPairDistance
from desimate.encoders.neuroseed.datasets import (
    TrainingEditDistanceDataset,
    ValidationEditDistanceDataset,
)
from desimate.encoders.neuroseed.networks import CNN, PairEmbeddingDistance
from desimate.utils.torchutils import get_device
from desimate.encoders.neuroseed.utils import AverageMeter, mape

logger = logging.getLogger(__name__)


class NeuroSeedCNNTrainer(VectorizerTrainer):
    def __init__(
        self,
        char_vectorizer: CharacterEncoder,
        no_cuda=False,
        memory: joblib.Memory = None,
    ):
        super().__init__()
        self.device = get_device(no_cuda=no_cuda)
        self.char_vectorizer = char_vectorizer
        self.memory = memory or joblib.Memory()

    def _separate_dataset(self, n: int, nt: int, nq: int):
        """
        Randomly separate a dataset of `n` elements into 3 parts: `nt` training items, `nq` query items,
        and everything else.

        Returns a tuple of 3 arrays containing the indices of each part as (training, query, others).
        """
        if n - nq - nt < 0:
            raise ValueError(
                "Cannot sample %d query items and %d training samples with a dataset of %d items."
                % (nq, nt, n)
            )
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        train = idxs[:nt]
        query = idxs[nt : nt + nq]
        base = idxs[nt + nq :]
        return train, query, base

    def _get_maximum_length(self, texts: list[str]):
        _m = max([len(t) for t in texts])
        return _m

    def _save_model(self, model, path: Path, maximum_length: int):
        torch.save(
            {
                "embedding_model": model.embedding_model,
                "max_length": maximum_length,
                "char_vectorizer": self.char_vectorizer,
            },
            path,
        )

    def _validate(self, model, validate_loader, loss):
        task = self.progress.add_task("Validating", total=len(validate_loader))
        avg_loss = AverageMeter(len_tuple=2)
        model.eval()

        for sequences, labels in validate_loader:
            sequences, labels = sequences.to(self.device), labels.to(self.device)

            output = model(sequences)
            loss_val = loss(output, labels).data.item()
            mape_ = mape(output, labels).data.item()
            avg_loss.update((loss_val, mape_), sequences.shape[0])
            self.progress.advance(task, advance=1)

        self.progress.remove_task(task)

        return avg_loss

    @staticmethod
    def _loss3(pred, target, alpha=0.7, eps=1e-6):
        pred = pred.clamp(min=0)
        relative = ((pred - target) / (pred + target + eps)).pow(2).mean()
        absolute = smooth_l1_loss(pred, target)
        return alpha * relative + (1 - alpha) * absolute

    @staticmethod
    def _loss4(pred, target, alpha=0.6, beta=1.5, eps=1e-6):
        pred = pred.clamp(min=0)
        weights = torch.where(pred > target, beta, 1.0)
        relative = ((pred - target) / (pred + target + eps)).pow(2)
        rel_loss = (weights * relative).mean()
        absolute = smooth_l1_loss(pred, target, reduction="none")
        abs_loss = (weights * absolute).mean()
        return alpha * rel_loss + (1 - alpha) * abs_loss

    def _run_epoch(
        self,
        model,
        epochs,
        train_loader,
        validate_loader,
        optimizer,
        scheduler,
        maximum_length: int,
        save_every=None,
        patience: int = 20,
    ):
        task = self.progress.add_task("Training", total=epochs)

        loss = self._loss4
        # loss = MSELoss()
        best = 1e10
        best_mape = 1e10
        best_epoch = -1

        for epoch in range(epochs):
            epoch_task = self.progress.add_task(
                f"Epoch {epoch}", total=len(train_loader)
            )

            avg_loss = AverageMeter()
            model.train()
            for sequences, labels in train_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = model(sequences)
                loss_train = loss(output, labels)
                loss_train.backward()
                optimizer.step()
                avg_loss.update(loss_train.data.item(), sequences.shape[0])
                self.progress.update(epoch_task, advance=1)
            self.progress.remove_task(epoch_task)

            val_loss, mape_ = self._validate(model, validate_loader, loss).get_average()
            train_loss = avg_loss.get_average()

            if val_loss < best:
                best = val_loss
                best_epoch = epoch
                p = Path("models") / "model_best.torch"
                logger.info(
                    "Saving model at '%s' because it is the current best one", str(p)
                )
                self._save_model(model, p, maximum_length)

            if mape_ < best_mape:
                best_mape = mape_
                p = Path("models") / "model_best_mape.torch"
                logger.info(
                    "Saving model at '%s' because it is the current best one (MAPE)",
                    str(p),
                )
                self._save_model(model, p, maximum_length)

            if save_every and epoch % save_every == 0:
                p = Path("models") / f"model_{epoch}.torch"
                logger.info("Saving model at '%s' for epoch %d... ", str(p), epoch)
                self._save_model(model, p, maximum_length)

            if 0 < patience < epoch - best_epoch:
                logger.info("Stopping there was no enhancements in %d epochs", patience)
                break

            if scheduler:
                scheduler.step(val_loss)

            self.progress.update(task, advance=1)
            self.update_status(
                f"Epoch: {epoch + 1:2d}\t"
                f"lr: {optimizer.param_groups[0]['lr']:.2e}\t"
                f"Train Loss: {train_loss:.2f}\t"
                f"Validation Loss: {val_loss:.2f}\t"
                f"MAPE: {mape_:.2f}\t"
                f"Best Validation Loss: {best:.2f}\t"
                f"Best MAPE: {best_mape:.2f}\t"
            )

        return model

    def run(
        self,
        output_model: Path,
        texts: list[str],
        n_training: int,
        n_query: int,
        maximum_length: int | None = None,
        epochs: int = 20,
        batch_size: int = 64,
        embedding_size: int = 128,
        channel: int = 16,
        kernel_size: int = 3,
        pooling: str = "avg",
        layers: int = 4,
        readout_layers: int = 1,
        non_linearity: bool = True,
        lr: float = 0.01,
        weight_decay=0.0,
        save_every: int | None = 5,
        patience: int = -1,
        cache_distance_matrix: Path = None,
        distance: str = "euclidean",
        distance_metric: Type[DistanceMetric] = LevenshteinDistance,
    ):
        train, valid, _ = self._separate_dataset(len(texts), n_training, n_query)
        _maxl = self._get_maximum_length(texts)
        maximum_length = maximum_length or _maxl
        self.char_vectorizer.train(texts)
        self.char_vectorizer.max_length = maximum_length
        if maximum_length < _maxl:
            logger.warning(
                "The given maximum length (%d) is lower than "
                "the maximum length of the training data (%d).",
                maximum_length,
                _maxl,
            )

        embedding_model = CNN(
            len_sequence=maximum_length,
            embedding_size=embedding_size,
            channels=channel,
            readout_layers=readout_layers,
            layers=layers,
            kernel_size=kernel_size,
            alphabet_size=len(self.char_vectorizer),
            pooling=pooling,
            non_linearity=non_linearity,
        )
        model = PairEmbeddingDistance(
            embedding_model=embedding_model, distance=distance, scaling=False
        )
        model.to(self.device)
        train_texts: list[str] = [texts[_i][:maximum_length] for _i in train]
        valid_texts: list[str] = [texts[_i][:maximum_length] for _i in valid]

        train_dataset = TrainingEditDistanceDataset(
            train_texts, distance_metric, self.char_vectorizer, maximum_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        validation_dataset = ValidationEditDistanceDataset(
            valid_texts, distance_metric, self.char_vectorizer, maximum_length
        )
        valid_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, threshold=1e-4, min_lr=1e-6
        )
        model = self._run_epoch(
            model=model,
            epochs=epochs,
            train_loader=train_loader,
            validate_loader=valid_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            maximum_length=maximum_length,
            save_every=save_every,
            patience=patience,
        )
        self._save_model(model, output_model, maximum_length)
