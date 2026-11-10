from pathlib import Path
import logging
from typing import Generator, Tuple

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from desimate.charencoders import CharacterEncoder
from desimate.distances import ComputeAllPairDistance
from desimate.distances.levenshtein import LevenshteinDistance
from desimate.encoders import Vectorizer, VectorizerTrainer
from desimate.encoders.cnned.datasets import (
    TripletString,
)
from desimate.encoders.datasets import StringDataset
from desimate.encoders.cnned.networks import (
    TwoLayerCNN,
    TripletNet,
    AdaptativeTripletLoss,
    FixedTripletLoss,
)
from desimate.utils.torchutils import get_device

logger = logging.getLogger(__name__)


class CNNEDTrainer(VectorizerTrainer):
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

    def _compute_dist(self, texts: list[str]) -> np.ndarray:
        all_pair = ComputeAllPairDistance(LevenshteinDistance(texts))
        return all_pair()

    def _compute_knn(self, dist: np.ndarray, k: int | None = None):
        k = k or (dist.shape[1] - 1)
        return np.argpartition(dist, k, axis=1)[:, :k]

    def _compute_dist_knn(self, texts: list[str], k: int | None = None):
        dists = self._compute_dist(texts)
        knn = self._compute_knn(dists, k)
        return dists, knn

    def _save_model(self, model, path: Path, maximum_length: int):
        # TODO: Prefer save state dict than serialization
        torch.save(
            {
                "embedding_model": model.embedding_net,
                "max_length": maximum_length,
                "char_vectorizer": self.char_vectorizer,
            },
            path,
        )

    def _step(
        self, model, optimizer, train_loader, loss, epoch
    ) -> Tuple[float, float, float]:
        model.train()
        epoch_task = self.progress.add_task(f"Epoch {epoch}", total=len(train_loader))
        agg, agg_r, agg_m = 0, 0, 0
        optimizer.zero_grad()
        for idx, batch in enumerate(train_loader):
            (
                anchor,
                pos,
                neg,
                anchor_len,
                pos_len,
                neg_len,
                pos_dist,
                neg_dist,
                pos_neg_dist,
            ) = (i.to(self.device) for i in batch)

            optimizer.zero_grad()
            output = model((anchor, pos, neg))
            r, m, train_loss = loss(
                output,
                (anchor_len, pos_len, neg_len),
                (pos_dist, neg_dist, pos_neg_dist),
                epoch,
            )
            train_loss.backward()
            optimizer.step()

            agg += train_loss.item()
            agg_r += r.item()
            agg_m += m.item()
            self.progress.update(epoch_task, advance=1)
        self.progress.remove_task(epoch_task)
        return agg / (epoch + 1), agg_r / (epoch + 1), agg_m / (epoch + 1)

    def _validate(self, model, valid_loader, loss, epoch) -> Tuple[float, float, float]:
        valid_task = self.progress.add_task("Validation", total=len(valid_loader))
        model.eval()
        agg, agg_r, agg_m = 0.0, 0.0, 0.0
        for idx, batch in enumerate(valid_loader):
            (
                anchor,
                pos,
                neg,
                anchor_len,
                pos_len,
                neg_len,
                pos_dist,
                neg_dist,
                pos_neg_dist,
            ) = (i.to(self.device) for i in batch)
            output = model((anchor, pos, neg))
            r, m, valid_loss = loss(
                output,
                (anchor_len, pos_len, neg_len),
                (pos_dist, neg_dist, pos_neg_dist),
                epoch,
            )
            agg += valid_loss.item()
            agg_r += r.item()
            agg_m += m.item()
            self.progress.update(valid_task, advance=1)

        self.progress.remove_task(valid_task)
        return agg / (epoch + 1), agg_r / (epoch + 1), agg_m / (epoch + 1)

    def _train_epoch(
        self,
        model,
        train_loader,
        valid_loader,
        epochs,
        maximum_length,
        lr,
        patience: int = 20,
        save_every=None,
    ):
        task = self.progress.add_task("Training", total=epochs)
        losser = AdaptativeTripletLoss(epochs)
        valid_losser = FixedTripletLoss()
        optimizer = Adam(model.parameters(), lr=lr)
        best = 1e10
        best_epoch = -1

        for epoch in range(epochs):
            agg, agg_r, agg_m = self._step(
                model, optimizer, train_loader, losser, epoch
            )
            v_agg, v_agg_r, v_agg_m = self._validate(
                model, valid_loader, valid_losser, epoch
            )

            self.progress.update(task, advance=1)
            self.update_status(
                f"Epoch: {epoch + 1:2d}\t"
                f"Loss: {agg:.2f}\t"
                f"r:{agg_r:.2f}\t"
                f"m: {agg_m:.2f}\t"
                f"Validation Loss: {v_agg:.2f}\t"
                f"Vr:{v_agg_r:.2f}\t"
                f"Vm: {v_agg_m:.2f}"
            )
            if save_every and epoch % save_every == 0:
                p = Path("models") / f"model_{epoch}.torch"
                logger.info("Saving model at '%s' for epoch %d... ", str(p), epoch)
                self._save_model(model, p, maximum_length)

            if v_agg < best:
                best = v_agg
                best_epoch = epoch
                p = Path("models") / "model_best.torch"
                logger.info(
                    "Saving model at '%s' because it is the current best one", str(p)
                )
                self._save_model(model, p, maximum_length)

            if epoch - best_epoch > patience:
                logger.info("Stopping there was no enhancements in %d epochs", patience)
                break

        return model

    def run(
        self,
        output_model: Path,
        texts: list[str],
        n_training: int,
        n_query: int,
        sampling_threshold: int,
        maximum_length: int | None = None,
        epochs: int = 20,
        batch_size: int = 64,
        embedding_size: int = 128,
        channel: int = 8,
        mtc: bool = False,
        lr: float = 1e-3,
        save_every: int | None = 5,
        patience: int = 20,
        cache_distance_matrix: Path = None,
    ):
        logger.debug("Starting training")
        output_model.parent.mkdir(exist_ok=True)
        train, query, _ = self._separate_dataset(len(texts), n_training, n_query)

        # Check maximum length

        _maxl = self._get_maximum_length(texts)
        maximum_length = maximum_length or _maxl
        if maximum_length < _maxl:
            logger.warning(
                "The given maximum length (%d) is lower than "
                "the maximum length of the training data (%d).",
                maximum_length,
                _maxl,
            )

        # C = len(alphabet), M = max length, sig = ligne transformé en list[int]

        logger.info("Training the character encoders...")
        self.char_vectorizer.train(texts)

        logger.info("Creating the datasets...")
        train_lines: list[str] = [texts[_i][:maximum_length] for _i in train]
        valid_lines: list[str] = [texts[_i][:maximum_length] for _i in query]

        train_lines_dataset = StringDataset(
            train_lines, self.char_vectorizer, maximum_length
        )
        valid_lines_dataset = StringDataset(
            valid_lines, self.char_vectorizer, maximum_length
        )

        compute_cached = self.memory.cache(self._compute_dist_knn)
        train_dist, train_knn = compute_cached(train_lines)
        valid_dist, valid_knn = compute_cached(valid_lines)

        train_dataset = TripletString(
            train_lines, train_lines_dataset, train_knn, train_dist
        )
        valid_dataset = TripletString(
            valid_lines, valid_lines_dataset, valid_knn, valid_dist
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        net = TwoLayerCNN(
            len(self.char_vectorizer),
            maximum_length,
            embedding=embedding_size,
            channel=channel,
            mtc_input=mtc,
        )
        model = TripletNet(net).to(self.device)

        logger.info("Beginning training...")
        model = self._train_epoch(
            model,
            train_loader,
            valid_loader,
            epochs,
            maximum_length,
            lr,
            patience,
            save_every,
        )
        logger.info("Saving model at '%s'", output_model)
        torch.save(
            {
                "embedding_model": model.embedding_net,
                "max_length": maximum_length,
                "char_vectorizer": self.char_vectorizer,
            },
            output_model,
        )
        logger.debug("Training ended successfully.")
