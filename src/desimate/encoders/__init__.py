from pathlib import Path
from typing import Generator, NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from desimate import ProgressStep, StatusProgressStep
from desimate.utils.torchutils import get_device
from desimate.encoders.datasets import StringDataset


class ValueEmbedding(NamedTuple):
    file_id: str
    ind: int
    embedding: np.ndarray


class Vectorizer(ProgressStep):
    def __init__(self, show_progress: bool = True):
        super().__init__(show_progress)

    def run(self, text: str | list[str]) -> np.ndarray:
        """
        Vectorize a given text (or list of text) towards a representative
        vectorial embedding.

        Returns a 1D array if only a single str is given.
        Returns a 2D array if a list of str is given, where the embeddings
        are in the same order of the given list.
        """
        pass


class VectorizerTrainer(StatusProgressStep):
    def __init__(self):
        super().__init__()

    def run(self, *args, **kwargs):
        """
        Train a model given a representative dataset.
        """
        raise NotImplementedError()


class BaseVectorizer(Vectorizer):
    def __init__(self, model_path: Path, no_cuda=False, show_progress: bool = True):
        super().__init__(show_progress)
        self.no_cuda = no_cuda
        self.model_path = model_path
        self.char_vectorizer = None
        self.device = get_device(self.no_cuda)
        self.model = None
        self.max_length = None

    def load_encoder(self):
        model = torch.load(self.model_path, weights_only=False)
        self.model, self.max_length, self.char_vectorizer = (
            model["embedding_model"].to(self.device),
            int(model["max_length"]),
            model["char_vectorizer"],
        )
        self.model.to(self.device)
        return True

    def embed_texts(
        self,
        texts: list[str],
        batch_size=128,
    ) -> Generator[np.ndarray]:
        """Embed a string or a list of strings.

        ``texts: list[str]`` -> 2-D numpy array of shape (n, dim)
        """
        if self.model is None:
            raise Exception(
                "The encoder is not yet loaded... Call `load_encoder` in order to load one."
            )

        if len(texts) == 0:
            raise ValueError("No strings to embed")
        ds = StringDataset(texts, self.char_vectorizer, self.max_length)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        task = self.progress.add_task("Embedding", total=len(loader))

        self.model.eval()
        with torch.no_grad():
            for _e in loader:
                _r = self.model(_e.to(self.device)).cpu().numpy()
                yield _r
                self.progress.update(task, advance=1)

    def run(self, text: list[str], batch_size=128) -> np.ndarray:
        if not self.model:
            self.load_encoder()
        return self.embed_texts(text, batch_size)
