from abc import abstractmethod

from desimate import Step
import numpy as np


class CharacterEncoder(Step):
    """
    This class represents the logic for transforming a single character
    into a vector representing this character.
    Possible representation might include onehot encoding, UTF-8 padded to
    4-bytes, etc.

    Before using this encoder, you should first call `train` with your data.
    You can then use `run` to encode your string or list of character.
    """

    def __init__(self, max_length: int | None = None):
        super().__init__()
        self.max_length = max_length

    @abstractmethod
    def train(self, texts: list[str]):
        pass

    @abstractmethod
    def run(self, text: str | list[str]) -> np.ndarray:
        """
        Vectorize a str (or list of characters) towards a representative
        vectorial embedding.

        Returns a 2D array where the embeddings are in the same order of the input.
        """
        pass

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()
