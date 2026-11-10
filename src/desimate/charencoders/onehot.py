import logging

import numpy as np
from unidecode import unidecode

from desimate.charencoders import CharacterEncoder

logger = logging.getLogger(__name__)


class OneHotCE(CharacterEncoder):
    """
    Encode a character using a OneHot representation.
    An alphabet is locally maintained.
    You should call `train` first to construct this alphabet.
    In case a character in the text is not in the alphabet, it is
    silently discarded.
    """

    def __init__(self, max_length: int | None = None):
        super().__init__(max_length)
        self.alphabet = {}

    def train(self, texts: list[str]):
        characters = set()
        max_length = 0
        for text in texts:
            max_length = max(max_length, len(text))
            characters.update(text)
        if self.max_length is None:
            self.max_length = max_length
        self.alphabet = {_c: _i for _i, _c in enumerate(characters)}
        logger.info(
            "Total of %d characters found, forming the OneHot alphabet.",
            len(self.alphabet),
        )

    def text_to_signal(self, text: list[str]):
        return [self.alphabet[_c] for _c in text if _c in self.alphabet]

    def run(self, text: str | list[str]) -> np.ndarray:
        _text = text[: self.max_length]
        if isinstance(_text, str):
            _text = list(_text)
        sig = self.text_to_signal(_text)

        encode = np.zeros((self.max_length, len(self.alphabet)), dtype=np.float32)
        encode[np.arange(len(sig), dtype=int), np.array(sig, dtype=int)] = 1.0
        return encode

    def __len__(self):
        return len(self.alphabet)


class AsciiOneHotCE(OneHotCE):
    """
    Encode a character using a OneHot representation.
    To simplify the alphabet, first transform the text towards
    ASCII only character using `unidecode`.

    """

    def __init__(self, max_length: int | None = None):
        super().__init__(max_length)

    def _apply(self, c):
        return unidecode(c).lower()

    def train(self, texts: list[str]):
        super().train([self._apply(_t) for _t in texts])

    def run(self, text: str | list[str]) -> np.ndarray:
        return super().run([self._apply(_t) for _t in text])
