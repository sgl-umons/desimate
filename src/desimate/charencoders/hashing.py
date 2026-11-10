from functools import lru_cache

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from hashlib import sha256

from desimate.charencoders import CharacterEncoder


class HashingCE(CharacterEncoder):
    """
    Encode a character using the hashing trick.
    """

    def __init__(self, max_length: int | None = None, length: int | None = None):
        super().__init__(max_length)
        self.config = self.get_config()
        self.length = length or self.config.get("length") or 128
        if not length:
            raise ValueError("No length was given...")

    def train(self, texts: list[str]):
        if self.max_length is None:
            self.max_length = max((len(_t) for _t in texts))

    def run(self, text: str | list[str]) -> np.ndarray:
        feature_hasher = FeatureHasher(n_features=self.length, input_type="string")
        encode = np.zeros((self.max_length, self.length), dtype=np.float32)
        text_length = min(len(text), self.max_length)
        encode[:text_length] = feature_hasher.transform(
            [[_c] for _c in text]
        ).toarray()[:text_length]
        return encode

    def __len__(self):
        return self.length


class HashingCE2(CharacterEncoder):
    """
    Encode a character using the hashing trick.
    Unlike `HashingCE`, it does not rely on `FeatureHasher`,
    but on a BLOOM filter.
    """

    def __init__(self, max_length: int | None = None, length: int = 64):
        super().__init__(max_length)
        self.length = length

    def train(self, texts: list[str]):
        if self.max_length is None:
            self.max_length = max((len(_t) for _t in texts))

    @lru_cache(maxsize=128)
    def _hash(self, c):
        digest = sha256(c.encode("utf-8")).digest()
        h1 = int.from_bytes(digest[:8], "little") % self.length
        h2 = int.from_bytes(digest[8:16], "little") % self.length
        h3 = int.from_bytes(digest[16:24], "little") % self.length
        return h1, h2, h3

    def run(self, text: str | list[str]) -> np.ndarray:
        encode = np.zeros((self.max_length, self.length), dtype=np.float32)
        for _i, _c in enumerate(text):
            _h1, _h2, _h3 = self._hash(_c)
            encode[_i, [_h1, _h2, _h3]] = 1
        return encode

    def __len__(self):
        return self.length
