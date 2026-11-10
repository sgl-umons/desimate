import numpy as np

from desimate.charencoders import CharacterEncoder


class OrdCV(CharacterEncoder):
    def __init__(self, max_length: int | None = None, length: int | None = 32):
        super().__init__(max_length)
        self.config = self.get_config()
        self.length = length or self.config.get("length")
        if not self.length:
            raise ValueError("No length was given...")

    def train(self, texts: list[str]):
        if self.max_length is None:
            self.max_length = max((len(_t) for _t in texts))

    def run(self, text: str | list[str]) -> np.ndarray:
        res = np.zeros((self.max_length, self.length), dtype=np.float32)
        values = np.fromiter((ord(c) for c in text), dtype=np.uint32)
        text_length = min(len(text), self.max_length)
        res[:text_length] = (
            (values[:, None] >> np.arange(self.length - 1, -1, -1)) & 1
        ).astype(np.float32)[:text_length]
        return res

    def __len__(self):
        return self.length


class OrdCV2(CharacterEncoder):
    def __init__(self, max_length: int | None = None, length: int | None = 64):
        super().__init__(max_length)
        self.config = self.get_config()
        self.length = length or self.config.get("length")
        if not self.length:
            raise ValueError("No length was given...")

    def train(self, texts: list[str]):
        if self.max_length is None:
            self.max_length = max((len(_t) for _t in texts))

    def run(self, text: str | list[str]) -> np.ndarray:
        chars = text[: self.max_length]
        values = np.fromiter(map(ord, chars), dtype=np.int32)

        res = np.zeros((self.max_length, self.length), dtype=np.float32)

        while np.any(values > 0):
            mask = values > 0
            indices = values[mask] % self.length
            res[np.flatnonzero(mask), indices] = 1.0
            values[mask] //= self.length

        return res

    def __len__(self):
        return self.length
