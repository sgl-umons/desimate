import torch
from torch.utils.data import Dataset

from desimate.charencoders import CharacterEncoder


class StringDataset(Dataset):
    def __init__(
        self, texts: list[str], char_vectorizer: CharacterEncoder, max_length: int
    ):
        super().__init__()
        self.texts = texts
        self.max_length = max_length
        self.char_vectorizer = char_vectorizer

    def __getitem__(self, index):
        text = self.texts[index][: self.max_length]
        return torch.from_numpy(self.char_vectorizer(text))

    def __len__(self):
        return len(self.texts)
