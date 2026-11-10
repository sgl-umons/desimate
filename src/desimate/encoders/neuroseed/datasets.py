import random
import string
from typing import Type

from rapidfuzz.distance import Levenshtein
import numpy as np
import torch
from torch.utils.data import Dataset

from desimate.charencoders import CharacterEncoder
from desimate.distances import DistanceMetric

DEFAULT_ALPHABET = string.ascii_letters + string.digits + " "


def modify_text(text, n_edits: int, alphabet=DEFAULT_ALPHABET) -> str:
    chars = list(text)
    for _ in range(n_edits):
        op = random.randrange(3)
        if op == 0:
            chars.insert(random.randrange(len(chars) + 1), random.choice(alphabet))
        elif op == 1 and len(chars) > 0:
            chars.pop(random.randrange(len(chars)))
        elif op == 2 and len(chars) > 0:
            chars[random.randrange(len(chars))] = random.choice(alphabet)
    return "".join(chars)


class TrainingEditDistanceDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        distance_metric: Type[DistanceMetric],
        char_vectorizer: CharacterEncoder,
        max_length: int,
        sampling_threshold=None,
        bootstrapping_threshold=None,
        max_distance=100,
    ):
        self.texts = texts
        self.distance_metric = distance_metric(texts)
        self.max_length = max_length
        self.char_vectorizer = char_vectorizer
        self.sampling_threshold = sampling_threshold or len(self.char_vectorizer) // 4
        self.bootstrapping_threshold = (
            bootstrapping_threshold or len(self.char_vectorizer) // 2
        )
        self.max_distance = max_distance or self.bootstrapping_threshold * 2

    def __len__(self):
        return len(self.texts) * (
            self.sampling_threshold + self.bootstrapping_threshold
        )

    def __getitem__(self, index):
        idx = index // (self.sampling_threshold + self.bootstrapping_threshold)
        rest = index % (self.sampling_threshold + self.bootstrapping_threshold)
        text = self.texts[idx][: self.max_length]
        if rest < self.sampling_threshold:
            idx2 = random.randint(0, len(self.texts) - 1)
            text2 = self.texts[idx2][: self.max_length]
            d = self.distance_metric.compute_distance(text, text2)
        else:
            _r = (rest - self.sampling_threshold) / self.bootstrapping_threshold
            _d = int(np.random.normal(self.max_distance * _r, 2))
            text2 = modify_text(text, _d)
            d = Levenshtein.distance(text, text2)
        return (
            torch.cat(
                [
                    torch.from_numpy(self.char_vectorizer(text)).unsqueeze(0),
                    torch.from_numpy(self.char_vectorizer(text2)).unsqueeze(0),
                ],
                dim=0,
            ),
            d,
        )


class ValidationEditDistanceDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        distance_metric: Type[DistanceMetric],
        char_vectorizer: CharacterEncoder,
        max_length: int,
        sampling_threshold=None,
        bootstrapping_threshold=None,
        max_distance=100,
    ):
        self.texts = texts
        self.distance_metric = distance_metric(texts)
        self.max_length = max_length
        self.char_vectorizer = char_vectorizer
        self.sampling_threshold = sampling_threshold or len(self.char_vectorizer)
        self.bootstrapping_threshold = (
            bootstrapping_threshold or len(self.char_vectorizer) // 4
        )
        self.max_distance = max_distance or self.bootstrapping_threshold * 2

    def __len__(self):
        return len(self.texts) * (
            self.sampling_threshold + self.bootstrapping_threshold
        )

    def __getitem__(self, index):
        idx = index // (self.sampling_threshold + self.bootstrapping_threshold)
        rest = index % (self.sampling_threshold + self.bootstrapping_threshold)
        text = self.texts[idx][: self.max_length]
        if rest < self.sampling_threshold:
            idx2 = index % self.sampling_threshold
            text2 = self.texts[idx2][: self.max_length]
            d = self.distance_metric.compute_distance(text, text2)
        else:
            _r = (rest - self.sampling_threshold) / self.bootstrapping_threshold
            _d = int(self.max_distance * _r)
            text2 = modify_text(text, _d)
            d = Levenshtein.distance(text, text2)
        return (
            torch.cat(
                [
                    torch.from_numpy(self.char_vectorizer(text)).unsqueeze(0),
                    torch.from_numpy(self.char_vectorizer(text2)).unsqueeze(0),
                ],
                dim=0,
            ),
            d,
        )
