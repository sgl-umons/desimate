from random import randint

import numpy as np
from torch.utils.data import Dataset

from desimate.encoders.datasets import StringDataset


class TripletString(Dataset):
    def __init__(
        self,
        texts: list[str],
        text_sigs_dataset: StringDataset,
        knn: np.ndarray,
        dist: np.ndarray,
    ):
        self.texts = texts
        self.text_sigs_dataset = text_sigs_dataset
        self.knn = knn
        self.dist = dist
        # self.avg_dist = np.mean(self.dist)
        self.lens = [len(_s) for _s in self.text_sigs_dataset.texts]

    def __getitem__(self, idx):
        _n = min(len(self.text_sigs_dataset) - 1, self.knn.shape[1] - 1)
        pos = self.knn[idx, randint(1, _n)]
        neg = self.knn[idx, randint(1, _n)]
        pos_dist = self.dist[idx, pos]
        neg_dist = self.dist[idx, neg]
        if pos_dist > neg_dist:
            pos, neg = neg, pos
            pos_dist, neg_dist = neg_dist, pos_dist
        pos_neg_dist = self.dist[pos, neg]
        return (
            self.text_sigs_dataset[idx],
            self.text_sigs_dataset[pos],
            self.text_sigs_dataset[neg],
            self.lens[idx],
            self.lens[pos],
            self.lens[neg],
            pos_dist,
            neg_dist,
            pos_neg_dist,
        )

    def __len__(self):
        return len(self.text_sigs_dataset)
