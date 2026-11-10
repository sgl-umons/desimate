import torch
import torch.nn as nn
import numpy as np


class AverageMeter(object):
    def __init__(self, len_tuple=0):
        self.len_tuple = len_tuple
        self.avg = 0 if len_tuple == 0 else tuple([0] * len_tuple)
        self.sum = 0 if len_tuple == 0 else [0] * len_tuple
        self.count = 0

    def update(self, val, n=1):
        self.count += n
        if self.len_tuple == 0:
            self.sum += (val.data.item() if torch.is_tensor(val) else val) * n
            self.avg = self.sum / self.count
        else:
            self.sum = [
                self.sum[i]
                + (val[i].data.item() if torch.is_tensor(val) else val[i]) * n
                for i in range(self.len_tuple)
            ]
            self.avg = tuple(s / self.count for s in self.sum)

    def get_average(self):
        return self.avg


def square_distance(t1_emb, t2_emb):
    D = t1_emb - t2_emb
    d = torch.sum(D * D, dim=-1)
    return d


def euclidean_distance(t1_emb, t2_emb):
    D = t1_emb - t2_emb
    d = torch.norm(D, dim=-1)
    return d


def cosine_distance(t1_emb, t2_emb):
    return 1 - nn.functional.cosine_similarity(t1_emb, t2_emb, dim=-1, eps=1e-6)


def manhattan_distance(t1_emb, t2_emb):
    D = t1_emb - t2_emb
    d = torch.sum(torch.abs(D), dim=-1)
    return d


def hyperbolic_distance(
    u, v, epsilon=1e-7
):  # changed from epsilon=1e-7 to reduce error
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u**2, dim=-1)
    sqvnorm = torch.sum(v**2, dim=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = torch.sqrt(x**2 - 1)
    return torch.log(x + z)


def hyperbolic_distance_numpy(u, v, epsilon=1e-9):
    sqdist = np.sum((u - v) ** 2, axis=-1)
    squnorm = np.sum(u**2, axis=-1)
    sqvnorm = np.sum(v**2, axis=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = np.sqrt(x**2 - 1)
    return np.log(x + z)


DISTANCE_TORCH = {
    "square": square_distance,
    "euclidean": euclidean_distance,
    "cosine": cosine_distance,
    "manhattan": manhattan_distance,
    "hyperbolic": hyperbolic_distance,
}


def mape(pred, label, eps=1e-6):
    return torch.mean(torch.abs(pred - label) / (label + eps))
