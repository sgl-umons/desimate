from rapidfuzz.distance import Levenshtein

from desimate.distances import DistanceMetric


class LevenshteinDistance(DistanceMetric):
    def __init__(self, texts: list[str]):
        super().__init__(texts)

    def compute_distance(self, i1, i2):
        return Levenshtein.distance(i1, i2)


class NormalizedLevenshteinDistance(DistanceMetric):
    def __init__(self, texts: list[str]):
        super().__init__(texts)

    def compute_distance(self, i1, i2):
        return Levenshtein.distance(i1, i2) * 100 / max(len(i1), len(i2))
