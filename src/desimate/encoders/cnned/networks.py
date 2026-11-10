import torch
import torch.nn as nn
import torch.nn.functional as F

POOL = nn.AvgPool1d


class TwoLayerCNN(nn.Module):
    def __init__(self, C, M, embedding, channel, mtc_input):
        super(TwoLayerCNN, self).__init__()
        self.C = C
        self.M = M
        self.embedding = embedding
        self.mtc_input = C if mtc_input else 1

        self.conv1 = nn.Conv1d(self.mtc_input, channel, 3, 1, padding=1, bias=False)
        self.flat_size = M // 2 * C // self.mtc_input * channel
        self.fc1 = nn.Linear(self.flat_size, embedding)

    def forward(self, x):
        N = len(x)
        x = x.view(-1, self.mtc_input, self.M)

        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)

        x = x.view(N, self.flat_size)
        x = self.fc1(x)

        return x


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        x1, x2, x3 = x
        return self.embedding_net(x1), self.embedding_net(x2), self.embedding_net(x3)


class AdaptativeTripletLoss(nn.Module):
    def __init__(self, epochs):
        super(AdaptativeTripletLoss, self).__init__()
        self.l, self.r = 1, 1
        step = epochs // 5
        self.Ls = {
            step * 0: (0, 10),
            step * 1: (10, 10),
            step * 2: (10, 1),
            step * 3: (5, 0.1),
            step * 4: (1, 0.01),
        }

    def dist(self, ins, pos):
        return torch.linalg.norm(ins - pos, dim=1)

    def forward(self, x, lens, dists, epoch):
        if epoch in self.Ls:
            self.l, self.r = self.Ls[epoch]
        anchor, positive, negative = x
        pos_dist, neg_dist, pos_neg_dist = (d.type(torch.float32) for d in dists)

        pos_embed_dist = self.dist(anchor, positive)
        neg_embed_dist = self.dist(anchor, negative)
        pos_neg_embed_dist = self.dist(positive, negative)

        threshold = neg_dist - pos_dist
        rank_loss = F.relu(pos_embed_dist - neg_embed_dist + threshold)

        mse_loss = (
            (pos_embed_dist - pos_dist) ** 2
            + (neg_embed_dist - neg_dist) ** 2
            + (pos_neg_embed_dist - pos_neg_dist) ** 2
        )

        return (
            torch.mean(rank_loss),
            torch.mean(mse_loss),
            torch.mean(
                self.l * rank_loss
                + self.r * torch.sqrt(torch.clamp_min(mse_loss, 1e-8))
            ),  # clamp_min due to NaN values
        )


class FixedTripletLoss(nn.Module):
    def __init__(self):
        super(FixedTripletLoss, self).__init__()
        self.l, self.r = 1, 1

    def dist(self, ins, pos):
        return torch.linalg.norm(ins - pos, dim=1)

    def forward(self, x, lens, dists, epoch):
        anchor, positive, negative = x
        pos_dist, neg_dist, pos_neg_dist = (d.type(torch.float32) for d in dists)

        pos_embed_dist = self.dist(anchor, positive)
        neg_embed_dist = self.dist(anchor, negative)
        pos_neg_embed_dist = self.dist(positive, negative)

        threshold = neg_dist - pos_dist
        rank_loss = F.relu(pos_embed_dist - neg_embed_dist + threshold)

        mse_loss = (
            (pos_embed_dist - pos_dist) ** 2
            + (neg_embed_dist - neg_dist) ** 2
            + (pos_neg_embed_dist - pos_neg_dist) ** 2
        )

        return (
            torch.mean(rank_loss),
            torch.mean(mse_loss),
            torch.mean(
                self.l * rank_loss
                + self.r * torch.sqrt(torch.clamp_min(mse_loss, 1e-8))
            ),  # clamp_min due to NaN values
        )
