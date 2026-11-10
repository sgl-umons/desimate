import torch
import torch.nn as nn
import torch.nn.functional as F

from desimate.encoders.neuroseed.utils import DISTANCE_TORCH

SUPPORTED_ACTIVATION_MAP = {
    "ReLU",
    "Sigmoid",
    "Tanh",
    "ELU",
    "SELU",
    "GLU",
    "LeakyReLU",
    "Softplus",
    "None",
}


def get_activation(activation):
    """returns the activation function represented by the input string"""
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [
        x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()
    ]
    assert len(activation) == 1 and isinstance(activation[0], str), (
        "Unhandled activation function"
    )
    activation = activation[0]
    if activation.lower() == "none":
        return None
    return vars(torch.nn.modules.activation)[activation]()


class FCLayer(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        activation="relu",
        dropout=0.0,
        b_norm=False,
        bias=True,
        init_fn=None,
    ):
        super(FCLayer, self).__init__()

        self.__params = locals()
        del self.__params["__class__"]
        del self.__params["self"]
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.linear = nn.Linear(in_size, out_size, bias=bias)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(out_size)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_size)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            if h.shape[1] != self.out_size:
                h = self.b_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.b_norm(h)
        return h

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_size)
            + " -> "
            + str(self.out_size)
            + ")"
        )


class MLP(nn.Module):
    """
    Simple multi-layer perceptron, built of a series of FCLayers
    """

    def __init__(
        self,
        in_size,
        hidden_size,
        out_size,
        layers,
        mid_activation="relu",
        last_activation="none",
        dropout=0.0,
        mid_b_norm=False,
        last_b_norm=False,
    ):
        super(MLP, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(
                FCLayer(
                    in_size,
                    out_size,
                    activation=last_activation,
                    b_norm=last_b_norm,
                    dropout=dropout,
                )
            )
        else:
            self.fully_connected.append(
                FCLayer(
                    in_size,
                    hidden_size,
                    activation=mid_activation,
                    b_norm=mid_b_norm,
                    dropout=dropout,
                )
            )
            for _ in range(layers - 2):
                self.fully_connected.append(
                    FCLayer(
                        hidden_size,
                        hidden_size,
                        activation=mid_activation,
                        b_norm=mid_b_norm,
                        dropout=dropout,
                    )
                )
            self.fully_connected.append(
                FCLayer(
                    hidden_size,
                    out_size,
                    activation=last_activation,
                    b_norm=last_b_norm,
                    dropout=dropout,
                )
            )

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_size)
            + " -> "
            + str(self.out_size)
            + ")"
        )


class CNN(nn.Module):
    def __init__(
        self,
        len_sequence,
        embedding_size,
        channels,
        readout_layers=2,
        layers=3,
        dropout=0.0,
        kernel_size=3,
        alphabet_size=4,
        pooling="avg",
        non_linearity=True,
        batch_norm=False,
        stride=1,
    ):
        super(CNN, self).__init__()
        assert pooling == "none" or pooling == "avg" or pooling == "max", (
            "Wrong pooling type"
        )

        self.layers = layers
        self.kernel_size = kernel_size
        self.embedding = nn.Linear(alphabet_size, channels)

        # construct convolutional layers
        self.conv = torch.nn.Sequential()
        for l in range(self.layers):
            self.conv.add_module(
                "conv_" + str(l + 1),
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=stride,
                ),
            )
            len_sequence = (len_sequence - 1) // stride + 1
            if batch_norm:
                self.conv.add_module(
                    "batchnorm_" + str(l + 1), nn.BatchNorm1d(num_features=channels)
                )
            if non_linearity:
                self.conv.add_module("relu_" + str(l + 1), nn.ReLU())

            if pooling == "avg":
                self.conv.add_module(pooling + "_pool_" + str(l + 1), nn.AvgPool1d(2))
                len_sequence //= 2
            elif pooling == "max":
                self.conv.add_module(pooling + "pool_" + str(l + 1), nn.MaxPool1d(2))
                len_sequence //= 2

        # construct readout
        flat_size = channels * len_sequence
        self.readout = MLP(
            in_size=flat_size,
            hidden_size=embedding_size,
            out_size=embedding_size,
            layers=readout_layers,
            mid_activation="relu",
            dropout=dropout,
        )

    def forward(self, sequence):
        (B, N, _) = sequence.shape

        # initial transformation
        sequence = self.embedding(sequence)

        # apply convolutions
        enc_sequence = sequence.transpose(1, 2)
        enc_sequence = self.conv(enc_sequence)

        # flatten and apply readout layer
        enc_sequence = enc_sequence.reshape(B, -1)
        embedding = self.readout(enc_sequence)
        return embedding


class PairEmbeddingDistance(nn.Module):
    def __init__(self, embedding_model, distance="euclidean", scaling=False):
        super(PairEmbeddingDistance, self).__init__()

        self.embedding_model = embedding_model
        self.distance = DISTANCE_TORCH[distance]
        self.distance_str = distance

        self.scaling = None
        if scaling:
            self.radius = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)
            self.scaling = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

    def normalize_embeddings(self, embeddings):
        """Project embeddings to an hypersphere of a certain radius"""
        min_scale = 1e-7

        if self.distance_str == "hyperbolic":
            max_scale = 1 - 1e-3
        else:
            max_scale = 1e10

        return F.normalize(embeddings, p=2, dim=1) * self.radius.clamp_min(
            min_scale
        ).clamp_max(max_scale)

    def encode(self, sequence):
        """Use embedding model and normalization to encode some sequences."""
        enc_sequence = self.embedding_model(sequence)
        if self.scaling is not None:
            enc_sequence = self.normalize_embeddings(enc_sequence)
        return enc_sequence

    def forward(self, sequence):
        # flatten couples
        (B, _, N, _) = sequence.shape
        sequence = sequence.reshape(2 * B, N, -1)

        # encode sequences
        enc_sequence = self.encode(sequence)

        # compute distances
        enc_sequence = enc_sequence.reshape(B, 2, -1)
        distance = self.distance(enc_sequence[:, 0], enc_sequence[:, 1])

        if self.scaling is not None:
            distance = distance * self.scaling

        return distance
