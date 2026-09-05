import torch
import torch.nn as nn
from torch import Tensor

class Residual(nn.Module):
    """Residual connection wrapper for PyTorch modules."""

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)

class MLP(nn.Module):
    """
    Implements a simple configurable MLP with optional residual connections and dropout.

    If used in conjunction with a coupling net, a diffusion model, or a flow matching model, it assumes
    that the input and conditions are already concatenated (i.e., this is a single-input model).
    """

    def __init__(
        self,
        widths: tuple,
        *,
        activation: str = "mish",
        kernel_initializer: str = "he_normal",
        residual: bool = True,
        dropout: float = 0.05,
        norm: str = None,
        spectral_normalization: bool = False,
        use_bias: bool = True,
        **kwargs,
    ):
        """
        Implements a flexible multi-layer perceptron (MLP) with optional residual connections, dropout, and
        spectral normalization.

        This MLP can be used as a general-purpose feature extractor or function approximator, supporting configurable
        depth, width, activation functions, and weight initializations.

        If `residual` is enabled, each layer includes a skip connection for improved gradient flow. The model also
        supports dropout for regularization and spectral normalization for stability in learning smooth functions.

        Parameters
        ----------
        widths : tuple[int]
            Defines the number of units in each layer, including input and output dimensions.
        activation : str, optional
            Activation function applied in the hidden layers, such as "mish". Default is "mish".
        kernel_initializer : str, optional
            Initialization strategy for kernel weights, such as "he_normal". Default is "he_normal".
        residual : bool, optional
            Whether to use residual connections for improved training stability. Default is True.
        dropout : float or None, optional
            Dropout rate applied within the MLP layers for regularization. Default is 0.05.
        norm: str, optional
            Normalization type: "batch", "layer", or None. Default is None.
        spectral_normalization : bool, optional
            Whether to apply spectral normalization to stabilize training. Default is False.
        use_bias : bool, optional
            Whether to use bias in linear layers. Default is True.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__()

        self.widths = list(widths)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.residual = residual
        self.dropout = dropout
        self.norm = norm
        self.spectral_normalization = spectral_normalization
        self.use_bias = use_bias

        layers = []

        for i in range(len(widths) - 1):
            input_dim = widths[i]
            output_dim = widths[i + 1]

            is_hidden_layer = (i > 0 and i < len(widths) - 2)
            can_use_residual = (input_dim == output_dim and residual and is_hidden_layer)

            if can_use_residual:
                block = self._make_residual_block(
                    input_dim, output_dim, activation, kernel_initializer,
                    dropout, norm, spectral_normalization, use_bias
                )
                layers.append(block)
            else:
                block = self._make_basic_block(
                    input_dim, output_dim, activation, kernel_initializer,
                    dropout, norm, spectral_normalization, use_bias,
                    is_last_layer=(i == len(widths) - 2)
                )
                layers.append(block)

        self.layers = nn.Sequential(*layers)

    def _make_basic_block(self, input_dim, output_dim, activation, kernel_initializer,
                         dropout, norm, spectral_normalization, use_bias, is_last_layer=False):
        """Create a basic linear block without residual connection."""
        layers = []

        dense = nn.Linear(input_dim, output_dim, bias=use_bias)
        self._init_weights(dense, kernel_initializer)

        if spectral_normalization:
            dense = nn.utils.spectral_norm(dense)

        layers.append(dense)

        if not is_last_layer:

            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))

            layers.append(self._get_activation(activation))

            if norm == "batch":
                layers.append(nn.BatchNorm1d(output_dim))
            elif norm == "layer":
                layers.append(nn.LayerNorm(output_dim))

        return nn.Sequential(*layers)

    def _make_residual_block(self, input_dim, output_dim, activation, kernel_initializer,
                           dropout, norm, spectral_normalization, use_bias):
        """Create a residual block."""
        layers = []

        dense = nn.Linear(input_dim, output_dim, bias=use_bias)
        self._init_weights(dense, kernel_initializer)

        if spectral_normalization:
            dense = nn.utils.spectral_norm(dense)

        layers.append(dense)

        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(self._get_activation(activation))

        if norm == "batch":
            layers.append(nn.BatchNorm1d(output_dim))
        elif norm == "layer":
            layers.append(nn.LayerNorm(output_dim))

        return Residual(*layers)

    def _get_activation(self, activation):
        """Get activation function."""
        if activation == "mish":
            return nn.Mish()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "elu":
            return nn.ELU()
        elif activation == "silu":
            return nn.SiLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:

            return nn.ReLU()

    def _init_weights(self, layer, kernel_initializer):
        """Initialize weights of a layer."""
        if kernel_initializer == "he_normal":
            nn.init.kaiming_normal_(layer.weight)
        elif kernel_initializer == "lecun_normal":
            nn.init.normal_(layer.weight, 0, (1.0 / layer.in_features) ** 0.5)
        elif kernel_initializer == "glorot_uniform":
            nn.init.xavier_uniform_(layer.weight)
        elif kernel_initializer == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight)

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
