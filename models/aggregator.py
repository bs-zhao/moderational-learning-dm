import torch
import torch.nn as nn
from collections.abc import Sequence

class pool_aggregator(nn.Module):
    def __init__(self, middle_dim=64):
        super(pool_aggregator, self).__init__()

        self.pool = nn.AdaptiveAvgPool1d(middle_dim)
        self.apool = nn.Sequential(
            nn.Linear(middle_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),

            )

    def forward(self, x):

        x = x.permute(0, 2, 1)

        aggregated = self.pool(x).squeeze(-1)
        aggregated = self.apool(aggregated).squeeze(-1)
        return aggregated

class deep_set(nn.Module):
    def __init__(self, middle_dim=64):
        super(deep_set, self).__init__()

        self.middle_dim = middle_dim

        self.phi = nn.Sequential(
            nn.LazyLinear(middle_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(middle_dim, middle_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.rho = None

    def forward(self, x):

        batch_size, n_trials, input_dim = x.shape

        h = x.reshape(batch_size * n_trials, input_dim)
        h = self.phi(h)
        h = h.view(batch_size, n_trials, self.middle_dim)

        h_agg = h.mean(dim=1)

        if self.rho is None:
            self.rho = nn.Sequential(
                nn.Linear(self.middle_dim, 128),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(128, 64),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(64, input_dim)
            ).to(x.device)

        out = self.rho(h_agg)
        return out

def _get_activation(activation: str) -> nn.Module:
    act = activation.lower() if isinstance(activation, str) else activation
    if act == "gelu":
        return nn.GELU()
    if act == "silu" or act == "swish":
        return nn.SiLU()
    if act == "relu":
        return nn.ReLU()
    if act == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if act == "mish":
        return nn.Mish()
    return nn.SiLU()

def _maybe_spectral_norm(linear: nn.Linear, spectral_normalization: bool) -> nn.Module:
    if spectral_normalization:
        return nn.utils.spectral_norm(linear)
    return linear

def _init_linear(linear: nn.Linear, kernel_initializer: str | None):
    if kernel_initializer in (None, "he_normal", "kaiming_normal"):
        nn.init.kaiming_normal_(linear.weight, nonlinearity="linear")
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

class _MLP(nn.Module):
    def __init__(
        self,
        widths: Sequence[int],
        *,
        activation: str = "silu",
        kernel_initializer: str = "he_normal",
        dropout: float | None = 0.05,
        spectral_normalization: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        act_layer = _get_activation(activation)
        for i, width in enumerate(widths):
            linear = nn.Linear(widths[i - 1] if i > 0 else widths[0], width)
            _init_linear(linear, kernel_initializer)
            linear = _maybe_spectral_norm(linear, spectral_normalization)
            layers.append(linear)
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(p=dropout))

            layers.append(_get_activation(activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class _InvariantLayer(nn.Module):
    def __init__(
        self,
        mlp_widths_inner: Sequence[int] = (64, 64),
        mlp_widths_outer: Sequence[int] = (64, 64),
        activation: str = "silu",
        kernel_initializer: str = "he_normal",
        dropout: float | None = 0.05,
        pooling: str = "mean",
        spectral_normalization: bool = False,
    ):
        super().__init__()
        self.pooling = pooling
        self.inner_fc = _MLP(
            mlp_widths_inner,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            spectral_normalization=spectral_normalization,
        )
        self.inner_projector = nn.Linear(mlp_widths_inner[-1], mlp_widths_inner[-1])
        _init_linear(self.inner_projector, kernel_initializer)

        self.outer_fc = _MLP(
            mlp_widths_outer,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            spectral_normalization=spectral_normalization,
        )
        self.outer_projector = nn.Linear(mlp_widths_outer[-1], mlp_widths_outer[-1])
        _init_linear(self.outer_projector, kernel_initializer)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:

        if self.pooling == "mean":
            return x.mean(dim=-2)
        if self.pooling == "sum":
            return x.sum(dim=-2)
        if self.pooling == "max":
            return x.max(dim=-2).values

        return x.mean(dim=-2)

    def forward(self, input_set: torch.Tensor) -> torch.Tensor:

        in_shape = input_set.shape
        features_dim = in_shape[-1]
        x = input_set.reshape(-1, features_dim)
        x = self.inner_fc(x)
        x = self.inner_projector(x)
        x = x.reshape(*in_shape[:-1], -1)
        x = self._pool(x)

        out_feat = x.shape[-1]
        x = x.reshape(-1, out_feat)
        x = self.outer_fc(x)
        x = self.outer_projector(x)
        x = x.reshape(*in_shape[:-2], -1)
        return x

class _EquivariantLayer(nn.Module):
    def __init__(
        self,
        mlp_widths_equivariant: Sequence[int] = (64, 64),
        mlp_widths_invariant_inner: Sequence[int] = (64, 64),
        mlp_widths_invariant_outer: Sequence[int] = (64, 64),
        pooling: str = "mean",
        activation: str = "silu",
        kernel_initializer: str = "he_normal",
        dropout: float | None = 0.05,
        layer_norm: bool = True,
        spectral_normalization: bool = False,
    ):
        super().__init__()

        self.invariant_module = _InvariantLayer(
            mlp_widths_inner=mlp_widths_invariant_inner,
            mlp_widths_outer=mlp_widths_invariant_outer,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            pooling=pooling,
            spectral_normalization=spectral_normalization,
        )

        rep_dim = mlp_widths_equivariant[-1]
        self.input_projector = nn.Linear(rep_dim, rep_dim)
        _init_linear(self.input_projector, kernel_initializer)

        self.equivariant_fc = _MLP(
            mlp_widths_equivariant,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            spectral_normalization=spectral_normalization,
        )
        self.out_fc_projector = nn.Linear(rep_dim, rep_dim)
        _init_linear(self.out_fc_projector, kernel_initializer)

        self.layer_norm = nn.LayerNorm(rep_dim) if layer_norm else None

    def forward(self, input_set: torch.Tensor) -> torch.Tensor:

        in_shape = input_set.shape
        feat = in_shape[-1]
        set_size = in_shape[-2]

        x = input_set.reshape(-1, feat)

        if self.input_projector.in_features != feat:
            self.input_projector = nn.Linear(feat, self.input_projector.out_features).to(input_set.device)
            _init_linear(self.input_projector, "he_normal")
        x = self.input_projector(x)
        x = x.reshape(*in_shape[:-1], -1)

        inv = self.invariant_module(x)
        inv = inv.unsqueeze(-2).expand(*in_shape[:-2], set_size, inv.shape[-1])

        concat = torch.cat([x, inv], dim=-1)
        c_shape = concat.shape
        concat = concat.reshape(-1, c_shape[-1])
        out_fc = self.equivariant_fc(concat)
        out_fc = self.out_fc_projector(out_fc)
        out_fc = out_fc.reshape(*c_shape[:-1], -1)

        output_set = x + out_fc
        if self.layer_norm is not None:
            output_set = self.layer_norm(output_set)
        return output_set

class deep_set2(nn.Module):
    def __init__(
        self,
        summary_dim: int = 16,
        depth: int = 2,
        inner_pooling: str = "mean",
        output_pooling: str = "mean",
        mlp_widths_equivariant: Sequence[int] = (64, 64),
        mlp_widths_invariant_inner: Sequence[int] = (64, 64),
        mlp_widths_invariant_outer: Sequence[int] = (64, 64),
        mlp_widths_invariant_last: Sequence[int] = (64, 64),
        activation: str = "silu",
        kernel_initializer: str = "he_normal",
        dropout: float | None = 0.05,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.equivariant_modules = nn.ModuleList()
        for _ in range(depth):
            self.equivariant_modules.append(
                _EquivariantLayer(
                    mlp_widths_equivariant=mlp_widths_equivariant,
                    mlp_widths_invariant_inner=mlp_widths_invariant_inner,
                    mlp_widths_invariant_outer=mlp_widths_invariant_outer,
                    pooling=inner_pooling,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    dropout=dropout,
                    spectral_normalization=spectral_normalization,
                )
            )

        self.invariant_module = _InvariantLayer(
            mlp_widths_inner=mlp_widths_invariant_last,
            mlp_widths_outer=mlp_widths_invariant_last,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            pooling=output_pooling,
            spectral_normalization=spectral_normalization,
        )

        rep_dim = mlp_widths_invariant_last[-1]
        self.output_projector = nn.Linear(rep_dim, summary_dim)
        _init_linear(self.output_projector, kernel_initializer)
        self.summary_dim = summary_dim

    def forward(self, x: torch.Tensor, training: bool = False, **kwargs) -> torch.Tensor:

        for em in self.equivariant_modules:
            x = em(x)
        x = self.invariant_module(x)
        x = self.output_projector(x)
        return x

class deep_set_mask(nn.Module):
    def __init__(self, middle_dim=64):
        super(deep_set_mask, self).__init__()

        self.middle_dim = middle_dim

        self.phi = nn.Sequential(
            nn.LazyLinear(middle_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(middle_dim, middle_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.rho = None

    def forward(self, x, trial_mask: torch.Tensor | None = None):

        batch_size, n_trials, input_dim = x.shape

        h = x.reshape(batch_size * n_trials, input_dim)
        h = self.phi(h)
        h = h.view(batch_size, n_trials, self.middle_dim)

        if trial_mask is not None:
            mask = trial_mask.unsqueeze(-1).to(h.dtype)
            h = h * mask
            counts = mask.sum(dim=1).clamp(min=1.0)
            h_agg = h.sum(dim=1) / counts
        else:
            h_agg = h.mean(dim=1)

        if self.rho is None:
            self.rho = nn.Sequential(
                nn.Linear(self.middle_dim, 128),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(128, 64),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(64, input_dim)
            ).to(x.device)

        out = self.rho(h_agg)
        return out

class deep_set2_mask(nn.Module):
    def __init__(
        self,
        summary_dim: int = 16,
        depth: int = 2,
        inner_pooling: str = "mean",
        output_pooling: str = "mean",
        mlp_widths_equivariant: Sequence[int] = (64, 64),
        mlp_widths_invariant_inner: Sequence[int] = (64, 64),
        mlp_widths_invariant_outer: Sequence[int] = (64, 64),
        mlp_widths_invariant_last: Sequence[int] = (64, 64),
        activation: str = "silu",
        kernel_initializer: str = "he_normal",
        dropout: float | None = 0.05,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.equivariant_modules = nn.ModuleList()
        for _ in range(depth):
            self.equivariant_modules.append(
                _EquivariantLayer(
                    mlp_widths_equivariant=mlp_widths_equivariant,
                    mlp_widths_invariant_inner=mlp_widths_invariant_inner,
                    mlp_widths_invariant_outer=mlp_widths_invariant_outer,
                    pooling=inner_pooling,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    dropout=dropout,
                    spectral_normalization=spectral_normalization,
                )
            )

        self.invariant_module = _InvariantLayer(
            mlp_widths_inner=mlp_widths_invariant_last,
            mlp_widths_outer=mlp_widths_invariant_last,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            pooling=output_pooling,
            spectral_normalization=spectral_normalization,
        )

        rep_dim = mlp_widths_invariant_last[-1]
        self.output_projector = nn.Linear(rep_dim, summary_dim)
        _init_linear(self.output_projector, kernel_initializer)
        self.summary_dim = summary_dim

    def forward(self, x: torch.Tensor, trial_mask: torch.Tensor | None = None) -> torch.Tensor:

        for em in self.equivariant_modules:
            x = em(x)

        if trial_mask is not None:

            mask = trial_mask.unsqueeze(-1).to(x.dtype)
            x = x * mask
            counts = mask.sum(dim=1).clamp(min=1.0)
            x = x.sum(dim=1) / counts
        else:
            x = self.invariant_module(x)

        x = self.output_projector(x)
        return x
