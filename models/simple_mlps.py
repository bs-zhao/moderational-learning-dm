import torch
import torch.nn as nn

class simple_mlp1(nn.Module):
    """decoder
 latent_dim + x_dim y_dim"""

    def __init__(self, x_dim: int, latent_dim: int, y_dim: int):
        super().__init__()
        input_dim = latent_dim + x_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, y_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class simple_mlp(nn.Module):
    def __init__(self, x_dim: int, y_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, y_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class simple_mlp1_batch(nn.Module):
    """simple_mlp1
 - x [B, T, x_dim]
 - forward trial_mask [B, T]bool 0/1mask=1 0
 - [B, T, y_dim]"""

    def __init__(self, x_dim: int, y_dim: int):
        super().__init__()
        self.y_dim = y_dim

        self.model = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, y_dim),
        )

    def forward(self, x: torch.Tensor, trial_mask: torch.Tensor) -> torch.Tensor:
        """Args:
 x: [B, T, input_dim]
 trial_mask: [B, T]True/1 trial
 Returns:
 y: [B, T, y_dim]"""
        assert x.dim() == 3, f"x shape must be [B, T, D], got {tuple(x.shape)}"
        assert trial_mask.dim() == 2, f"trial_mask shape must be [B, T], got {tuple(trial_mask.shape)}"
        B, T, _ = x.shape

        y = torch.zeros(B, T, self.y_dim, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(B * T, -1)
        mask_flat = trial_mask.reshape(B * T)
        if mask_flat.dtype != torch.bool:
            mask_flat = mask_flat != 0

        if mask_flat.any():
            idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
            x_valid = x_flat.index_select(0, idx)
            y_valid = self.model(x_valid)
            y.view(B * T, self.y_dim).index_copy_(0, idx, y_valid)

        return y
