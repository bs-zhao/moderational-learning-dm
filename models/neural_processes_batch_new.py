import torch
import torch.nn as nn
from mlp import MLP
from set_transformer import SetTransformer
from aggregator import deep_set_mask, deep_set2_mask

class basicNP_D_outer(nn.Module):
    def __init__(self, x_dim, y_dim=1, outer_dim=1, emb_size=32, latent_dim=16, with_outer=True):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.emb_size = emb_size
        self.latent_dim = latent_dim
        self.with_outer = with_outer

        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, emb_size),
        )

        self.aggregator = deep_set_mask(middle_dim=emb_size)

        if self.with_outer:
            self.encoder_outer = nn.Sequential(
                nn.Linear(outer_dim, 128),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(128, emb_size),
            )
            self.aggregator_outer = deep_set_mask(middle_dim=emb_size)

        if self.with_outer:
            self.fc_mu = nn.Linear(emb_size * 2, latent_dim)
            self.fc_log_var = nn.Linear(emb_size * 2, latent_dim)
        else:
            self.fc_mu = nn.Linear(emb_size, latent_dim)
            self.fc_log_var = nn.Linear(emb_size, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + x_dim, 128),
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

        if self.with_outer:
            self.ln_inner = nn.LayerNorm(emb_size)
            self.ln_outer = nn.LayerNorm(emb_size)
            self.outer_gate_param = nn.Parameter(torch.tensor(-1.0))
            self.dropout_outer = nn.Dropout(0.2)

    def reparameterize(self, mu, log_var):
        log_var = torch.clamp(log_var, min=-20, max=20)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, XY_context, XY_target, XY_outer=None, trial_mask_context=None, trial_mask_target=None, trial_mask_outer=None, mu_given=None):

        batch_size, n_trials_target, _ = XY_target.size()
        X_target = XY_target[:, :, :self.x_dim]

        if mu_given is None:
            encoded_inner = self.encoder(XY_context)
            aggregated_inner = self.aggregator(encoded_inner, trial_mask_context)

            if self.with_outer:
                encoded_outer = self.encoder_outer(XY_outer)
                aggregated_outer = self.aggregator_outer(encoded_outer, trial_mask_outer)

                aggregated_inner = self.ln_inner(aggregated_inner)
                aggregated_outer = self.ln_outer(aggregated_outer)
                aggregated_outer = self.dropout_outer(aggregated_outer)

                outer_gate = torch.sigmoid(self.outer_gate_param)
                aggregated = torch.cat([aggregated_inner, outer_gate * aggregated_outer], dim=1)
            else:
                aggregated = aggregated_inner

            mu = self.fc_mu(aggregated)
            log_var = self.fc_log_var(aggregated)

            z = self.reparameterize(mu, log_var)
            z_repeat = z.unsqueeze(1).expand(batch_size, n_trials_target, -1)
        else:
            mu = mu_given
            log_var = torch.zeros_like(mu)
            z_repeat = mu_given.unsqueeze(1).expand(batch_size, n_trials_target, -1)

        X_expanded = torch.cat([X_target, z_repeat], dim=-1)
        Yh = self.decoder(X_expanded)

        if trial_mask_target is not None:
            pad_mask = ~trial_mask_target
            Yh = Yh.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return Yh, mu, log_var

class basicNP_D2_outer(nn.Module):
    def __init__(self, x_dim, outer_dim=1, emb_size=32, latent_dim=16, y_dim=1, with_outer=True):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.emb_size = emb_size
        self.latent_dim = latent_dim
        self.with_outer = with_outer

        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, emb_size),
        )

        self.aggregator = deep_set2_mask(
            summary_dim=emb_size,
            depth=2,
            inner_pooling="mean",
            output_pooling="mean",
            mlp_widths_equivariant=(128, 64),
            mlp_widths_invariant_inner=(64, 64),
            mlp_widths_invariant_outer=(64, 64),
            mlp_widths_invariant_last=(64, 64),
            activation="silu",
            dropout=0.05,
            spectral_normalization=False,
        )

        if self.with_outer:
            self.encoder_outer = nn.Sequential(
                nn.Linear(outer_dim, 64),
                nn.LeakyReLU(0.01),
                nn.Linear(64, emb_size),
            )
            self.aggregator_outer = deep_set2_mask(
                summary_dim=emb_size,
                depth=2,
                inner_pooling="mean",
                output_pooling="mean",
                mlp_widths_equivariant=(128, 64),
                mlp_widths_invariant_inner=(64, 64),
                mlp_widths_invariant_outer=(64, 64),
                mlp_widths_invariant_last=(64, 64),
                activation="silu",
                dropout=0.05,
                spectral_normalization=False,
            )

        if self.with_outer:
            self.fc_mu = nn.Linear(emb_size * 2, latent_dim)
            self.fc_log_var = nn.Linear(emb_size * 2, latent_dim)
        else:
            self.fc_mu = nn.Linear(emb_size, latent_dim)
            self.fc_log_var = nn.Linear(emb_size, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + x_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, y_dim),
        )

        if self.with_outer:
            self.ln_inner = nn.LayerNorm(emb_size)
            self.ln_outer = nn.LayerNorm(emb_size)
            self.outer_gate_param = nn.Parameter(torch.tensor(-1.0))

    def reparameterize(self, mu, log_var):
        log_var = torch.clamp(log_var, min=-20, max=20)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        XY_context,
        XY_target,
        XY_outer=None,
        trial_mask_context=None,
        trial_mask_target=None,
        trial_mask_outer=None,
        mu_given=None,
    ):
        batch_size, n_trials_target, _ = XY_target.size()
        X_target = XY_target[:, :, :self.x_dim]

        if mu_given is None:

            encoded_inner = self.encoder(XY_context)
            aggregated_inner = self.aggregator(encoded_inner, trial_mask_context)

            if self.with_outer and XY_outer is not None:
                encoded_outer = self.encoder_outer(XY_outer)
                aggregated_outer = self.aggregator_outer(encoded_outer, trial_mask_outer)

                outer_gate = torch.sigmoid(self.outer_gate_param)
                aggregated = torch.cat([aggregated_inner, outer_gate * aggregated_outer], dim=1)

                aggregated = torch.cat([aggregated_inner, aggregated_outer], dim=1)
            else:
                aggregated = aggregated_inner

            mu = self.fc_mu(aggregated)
            log_var = self.fc_log_var(aggregated)
            z = self.reparameterize(mu, log_var)
            z_repeat = z.unsqueeze(1).expand(batch_size, n_trials_target, -1)
        else:
            mu = mu_given
            log_var = torch.zeros_like(mu)
            z_repeat = mu_given.unsqueeze(1).expand(batch_size, n_trials_target, -1)

        X_expanded = torch.cat([X_target, z_repeat], dim=-1)
        Yh = self.decoder(X_expanded)

        if trial_mask_target is not None:
            pad_mask = ~trial_mask_target
            Yh = Yh.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return Yh, mu, log_var
