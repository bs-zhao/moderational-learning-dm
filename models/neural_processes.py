import torch
import torch.nn as nn
from mlp import MLP
from set_transformer import SetTransformer
from aggregator import pool_aggregator, deep_set, deep_set2

class basicNP(nn.Module):
    def __init__(self, x_dim, emb_size=32, latent_dim=16, y_dim=1,
                 mlp_activation="mish", kernel_initializer="he_normal", dropout=0.05, use_bias=True):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.emb_size = emb_size
        self.latent_dim = latent_dim

        xy_dim = x_dim + y_dim
        self.encoder = MLP(
            widths=[xy_dim, max(xy_dim*2, 128), max(emb_size*2, 128), emb_size],
            activation=mlp_activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            use_bias=use_bias,
            residual=False
        )

        self.aggragator_trial = SetTransformer(
            input_dim=None,
            embed_dims=(emb_size, emb_size),
            num_heads=(4, 4),
            mlp_depths=(2, 2),
            mlp_widths=(128, 128),
            seed_dim=emb_size,
            num_seeds=1,
            summary_dim=emb_size,
        )
        self.fc_mu = nn.Linear(emb_size, latent_dim)
        self.fc_log_var = nn.Linear(emb_size, latent_dim)

        self.decoder = MLP(
            widths=[latent_dim + x_dim, max((latent_dim + x_dim)*2, 128), max(y_dim*2, 128), y_dim],
            activation=mlp_activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            use_bias=use_bias,
            residual=False
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, XY_context, XY_target, trial_mask, mu_given=0, mode=0):

        batch_size = XY_target.size(0)
        n_trials_target = XY_target.size(1)
        X_target = XY_target[:, :, :self.x_dim]
        key_padding_mask = ~trial_mask

        if mode == 0:
            encoded = self.encoder(XY_context)
            aggregated = self.aggragator_trial(encoded, key_padding_mask=key_padding_mask)

            mu = self.fc_mu(aggregated)
            log_var = self.fc_log_var(aggregated)

            z = self.reparameterize(mu, log_var)

            condition = z.unsqueeze(1).repeat(1, n_trials_target, 1)

        elif mode == 1:
            mu = mu_given
            log_var = torch.zeros_like(mu_given)

            condition = mu_given.unsqueeze(1).repeat(1, n_trials_target, 1)

        X_expanded = torch.cat([X_target, condition], dim=-1)
        Yh = self.decoder(X_expanded)

        return Yh, mu, log_var

class oldBP(nn.Module):
    def __init__(self, x_dim, emb_size=32, latent_dim=16, y_dim=1):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, emb_size),

        )

        self.aggregator = pool_aggregator(middle_dim=64)

        self.fc_mu = nn.Linear(emb_size, latent_dim)
        self.fc_log_var = nn.Linear(emb_size, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim+x_dim, 128),
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

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).
        """

        log_var = torch.clamp(log_var, min=-20, max=20)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, XY_context, XY_target, mu_given=None):

        n_trials_target = XY_target.size(1)
        X_target = XY_target[0,:,:-1]

        if mu_given is None:
            encoded = self.encoder(XY_context)
            aggregated = self.aggregator(encoded)

            mu = self.fc_mu(aggregated)
            log_var = self.fc_log_var(aggregated)

            z = self.reparameterize(mu, log_var)
            z_repeat = z.repeat(n_trials_target, 1)

        else:
            mu = mu_given
            log_var = 0
            z_repeat = mu_given.repeat(n_trials_target, 1)

        X_expanded = torch.cat([X_target, z_repeat], dim=1)
        Yh = self.decoder(X_expanded)

        return Yh, mu, log_var

class oldBP_T(nn.Module):
    def __init__(self, x_dim, emb_size=32, latent_dim=16, y_dim=1):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, emb_size),

        )

        self.aggregator = SetTransformer(
            input_dim=None,
            embed_dims=(emb_size, emb_size),
            num_heads=(4, 4),
            mlp_depths=(2, 2),
            mlp_widths=(128, 128),
            seed_dim=emb_size,
            num_seeds=1,
            summary_dim=emb_size,
        )

        self.fc_mu = nn.Linear(emb_size, latent_dim)
        self.fc_log_var = nn.Linear(emb_size, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + x_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, y_dim),
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).
        """

        log_var = torch.clamp(log_var, min=-20, max=20)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, XY_context, XY_target, mu_given=None):

        n_trials_target = XY_target.size(1)
        X_target = XY_target[0,:,:-1]

        if mu_given is None:
            encoded = self.encoder(XY_context)
            aggregated = self.aggregator(encoded)

            mu = self.fc_mu(aggregated)
            log_var = self.fc_log_var(aggregated)

            z = self.reparameterize(mu, log_var)
            z_repeat = z.repeat(n_trials_target, 1)

        else:
            mu = mu_given
            log_var = 0
            z_repeat = mu_given.repeat(n_trials_target, 1)

        X_expanded = torch.cat([X_target, z_repeat], dim=1)
        Yh = self.decoder(X_expanded)

        return Yh, mu, log_var

class oldBP_D(nn.Module):
    def __init__(self, x_dim, emb_size=32, latent_dim=16, y_dim=1):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, emb_size),

        )

        self.aggregator = deep_set(middle_dim=emb_size)

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

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).
        """

        log_var = torch.clamp(log_var, min=-20, max=20)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, XY_context, XY_target, mu_given=None):

        n_trials_target = XY_target.size(1)
        X_target = XY_target[0, :, :-1]

        if mu_given is None:
            encoded = self.encoder(XY_context)
            aggregated = self.aggregator(encoded)

            mu = self.fc_mu(aggregated)
            log_var = self.fc_log_var(aggregated)

            z = self.reparameterize(mu, log_var)
            z_repeat = z.repeat(n_trials_target, 1)

        else:
            mu = mu_given
            log_var = 0
            z_repeat = mu_given.repeat(n_trials_target, 1)

        X_expanded = torch.cat([X_target, z_repeat], dim=1)
        Yh = self.decoder(X_expanded)

        return Yh, mu, log_var

class oldBP_D2(nn.Module):
    def __init__(self, x_dim, emb_size=32, latent_dim=16, y_dim=1):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, emb_size),

        )

        self.aggregator = deep_set2(
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

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).
        """

        log_var = torch.clamp(log_var, min=-20, max=20)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, XY_context, XY_target, mu_given=None):

        n_trials_target = XY_target.size(1)
        X_target = XY_target[0, :, :-1]

        if mu_given is None:
            encoded = self.encoder(XY_context)
            aggregated = self.aggregator(encoded)

            mu = self.fc_mu(aggregated)
            log_var = self.fc_log_var(aggregated)

            z = self.reparameterize(mu, log_var)
            z_repeat = z.repeat(n_trials_target, 1)

        else:
            mu = mu_given
            log_var = 0
            z_repeat = mu_given.repeat(n_trials_target, 1)

        X_expanded = torch.cat([X_target, z_repeat], dim=1)
        Yh = self.decoder(X_expanded)

        return Yh, mu, log_var

if __name__ == "__main__":
    model = oldBP_D(x_dim=4, y_dim=1, emb_size=32, latent_dim=16)
    XY_context = torch.randn(1, 99, 5)
    XY_target = torch.randn(1, 99, 5)

    Yh, mu, log_var = model(XY_context, XY_target)
    print(Yh.shape)
    print(mu.shape)
    print(log_var.shape)
