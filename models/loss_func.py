import torch
import torch.nn as nn
import torch.nn.functional as F

def beta_vae_loss(reconstructed, original, mu, log_var, beta):

    reconstruction_loss = nn.MSELoss()(reconstructed, original)

    kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    return reconstruction_loss + beta * kl_divergence

def beta_vae_loss2(reconstructed, original, mu, log_var, beta):

    reconstruction_loss = nn.MSELoss()(reconstructed, original)

    log_var = torch.clamp(log_var, min=-20, max=20)

    kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    if torch.isnan(reconstruction_loss) or torch.isnan(kl_divergence):
        print(f"NaN detected! reconstruction_loss: {reconstruction_loss}, kl_divergence: {kl_divergence}")
        print(f"mu range: [{mu.min():.4f}, {mu.max():.4f}], log_var range: [{log_var.min():.4f}, {log_var.max():.4f}]")

    return reconstruction_loss + beta * kl_divergence

def beta_vae_loss_with_regularization(reconstructed, original, mu, log_var, beta, model,
                                     l1_lambda=0.0, l2_lambda=0.001, return_components=False):
    """Beta-VAE

 Args:
 reconstructed:
 original:
 mu:
 log_var:
 beta: KL
 model:
 l1_lambda: L1
 l2_lambda: L2
 return_components:"""

    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    l1_reg = torch.tensor(0.0, device=reconstructed.device)
    l2_reg = torch.tensor(0.0, device=reconstructed.device)
    if l1_lambda + l2_lambda>0:
        for param in model.parameters():
            if param.requires_grad:
                l1_reg += torch.sum(torch.abs(param))
                l2_reg += torch.sum(param.pow(2))

    total_loss = reconstruction_loss + beta * kl_divergence + l1_lambda * l1_reg + l2_lambda * l2_reg

    if return_components:
        return total_loss, {
            'reconstruction': reconstruction_loss.item(),
            'kl_divergence': kl_divergence.item(),
            'l1_reg': l1_reg.item(),
            'l2_reg': l2_reg.item()
        }

    return total_loss

def info_vae_loss_with_regularization(
    reconstructed, original, mu, log_var,
    z=None, p_z_samples=None,
    beta=1, alpha=0.5, lambd=1000.0,
    model=None, l1_lambda=0.0, l2_lambda=0.001, return_components=False
):
    """InfoVAE ( z p_z_samples)

 Args:
 reconstructed:
 original:
 mu: q(z|x)
 log_var: q(z|x)
 z: q(z|x) ()
 p_z_samples: p(z) ()
 alpha: InfoVAE α
 lambd: InfoVAE λ
 model:
 l1_lambda: L1
 l2_lambda: L2
 return_components:"""

    if z is None:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps

    if p_z_samples is None:
        p_z_samples = torch.randn_like(z)

    reconstruction_loss = nn.MSELoss()(reconstructed, original)

    kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    def compute_mmd(x, y, sigma=1.0):
        "Maximum Mean Discrepancy (RBF kernel)"
        xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        K_xx = torch.exp(- (rx.t() + rx - 2*xx) / (2*sigma**2))
        K_yy = torch.exp(- (ry.t() + ry - 2*yy) / (2*sigma**2))
        K_xy = torch.exp(- (rx.t() + ry - 2*xy) / (2*sigma**2))

        return K_xx.mean() + K_yy.mean() - 2*K_xy.mean()

    mmd_loss = compute_mmd(z, p_z_samples)

    info_vae_loss = (
        reconstruction_loss
        + (1 - alpha) * kl_divergence
        + (alpha + lambd - 1) * mmd_loss
    )

    l1_reg, l2_reg = torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)
    if model is not None and (l1_lambda + l2_lambda > 0):
        for param in model.parameters():
            if param.requires_grad:
                l1_reg += torch.sum(torch.abs(param))
                l2_reg += torch.sum(param.pow(2))

    total_loss = info_vae_loss + l1_lambda * l1_reg + l2_lambda * l2_reg

    if return_components:
        return total_loss, {
            'reconstruction': reconstruction_loss.item(),
            'kl_divergence': kl_divergence.item(),
            'mmd_loss': mmd_loss.item(),
            'l1_reg': l1_reg.item(),
            'l2_reg': l2_reg.item()
        }

    return total_loss

def info_bce_loss_batch(logits, target, mu, log_var, alpha=0.5, lambda_mmd=1000.0, beta=1.0, z=None,
                        trial_mask=None, return_components=False, lambda_sparse=0.0, gamma=0.0):
    """InfoVAE batch :
 L = BCE + β(1 - alpha) * KL(q(z|x)||p(z)) + (alpha + lambda_mmd - 1) * MMD(q(z)||p(z))"""

    if trial_mask is None:
        bce_loss = nn.BCEWithLogitsLoss()(logits, target)
    else:
        elementwise_bce = nn.BCEWithLogitsLoss(reduction='none')(logits, target)
        mask = trial_mask
        while mask.dim() < elementwise_bce.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=elementwise_bce.dtype, device=elementwise_bce.device)
        masked_bce = elementwise_bce * mask
        denom = mask.sum().clamp(min=1.0)
        bce_loss = masked_bce.sum() / denom

    log_var_clamped = torch.clamp(log_var, min=-20, max=20)
    kl_divergence = -0.5 * torch.mean(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp())

    if z is None:
        std = torch.exp(0.5 * log_var_clamped)
        eps = torch.randn_like(std)
        z_samples = mu + std * eps
    else:
        z_samples = z

    def _flatten_to_2d(t):
        return t.reshape(-1, t.size(-1))
    z_flat = _flatten_to_2d(z_samples)
    if trial_mask is not None and z_samples.dim() >= 2:
        mask_flat = trial_mask
        while mask_flat.dim() < z_samples.dim() - 1:
            mask_flat = mask_flat.unsqueeze(-1)
        mask_flat = mask_flat.reshape(-1)
        if mask_flat.numel() == z_flat.size(0):
            keep = mask_flat > 0.5
            if keep.any():
                z_flat = z_flat[keep]

    prior_flat = torch.randn_like(z_flat)

    def _mmd_rbf(x, y):
        if x.numel() == 0 or y.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        x2 = (x * x).sum(dim=1, keepdim=True)
        y2 = (y * y).sum(dim=1, keepdim=True)
        dist_xx = x2 - 2 * (x @ x.t()) + x2.t()
        dist_yy = y2 - 2 * (y @ y.t()) + y2.t()
        dist_xy = x2 - 2 * (x @ y.t()) + y2.t()
        sigma_list = [0.5, 1.0, 2.0, 4.0, 8.0]
        K_xx = K_yy = K_xy = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma * sigma)
            K_xx += torch.exp(-gamma * dist_xx)
            K_yy += torch.exp(-gamma * dist_yy)
            K_xy += torch.exp(-gamma * dist_xy)
        mmd2 = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
        return torch.relu(mmd2)

    mmd = _mmd_rbf(z_flat, prior_flat)

    sparse_loss = torch.abs(mu).sum(dim=1).mean()

    decor_loss = _decorrelation_loss_from_mu(mu)

    total_loss = bce_loss + beta * (1.0 - alpha) * kl_divergence + (alpha + lambda_mmd - 1.0) * mmd + lambda_sparse * sparse_loss + gamma * decor_loss

    if return_components:
        return total_loss, {
            'reconstruction': bce_loss.item(),
            'kl_divergence': kl_divergence.item(),
            'mmd': mmd.item(),
            'sparse_loss': sparse_loss.item(),
            'decor_loss': decor_loss.item(),
        }

    return total_loss

def beta_vae_loss_without_regularization(reconstructed, original, mu, log_var, beta,
                                    return_components=False):

    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = reconstruction_loss + beta * kl_divergence

    if return_components:
        return total_loss, {
            'reconstruction': reconstruction_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }

    return total_loss

def beta_vae_bce_loss_with_regularization(logits, target, mu, log_var, beta, model,
                                          l1_lambda=0.0, l2_lambda=0.001, return_components=False):
    """Beta-VAE BCEWithLogitsLoss

 Args:
 logits: sigmoid
 target: {0,1}
 mu, log_var, beta, model, l1_lambda, l2_lambda, return_components:"""

    bce_loss = nn.BCEWithLogitsLoss()(logits, target)

    log_var = torch.clamp(log_var, min=-20, max=20)
    kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    l1_reg = torch.tensor(0.0, device=logits.device)
    l2_reg = torch.tensor(0.0, device=logits.device)
    if l1_lambda + l2_lambda > 0:
        for param in model.parameters():
            if param.requires_grad:
                l1_reg += torch.sum(torch.abs(param))
                l2_reg += torch.sum(param.pow(2))

    total_loss = bce_loss + beta * kl_divergence + l1_lambda * l1_reg + l2_lambda * l2_reg

    if return_components:
        return total_loss, {
            'reconstruction': bce_loss.item(),
            'kl_divergence': kl_divergence.item(),
            'l1_reg': l1_reg.item(),
            'l2_reg': l2_reg.item()
        }

    return total_loss

def _decorrelation_loss_from_mu(mu):
    z = mu.reshape(-1, mu.size(-1))
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.t() @ z_centered) / z_centered.shape[0]
    I = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    decor_loss = ((cov - I) ** 2).mean()
    return decor_loss

def beta_bce_loss_batch(logits, target, mu, log_var, beta, trial_mask=None, return_components=False, lambda_sparse=0.0, gamma=0.0):
    """Beta-VAE batch L1/L2 BCEWithLogitsLoss

 Args:
 logits: sigmoid batch
 target: {0,1} logits
 mu: (batch, latent_dim)
 log_var: (batch, latent_dim)
 beta: KL
 return_components:"""

    if trial_mask is None:

        bce_loss = nn.BCEWithLogitsLoss()(logits, target)
    else:

        elementwise_bce = nn.BCEWithLogitsLoss(reduction='none')(logits, target)

        mask = trial_mask
        while mask.dim() < elementwise_bce.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=elementwise_bce.dtype, device=elementwise_bce.device)
        masked_bce = elementwise_bce * mask
        denom = mask.sum().clamp(min=1.0)
        bce_loss = masked_bce.sum() / denom

    log_var = torch.clamp(log_var, min=-20, max=20)
    kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    sparse_loss = torch.abs(mu).sum(dim=1).mean()

    decor_loss = _decorrelation_loss_from_mu(mu)

    total_loss = bce_loss + beta * kl_divergence + lambda_sparse * sparse_loss + gamma * decor_loss

    if return_components:
        return total_loss, {
            'reconstruction': bce_loss.item(),
            'kl_divergence': kl_divergence.item(),
            'sparse_loss': sparse_loss.item(),
            'decor_loss': decor_loss.item(),
        }

    return total_loss

def bce_loss_batch(logits, target, trial_mask=None):

    if trial_mask is None:
        bce_loss = nn.BCEWithLogitsLoss()(logits, target)
    else:
        elementwise_bce = nn.BCEWithLogitsLoss(reduction='none')(logits, target)
        mask = trial_mask
        while mask.dim() < elementwise_bce.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=elementwise_bce.dtype, device=elementwise_bce.device)
        masked_bce = elementwise_bce * mask
        denom = mask.sum().clamp(min=1.0)
        bce_loss = masked_bce.sum() / denom

    return bce_loss

def beta_mse_loss_batch(preds, target, mu, log_var, beta, trial_mask=None, return_components=False, lambda_sparse=0.0, gamma=0.0):
    """Beta-VAE batch L1/L2 MSELoss

 Args:
 preds: batch
 target: preds
 mu: (batch, latent_dim)
 log_var: (batch, latent_dim)
 beta: KL
 return_components:"""

    if trial_mask is None:

        mse_loss = nn.MSELoss()(preds, target)
    else:

        elementwise_mse = nn.MSELoss(reduction='none')(preds, target)

        mask = trial_mask
        while mask.dim() < elementwise_mse.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=elementwise_mse.dtype, device=elementwise_mse.device)
        masked_mse = elementwise_mse * mask
        denom = mask.sum().clamp(min=1.0)
        mse_loss = masked_mse.sum() / denom

    log_var = torch.clamp(log_var, min=-20, max=20)
    kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    sparse_loss = torch.abs(mu).sum(dim=1).mean()

    decor_loss = _decorrelation_loss_from_mu(mu)

    total_loss = mse_loss + beta * kl_divergence + lambda_sparse * sparse_loss + gamma * decor_loss

    if return_components:
        return total_loss, {
            'reconstruction': mse_loss.item(),
            'kl_divergence': kl_divergence.item(),
            'sparse_loss': sparse_loss.item(),
            'decor_loss': decor_loss.item(),
        }

    return total_loss

def mse_loss_batch(preds, target, trial_mask=None):

    if trial_mask is None:
        mse_loss = nn.MSELoss()(preds, target)
    else:
        elementwise_mse = nn.MSELoss(reduction='none')(preds, target)
        mask = trial_mask
        while mask.dim() < elementwise_mse.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=elementwise_mse.dtype, device=elementwise_mse.device)
        masked_mse = elementwise_mse * mask
        denom = mask.sum().clamp(min=1.0)
        mse_loss = masked_mse.sum() / denom

    return mse_loss

def info_mse_loss_batch(preds, target, mu, log_var, alpha=0.5, lambda_mmd=1000.0, beta=1.0, z=None,
                        trial_mask=None, return_components=False, lambda_sparse=0.0, gamma=0.0):
    """InfoVAE batch :
 L = MSE + β(1 - alpha) * KL(q(z|x)||p(z)) + (alpha + lambda_mmd - 1) * MMD(q(z)||p(z))

 Args:
 preds: batch
 target: preds
 mu: [..., latent_dim]
 log_var: mu
 alpha: KL MMD [0, 1]
 lambda_mmd: MMD λ 1000
 beta: KL 1.0 InfoVAE>1 β-VAE KL
 z: z None reparameterization
 trial_mask: padding MMD
 return_components:"""

    if trial_mask is None:
        mse_loss = nn.MSELoss()(preds, target)
    else:
        elementwise_mse = nn.MSELoss(reduction='none')(preds, target)
        mask = trial_mask
        while mask.dim() < elementwise_mse.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=elementwise_mse.dtype, device=elementwise_mse.device)
        masked_mse = elementwise_mse * mask
        denom = mask.sum().clamp(min=1.0)
        mse_loss = masked_mse.sum() / denom

    log_var_clamped = torch.clamp(log_var, min=-20, max=20)
    kl_divergence = -0.5 * torch.mean(1 + log_var_clamped - mu.pow(2) - log_var_clamped.exp())

    if z is None:
        std = torch.exp(0.5 * log_var_clamped)
        eps = torch.randn_like(std)
        z_samples = mu + std * eps
    else:
        z_samples = z

    def _flatten_to_2d(t):
        return t.reshape(-1, t.size(-1))
    z_flat = _flatten_to_2d(z_samples)
    if trial_mask is not None and z_samples.dim() >= 2:
        mask_flat = trial_mask
        while mask_flat.dim() < z_samples.dim() - 1:
            mask_flat = mask_flat.unsqueeze(-1)
        mask_flat = mask_flat.reshape(-1)
        if mask_flat.numel() == z_flat.size(0):
            keep = mask_flat > 0.5
            if keep.any():
                z_flat = z_flat[keep]

    prior_flat = torch.randn_like(z_flat)

    def _mmd_rbf(x, y):
        if x.numel() == 0 or y.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        x2 = (x * x).sum(dim=1, keepdim=True)
        y2 = (y * y).sum(dim=1, keepdim=True)
        dist_xx = x2 - 2 * (x @ x.t()) + x2.t()
        dist_yy = y2 - 2 * (y @ y.t()) + y2.t()
        dist_xy = x2 - 2 * (x @ y.t()) + y2.t()
        sigma_list = [0.5, 1.0, 2.0, 4.0, 8.0]
        K_xx = K_yy = K_xy = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma * sigma)
            K_xx += torch.exp(-gamma * dist_xx)
            K_yy += torch.exp(-gamma * dist_yy)
            K_xy += torch.exp(-gamma * dist_xy)
        mmd2 = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
        return torch.relu(mmd2)

    mmd = _mmd_rbf(z_flat, prior_flat)

    sparse_loss = torch.abs(mu).sum(dim=1).mean()

    decor_loss = _decorrelation_loss_from_mu(mu)

    total_loss = mse_loss + beta * (1.0 - alpha) * kl_divergence + (alpha + lambda_mmd - 1.0) * mmd + lambda_sparse * sparse_loss + gamma * decor_loss

    if return_components:
        return total_loss, {
            'reconstruction': mse_loss.item(),
            'kl_divergence': kl_divergence.item(),
            'mmd': mmd.item(),
            'sparse_loss': sparse_loss.item(),
            'decor_loss': decor_loss.item(),
        }

    return total_loss
