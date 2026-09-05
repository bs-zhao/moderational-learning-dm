import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
from tools import *

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

idx = 0
data_name = ["RiskC", "DelayC", "RiskP", "DelayP", "BB"][idx]

beta = [0.5, 0.5, 0.25, 1, 0][idx]
lambda_sparse =  [0, 0, 0.01, 0, 0][idx]

latent_dim = 8

loss_type = "beta"
with_outer = 0
nrpeat = 1
an_beta = 0
lambda_mmd = 1000

threshold = 0.1
log_var_eps = 1e-12

def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)

def get_path(beta, sparse_lambda):
    if an_beta != 1:
        dir_save = (
            f"save/s2_train/MLP_mod/"
            f"{data_name}_{loss_type}_b_{_fmt(beta)}_lat_{_fmt(latent_dim)}"
            f"_outer_{_fmt(with_outer)}_sp_{_fmt(sparse_lambda)}_rp_{_fmt(nrpeat)}/"
        )
    else:
        dir_save = (
            f"save/s2_train/MLP_mod/"
            f"{data_name}_{loss_type}_b_{_fmt(beta)}_la_{_fmt(lambda_mmd)}_lat_{_fmt(latent_dim)}"
            f"_outer_{_fmt(with_outer)}_sp_{_fmt(sparse_lambda)}_rp_{_fmt(nrpeat)}_an_{_fmt(an_beta)}/"
        )
    return f"{dir_save}/all.pkl"

load_path = get_path(beta, lambda_sparse)

print("Loading:", load_path)

if not os.path.exists(load_path):
    raise FileNotFoundError(f"Missing file: {load_path}")

with open(load_path, "rb") as f:
    data = pickle.load(f)

if "all_mu" in data:
    all_mu = np.concatenate(data["all_mu"], axis=0)
else:
    print("all.pkl does not contain all-subject latent mu. Recomputing mu from all_loader ...")
    required_keys = [
        "all_loader",
        "model",
        "best_model_state",
        "max_len_trials",
        "max_len_trials_outer",
        "x_dim",
        "y_dim",
        "outer_dim",
    ]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise KeyError(f"Missing keys required to rebuild all-subject mu: {missing_keys}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = data["model"].to(device)
    model.load_state_dict(data["best_model_state"])
    model.eval()

    all_loader = data["all_loader"]
    max_len_trials = int(data["max_len_trials"])
    max_len_trials_outer = int(data["max_len_trials_outer"])
    x_dim = int(data["x_dim"])
    y_dim = int(data["y_dim"])
    outer_dim = int(data["outer_dim"])

    all_mu_list = []
    with torch.no_grad():
        for X_Y_batch, X_Y_batch_outer, _ in all_loader:
            B = len(X_Y_batch)

            XY_context_batch = torch.zeros(B, max_len_trials, x_dim + y_dim, device=device, dtype=torch.float32)
            XY_outer_batch = torch.zeros(B, max_len_trials_outer, outer_dim, device=device, dtype=torch.float32)
            XY_target_batch = torch.zeros(B, max_len_trials, x_dim + y_dim, device=device, dtype=torch.float32)
            trial_mask_context = torch.zeros(B, max_len_trials, dtype=torch.bool, device=device)
            trial_mask_outer = torch.zeros(B, max_len_trials_outer, dtype=torch.bool, device=device)
            trial_mask_target = torch.zeros(B, max_len_trials, dtype=torch.bool, device=device)

            for i in range(B):
                X_Y = torch.tensor(X_Y_batch[i], device=device, dtype=torch.float32)
                X_Y_outer = torch.tensor(X_Y_batch_outer[i], device=device, dtype=torch.float32)

                X_Y_context = X_Y
                X_Y_target = X_Y

                Ti_ctx = X_Y_context.shape[0]
                Ti_outer = X_Y_outer.shape[0]
                Ti_outer_use = min(Ti_outer, max_len_trials_outer)
                Ti_tgt = X_Y_target.shape[0]
                XY_context_batch[i, :Ti_ctx, :] = X_Y_context.float()
                XY_outer_batch[i, :Ti_outer_use, :] = X_Y_outer[:Ti_outer_use, :].float()
                XY_target_batch[i, :Ti_tgt, :] = X_Y_target.float()
                trial_mask_context[i, :Ti_ctx] = True
                trial_mask_outer[i, :Ti_outer_use] = True
                trial_mask_target[i, :Ti_tgt] = True

            _, mu, _ = model(
                XY_context_batch,
                XY_target_batch,
                XY_outer=XY_outer_batch,
                trial_mask_context=trial_mask_context,
                trial_mask_outer=trial_mask_outer,
                trial_mask_target=trial_mask_target,
            )
            all_mu_list.append(mu.detach().cpu())

    if not all_mu_list:
        raise RuntimeError("Failed to rebuild all-subject mu from all_loader: empty result.")

    all_mu = torch.cat(all_mu_list, dim=0).numpy()

print("Shape of latent:", all_mu.shape)

mean_abs_mu = np.mean(np.abs(all_mu), axis=0)
var_mu = np.var(all_mu, axis=0)
active_idx = np.where(mean_abs_mu > threshold)[0]

n_z_full = mean_abs_mu.shape[0]
z_ix_full = np.arange(n_z_full)
z_labels_full = [f"z{i}" for i in range(n_z_full)]
active_mask_full = mean_abs_mu > threshold
bar_colors_full = np.where(active_mask_full, "steelblue", "lightgray")
log_var_full = np.log10(var_mu + log_var_eps)

fig_overview, axes_ov = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig_overview.suptitle(
    f"All latent dims ({data_name}) — active: mean|μ|>{threshold} "
    f"→ {int(active_mask_full.sum())}/{n_z_full}",
    fontsize=11,
)

ax0, ax1, ax2 = axes_ov
ax0.bar(z_ix_full, mean_abs_mu, color=bar_colors_full)
ax0.axhline(threshold, linestyle="--", color="crimson", label=f"threshold={threshold}")
ax0.set_ylabel("mean |μ|")
ax0.set_title("Mean |μ| per z (all dimensions)")
ax0.legend(loc="upper right", fontsize=8)

ax1.bar(z_ix_full, var_mu, color=bar_colors_full)
ax1.set_ylabel("Var(μ)")
ax1.set_title("Variance of μ across subjects (all z)")

ax2.bar(z_ix_full, log_var_full, color=bar_colors_full)
ax2.set_ylabel(r"$\log_{10}(\mathrm{Var}(\mu)+\epsilon)$")
ax2.set_title("Log variance of μ (all z)")
ax2.set_xticks(z_ix_full)
ax2.set_xticklabels(z_labels_full, rotation=0)

plt.tight_layout()
plt.show()

if active_idx.size == 0:
    raise RuntimeError(f"No active z dims found with threshold={threshold}.")

sorted_active_idx = active_idx[np.argsort(var_mu[active_idx])[::-1]]
all_mu = all_mu[:, sorted_active_idx]
mean_abs_mu = mean_abs_mu[sorted_active_idx]
var_mu = var_mu[sorted_active_idx]
z_labels = [f"z{idx}" for idx in sorted_active_idx]

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.bar(range(len(mean_abs_mu)), mean_abs_mu)
plt.axhline(threshold, linestyle='--', color='r')
plt.title("Mean |mu| per dimension")
plt.xticks(range(len(z_labels)), z_labels, rotation=90)

plt.subplot(1,2,2)
plt.bar(range(len(var_mu)), var_mu)
plt.title("Variance per dimension")
plt.xticks(range(len(z_labels)), z_labels, rotation=90)

plt.tight_layout()
plt.show()

print("Active dims after filtering:", len(sorted_active_idx))
print("Used z dims (sorted by variance):", sorted_active_idx.tolist())

corr = np.corrcoef(all_mu.T)
if np.ndim(corr) == 0:

    corr = np.array([[float(corr)]])

plt.figure(figsize=(6,5))
sns.heatmap(
    corr,
    cmap='coolwarm',
    center=0,
    annot=True,
    fmt=".2f",
    xticklabels=z_labels,
    yticklabels=z_labels,
)
plt.title("Latent correlation matrix")
plt.show()

pca = PCA()
pca.fit(all_mu)

explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(explained, marker='o')
plt.title("Explained variance per PC")

plt.subplot(1,2,2)
plt.plot(cum_explained, marker='o')
plt.axhline(0.9, linestyle='--', color='r')
plt.title("Cumulative variance")

plt.tight_layout()
plt.show()

print("PCs for 90% variance:", np.sum(cum_explained < 0.9) + 1)

z_pca = pca.transform(all_mu)

plt.figure(figsize=(6,6))
if z_pca.shape[1] >= 2:
    plt.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.3)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection")
else:
    plt.scatter(z_pca[:, 0], np.zeros_like(z_pca[:, 0]), alpha=0.3)
    plt.xlabel("PC1")
    plt.ylabel("(constant)")
    plt.title("PCA projection (only one active dimension)")
plt.show()

n_dim = all_mu.shape[1]
pc_z_corr = np.zeros((n_dim, n_dim))

for i in range(n_dim):
    for j in range(n_dim):
        pc_z_corr[i,j] = np.corrcoef(z_pca[:,i], all_mu[:,j])[0,1]

plt.figure(figsize=(6,5))
sns.heatmap(
    pc_z_corr,
    cmap='coolwarm',
    center=0,
    annot=True,
    fmt=".2f",
    xticklabels=z_labels,
    yticklabels=[f"PC{i+1}" for i in range(n_dim)],
)
plt.xlabel("z dimension")
plt.ylabel("PC dimension")
plt.title("Correlation between PCs and z")
plt.show()

print("\nInterpretation guide:")
print("- → axis aligned ✔")
print("- → rotated representation")
