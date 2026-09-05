import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
from tools import *

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

idx = 2
data_name = ["RiskC", "DelayC", "RiskP", "DelayP", "BB"][idx]

beta = [0.5, 0.5, 0.25, 1, 0.25][idx]
lambda_sparse =  [0, 0, 0.01, 0, 0][idx]

latent_dim = 8

loss_type = "beta"
with_outer = 0
nrpeat = 1
an_beta = 0
lambda_mmd = 1000

threshold_var = 1e-2
threshold_rel = 0.5
rel_eps = 1e-8

def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

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

def _forward_all_subjects(data, device):
    all_loader = data["all_loader"]
    max_len_trials = int(data["max_len_trials"])
    max_len_trials_outer = int(data["max_len_trials_outer"])
    x_dim = int(data["x_dim"])
    y_dim = int(data["y_dim"])
    outer_dim = int(data["outer_dim"])

    model = data["model"].to(device)
    model.load_state_dict(data["best_model_state"])
    model.eval()

    all_mu_list = []
    all_log_var_list = []
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

            _, mu, log_var = model(
                XY_context_batch,
                XY_target_batch,
                XY_outer=XY_outer_batch,
                trial_mask_context=trial_mask_context,
                trial_mask_outer=trial_mask_outer,
                trial_mask_target=trial_mask_target,
            )
            all_mu_list.append(mu.detach().cpu())
            all_log_var_list.append(log_var.detach().cpu())

    if not all_mu_list:
        raise RuntimeError("Failed to rebuild latent stats from all_loader: empty result.")

    return torch.cat(all_mu_list, dim=0).numpy(), torch.cat(all_log_var_list, dim=0).numpy()

def load_latent_posterior(data):
    """Return (all_mu, all_log_var or None) for all subjects."""
    if "all_mu" in data:
        all_mu = np.concatenate([_to_numpy(x) for x in data["all_mu"]], axis=0)
        if "all_log_var" in data:
            all_log_var = np.concatenate([_to_numpy(x) for x in data["all_log_var"]], axis=0)
            return all_mu, all_log_var
        if "all_log_var_test" in data:
            return all_mu, _to_numpy(data["all_log_var_test"])
        return all_mu, None

    if "all_mu_test" in data:
        all_mu = _to_numpy(data["all_mu_test"])
        if "all_log_var_test" in data:
            return all_mu, _to_numpy(data["all_log_var_test"])
        return all_mu, None

    print("all.pkl does not contain cached latent stats. Recomputing from all_loader ...")
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
        raise KeyError(f"Missing keys required to rebuild latent stats: {missing_keys}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _forward_all_subjects(data, device)

load_path = get_path(beta, lambda_sparse)

print("Loading:", load_path)

if not os.path.exists(load_path):
    raise FileNotFoundError(f"Missing file: {load_path}")

with open(load_path, "rb") as f:
    data = pickle.load(f)

all_mu, all_log_var = load_latent_posterior(data)
print("Shape of latent:", all_mu.shape)

var_mu = np.var(all_mu, axis=0, ddof=1)

if all_log_var is not None:
    mean_sigma2 = np.mean(np.exp(all_log_var), axis=0)
    reliability = var_mu / (var_mu + mean_sigma2 + rel_eps)
    sigma2 = np.exp(all_log_var)
    kl_per_dim = 0.5 * np.mean(all_mu**2 + sigma2 - all_log_var - 1.0, axis=0)
else:
    mean_sigma2 = None
    reliability = np.full_like(var_mu, np.nan)
    kl_per_dim = None

if all_log_var is not None:
    active_idx = np.where((var_mu > threshold_var) & (reliability > threshold_rel))[0]
    active_rule = f"Var(M)>{threshold_var:g} & R>{threshold_rel:g}"
else:
    active_idx = np.where(var_mu > threshold_var)[0]
    active_rule = f"Var(M)>{threshold_var:g} (no log_var → R unavailable)"

sort_idx_plot = np.argsort(var_mu)[::-1]
sort_idx_active = active_idx[np.argsort(var_mu[active_idx])[::-1]] if active_idx.size else active_idx

var_sorted = var_mu[sort_idx_plot]
rel_sorted = reliability[sort_idx_plot]
z_labels_sorted = [f"z{i}" for i in sort_idx_plot]

dic_title = {
    "DelayC": "Intertemporal\nchoice",
    "RiskC": "Risky\nchoice",
    "BB": "Risky\nchoice (BBRS)",
    "DelayP": "Intertemporal\npricing",
    "RiskP": "Risky\npricing",
}

fig, ax = plt.subplots(1, 1, figsize=(2.2, 3.2))
x_pos = range(1, len(var_sorted) + 1)

ax.plot(
    x_pos,
    var_sorted,
    marker="o",
    color="black",
    label=r"Var$_s$(M$_{sd}$)",
)
if all_log_var is not None:
    ax.plot(
        x_pos,
        rel_sorted,
        marker="o",
        linestyle="--",
        color="gray",
        label=r"$R_d$",
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(z_labels_sorted, fontsize=9)
ax.set_xlabel("Latent", fontsize=11)
ax.set_ylabel(" ", fontsize=11)
ax.set_title(dic_title.get(data_name, data_name), fontsize=13)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.show()

print("\n==== Latent dimension activity ====")
print("Active rule:", active_rule)
print("Var(M_d):", var_mu.tolist())
if all_log_var is not None:
    print("Mean Σ_dd:", mean_sigma2.tolist())
    print("Reliability R_d:", reliability.tolist())
    print("KL per dim:", kl_per_dim.tolist())
else:
    print("Reliability R_d: unavailable (missing log_var)")
print("Active dims (sorted by Var):", sort_idx_active.tolist())
print("Plot order z dims (Var high → low):", sort_idx_plot.tolist())
