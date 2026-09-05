import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
from tools import *

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

idx = 1
data_name = ["RiskC", "DelayC", "RiskP", "DelayP", "BB"][idx]

beta = [0.5, 0.5, 0.25, 1, 0][idx]
lambda_sparse =  [0, 0, 0.01, 0, 0][idx]

chosen_order = 0

latent_dim = 8

loss_type = "beta"
with_outer = 0
nrpeat = 1
an_beta = 0
lambda_mmd = 1000

threshold = 0.1

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

data_name = data.get("data_name", data_name)
latent_dim = int(data.get("latent_dim", latent_dim))

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

z_vars = np.var(all_mu, axis=0)
rank_desc = np.argsort(z_vars)[::-1]
co = int(np.clip(chosen_order, 0, len(rank_desc) - 1))
if co != chosen_order:
    print(f"chosen_order={chosen_order} out of range; using {co}")
z_active_idx = int(rank_desc[co])
print("Per-dim var(mu):", z_vars)
print(f"z dim rank #{co} by variance (chosen_order): index={z_active_idx}, var={z_vars[z_active_idx]:.6f}")
active_dims = np.where(z_vars >= threshold)[0]
print(f"Dims with var >= {threshold}: {active_dims.tolist()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = data["model"].to(device)
model.load_state_dict(data["best_model_state"])
model.eval()

x_dim = int(data["x_dim"])
y_dim = int(data["y_dim"])

num_subj = all_mu.shape[0]

is_choice_like = data_name[-1] == "C" or data_name == "BB"

sub_orders = [int(num_subj * 0.1), int(num_subj * 0.5), int(num_subj * 0.9)]
sorted_order = np.argsort(all_mu[:, z_active_idx])
sorted_mus = all_mu[sorted_order]

bound_x = 1.5

if data_name == "DelayP":
    labels = {"x1": "Money", "x2": "Time", "response": "Price"}
elif data_name == "RiskP":
    labels = {"x1": "Money", "x2": "Probability", "response": "Price"}
elif data_name == "DelayC":
    labels = {"x1": "Money", "x2": "Time", "response": "Choice probability"}
elif data_name == "RiskC":
    labels = {"x1": "Money", "x2": "Probability", "response": "Choice probability"}
elif data_name == "BB":
    labels = {"x1": "x1", "x2": "x2", "response": "Choice probability"}
else:
    labels = {"x1": "x1", "x2": "x2", "response": "y"}

def _x_grid_and_line(data_name_, device_):
    "s5 x_dim>2 X_target 0"
    if data_name_[-1] == "C" or data_name_ == "BB":
        all_x1 = torch.linspace(0, bound_x, 50, device=device_)
        all_x2 = torch.linspace(0, bound_x, 50, device=device_)
    elif data_name_[-1] == "P":
        all_x1 = torch.linspace(-bound_x, bound_x, 50, device=device_)
        all_x2 = torch.linspace(-bound_x, bound_x, 50, device=device_)
    else:
        all_x1 = torch.linspace(0, bound_x, 50, device=device_)
        all_x2 = torch.linspace(0, bound_x, 50, device=device_)
    return all_x1, all_x2

def decoder_surface(this_mu_1xlat, all_x1, all_x2):
    """this_mu_1xlat: (1, latent_dim) numpy"""
    this_mu = torch.tensor(this_mu_1xlat, dtype=torch.float32, device=device).unsqueeze(1)
    all_y = torch.zeros(len(all_x1), len(all_x2), device=device)
    for i, x1 in enumerate(all_x1):
        for j, x2 in enumerate(all_x2):
            X_target = torch.zeros(1, 1, x_dim, device=device)
            X_target[0, 0, 0] = x1
            X_target[0, 0, 1] = x2
            X_expanded = torch.cat([X_target, this_mu], dim=-1)
            all_y[i, j] = model.decoder(X_expanded).squeeze().item()
    return all_y

def decoder_line_x2(this_mu_1xlat, all_x2):
    this_mu = torch.tensor(this_mu_1xlat, dtype=torch.float32, device=device).unsqueeze(1)
    all_y_m0 = torch.zeros(len(all_x2), device=device)
    for j, x2 in enumerate(all_x2):
        X_target = torch.zeros(1, 1, x_dim, device=device)
        if is_choice_like:
            X_target[0, 0, 0] = 0
        X_target[0, 0, 1] = x2
        X_expanded = torch.cat([X_target, this_mu], dim=-1)
        all_y_m0[j] = model.decoder(X_expanded).squeeze().item()
    return all_y_m0

all_x1, all_x2 = _x_grid_and_line(data_name, device)

ft_label = 16
n_rows = len(sub_orders)
fig, axes_grid = plt.subplots(n_rows, 3, figsize=(18, 4.8 * n_rows))
if n_rows == 1:
    axes_grid = np.array([axes_grid])

all_all_y = []
all_all_y_m0 = []

for row_idx, sub_order in enumerate(sub_orders):
    this_mu_np = sorted_mus[sub_order : sub_order + 1, :].copy()

    all_y = decoder_surface(this_mu_np, all_x1, all_x2)
    all_y_m0 = decoder_line_x2(this_mu_np, all_x2)

    if is_choice_like:
        all_y = torch.sigmoid(all_y)
        all_y_m0 = torch.sigmoid(all_y_m0)

    all_all_y.append(all_y.cpu().numpy())
    all_all_y_m0.append(all_y_m0.cpu().numpy())

    if is_choice_like:
        lines = [
            0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2
        ]
    elif data_name[-1] == "P":
        lines = [
            torch.quantile(all_y, 0.1).cpu(),
            torch.quantile(all_y, 0.3).cpu(),
            torch.quantile(all_y, 0.5).cpu(),
            torch.quantile(all_y, 0.7).cpu(),
            torch.quantile(all_y, 0.9).cpu(),
        ]
    else:
        lines = np.linspace(float(all_y.min().cpu()), float(all_y.max().cpu()), 7).tolist()

    ax_left = axes_grid[row_idx, 0]
    ax_right = axes_grid[row_idx, 1]
    ax_line = axes_grid[row_idx, 2]

    im = ax_left.imshow(
        all_y.cpu(),
        cmap="viridis",
        origin="lower",
        extent=[
            all_x2[0].item(),
            all_x2[-1].item(),
            all_x1[0].item(),
            all_x1[-1].item(),
        ],
        aspect="auto",
    )

    X2, X1 = torch.meshgrid(all_x2, all_x1, indexing="xy")
    CS = ax_left.contour(
        X2.cpu().numpy(),
        X1.cpu().numpy(),
        all_y.cpu().numpy(),
        levels=lines,
        colors="white",
        linewidths=1.5,
    )
    ax_left.clabel(CS, inline=True, fontsize=10, fmt="%.2f", colors="white")

    ax_left.set_xlabel(labels["x2"], fontsize=ft_label)
    ax_left.set_ylabel(labels["x1"], fontsize=ft_label)
    ax_left.set_title(f"{labels['response']}", fontsize=ft_label)
    fig.colorbar(im, ax=ax_left)

    mean_val = all_y.mean().item()
    binary_map = (all_y > mean_val).float()
    im2 = ax_right.imshow(
        binary_map.cpu(),
        cmap="bwr",
        origin="lower",
        extent=[
            all_x2[0].item(),
            all_x2[-1].item(),
            all_x1[0].item(),
            all_x1[-1].item(),
        ],
        aspect="auto",
    )
    ax_right.set_xlabel(labels["x2"], fontsize=ft_label)
    ax_right.set_ylabel(labels["x1"], fontsize=ft_label)
    ax_right.set_title(f"{labels['response']} (mean={mean_val:.2f})", fontsize=ft_label)
    cbar2 = fig.colorbar(im2, ax=ax_right, ticks=[0, 1])
    cbar2.ax.set_yticklabels(["≤ mean", "> mean"])

    ax_line.plot(all_x2.cpu().numpy(), all_y_m0.cpu().numpy())
    ax_line.set_xlim(all_x2[0].item(), all_x2[-1].item())
    ax_line.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax_line.set_title(f"{labels['x1']} = mean", fontsize=ft_label)
    ax_line.set_xlabel(labels["x2"], fontsize=ft_label)
    ax_line.set_ylabel(labels["response"], fontsize=ft_label)

plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.06)
for row_idx, sub_order in enumerate(sub_orders):
    ax_left = axes_grid[row_idx, 0]
    bbox = ax_left.get_position()
    y_text = bbox.y1 + 0.015
    z_val = sorted_mus[sub_order, z_active_idx]
    fig.text(
        0.5,
        y_text,
        f"subj order {sub_order} (sorted by z[{z_active_idx}], z={z_val:.4f})",
        ha="center",
        va="bottom",
        fontsize=ft_label,
        fontweight="bold",
    )

plt.suptitle(
    f"{data_name} | z[{z_active_idx}] (var rank #{co}, chosen_order={chosen_order}, no PCA)",
    fontsize=14,
    y=0.98,
)
plt.show()

sub_orders_manual = [-2, -1, 0, 1, 2]
sub_orders_manual_labels = ["-2σ", "-σ", "mean", "+σ", "+2σ"]

mu_mean = np.mean(all_mu, axis=0)
mu_std = np.std(all_mu, axis=0)

all_all_y_m0_manual = []
for sub_order in sub_orders_manual:
    this_mu_np = mu_mean.reshape(1, -1).copy()
    this_mu_np[0, z_active_idx] = mu_mean[z_active_idx] + sub_order * mu_std[z_active_idx]

    all_y_m0_manual = decoder_line_x2(this_mu_np, all_x2)
    if is_choice_like:
        all_y_m0_manual = torch.sigmoid(all_y_m0_manual)
    all_all_y_m0_manual.append(all_y_m0_manual.cpu().numpy())

ft_label = 16
dic = {
    "DelayP": "Intertemporal pricing",
    "RiskP": "Risky pricing (factor 2)",
    "DelayC": "Intertemporal choice",
    "RiskC": "Risky choice",
    "BB": "BB task",
}
suptitle_text = dic.get(data_name, data_name)

n_plots = len(all_all_y)
color_subj = ["red", "green", "blue"]

widths = [0.8] * n_plots + [0.8, 0.8]

fig2, axes2 = plt.subplots(
    1, n_plots + 2,
    figsize=(3 * (n_plots + 2), 4),
    gridspec_kw={"width_ratios": widths}
)

if n_plots + 1 == 1:
    axes2 = np.array([axes2])

title_top = 1
fig2.suptitle(suptitle_text,
             fontsize=16, fontweight="bold", y=0.96)

for idx, all_y in enumerate(all_all_y):
    if is_choice_like:
        lines = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3,
                 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
    elif data_name[-1] == "P":
        lines = [np.quantile(all_y, 0.1),
                 np.quantile(all_y, 0.3),
                 np.quantile(all_y, 0.5),
                 np.quantile(all_y, 0.7),
                 np.quantile(all_y, 0.9)]
    else:
        lines = np.linspace(np.min(all_y), np.max(all_y), 7).tolist()

    im = axes2[idx].imshow(
        all_y,
        cmap="viridis",
        origin="lower",
        extent=[all_x2[0].item(), all_x2[-1].item(),
                all_x1[0].item(), all_x1[-1].item()],
        aspect="auto"
    )

    X2, X1 = torch.meshgrid(all_x2, all_x1, indexing="xy")
    CS = axes2[idx].contour(
        X2.cpu().numpy(),
        X1.cpu().numpy(),
        all_y,
        levels=lines,
        colors="white",
        linewidths=1.5
    )
    axes2[idx].clabel(CS, inline=True, fontsize=12, fmt="%.2f", colors="white")
    axes2[idx].set_xlabel(labels["x2"], fontsize=ft_label)
    if idx == 0:
        axes2[idx].set_ylabel(labels["x1"], fontsize=ft_label)
    axes2[idx].set_title(f"{labels['response']}\nSubject #{sub_orders[idx]}", fontsize=ft_label)
    axes2[idx].tick_params(axis="both", which="major", labelsize=12)

ax_last = axes2[-2]
cmap = cm.get_cmap("viridis", len(all_all_y_m0))
colors = [cmap(i) for i in range(len(all_all_y_m0))]
for i, L in enumerate(all_all_y_m0):
    ax_last.plot(L, label=f"#{sub_orders[i]}", color=colors[i])

    ax_last.set_xlabel(labels["x2"], fontsize=ft_label)
    ax_last.set_ylabel(labels["response"], fontsize=ft_label)
    ax_last.tick_params(axis="both", which="major", labelsize=12)
    ax_last.set_title("Real Subjects\n(Money fixed at mean)", fontsize=ft_label)

ax_last.legend(frameon=False, fontsize=14, loc="best")

ax_last = axes2[-1]

cmap = cm.get_cmap("viridis", len(all_all_y_m0_manual))
colors = [cmap(i) for i in range(len(all_all_y_m0_manual))]
for i, L in enumerate(all_all_y_m0_manual):
    ax_last.plot(L, label=f"{sub_orders_manual_labels[i]}", color=colors[i])

    ax_last.set_xlabel(labels["x2"], fontsize=ft_label)

    ax_last.tick_params(axis="both", which="major", labelsize=12)
    ax_last.set_title(f"Varying Selected z[{z_active_idx}]\n(Money fixed at mean)", fontsize=ft_label)

ax_last.legend(frameon=False, fontsize=14, loc="best")

plt.tight_layout(rect=[0, 0, 1, title_top])
plt.show()
