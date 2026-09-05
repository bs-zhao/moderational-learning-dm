import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
from tools import *

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from matplotlib.patches import Rectangle, Circle, PathPatch
from matplotlib.path import Path

all_task_data_name = ["RiskC", "DelayC", "RiskP", "DelayP"]
all_task_beta = [0.5, 0.5, 0.25, 1, 0.25]
all_task_lambda_sparse = [0, 0, 0.01, 0, 0]

all_task_beta = [0.25, 0.5, 1, 0.25]
all_task_lambda_sparse =  [0.0001, 0.005, 0.1, 0.01]

all_task_n_z = [1, 1, 2, 1]

latent_dim = 8

loss_type = "beta"
with_outer = 0
nrpeat = 1
an_beta = 0
lambda_mmd = 1000

threshold = 0.1

if not (
    len(all_task_data_name) == len(all_task_beta)
    == len(all_task_lambda_sparse)
    == len(all_task_n_z)
):
    raise ValueError(
        "all_task_data_name / all_task_beta / all_task_lambda_sparse / all_task_n_z must have the same length."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)

def get_path(data_name, beta, sparse_lambda):
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

def load_subject_ids(data_name):
    path = f"save/s1_data/{data_name}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    all_subj = data["all_subj"]
    return np.asarray(all_subj)

def extract_all_mu(data):

    if "all_mu" in data:
        return np.concatenate(data["all_mu"], axis=0)

    print("all.pkl does not contain all_mu. Recomputing all-subject mu from all_loader ...")
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
    return torch.cat(all_mu_list, dim=0).numpy()

def extract_behavior_metrics(data_name, data):
    """s8
 RiskC -> risky_choice
 DelayC -> delay_choice
 RiskP -> risky_price1/2/3 (x1 slope, x2 slope, intercept)
 DelayP -> delay_price1/2/3 (x1 slope, x2 slope, intercept)"""
    all_loader = data["all_loader"]

    metrics = {}
    risky_choice = []
    delay_choice = []
    risky_price1, risky_price2, risky_price3 = [], [], []
    delay_price1, delay_price2, delay_price3 = [], [], []

    with torch.no_grad():
        for X_Y_batch, _, _ in all_loader:
            B = len(X_Y_batch)
            for i in range(B):
                X_Y = torch.tensor(X_Y_batch[i], dtype=torch.float32)

                if data_name == "RiskC":
                    behavior_data = np.mean(
                        (X_Y[:, 1] > X_Y[:, 3]).cpu().numpy().astype(int)
                        == X_Y[:, -1].cpu().numpy().astype(int)
                    )
                    risky_choice.append(behavior_data)
                elif data_name == "DelayC":
                    behavior_data = np.mean(X_Y[:, -1].cpu().numpy().astype(int))
                    delay_choice.append(behavior_data)
                elif data_name == "RiskP":
                    x1 = X_Y[:, 0].cpu().numpy()
                    x2 = X_Y[:, 1].cpu().numpy()
                    y = X_Y[:, -1].cpu().numpy()
                    m = LinearRegression()
                    m.fit(np.column_stack((x1, x2)), y)
                    risky_price1.append(m.coef_[0])
                    risky_price2.append(m.coef_[1])
                    risky_price3.append(m.intercept_)
                elif data_name == "DelayP":
                    x1 = X_Y[:, 0].cpu().numpy()
                    x2 = X_Y[:, 1].cpu().numpy()
                    y = X_Y[:, -1].cpu().numpy()
                    m = LinearRegression()
                    m.fit(np.column_stack((x1, x2)), y)
                    delay_price1.append(m.coef_[0])
                    delay_price2.append(m.coef_[1])
                    delay_price3.append(m.intercept_)

    if data_name == "RiskC":
        metrics["risky_choice"] = np.asarray(risky_choice)
    elif data_name == "DelayC":
        metrics["delay_choice"] = np.asarray(delay_choice)
    elif data_name == "RiskP":
        metrics["risky_price1"] = np.asarray(risky_price1)
        metrics["risky_price2"] = np.asarray(risky_price2)
        metrics["risky_price3"] = np.asarray(risky_price3)
    elif data_name == "DelayP":
        metrics["delay_price1"] = np.asarray(delay_price1)
        metrics["delay_price2"] = np.asarray(delay_price2)
        metrics["delay_price3"] = np.asarray(delay_price3)

    return metrics

task_frames = {}

for data_name, beta, lambda_sparse, n_z in zip(
    all_task_data_name, all_task_beta, all_task_lambda_sparse, all_task_n_z
):
    load_path = get_path(data_name, beta, lambda_sparse)
    print(f"\nLoading [{data_name}] -> {load_path}")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Missing file for task {data_name}: {load_path}")

    with open(load_path, "rb") as f:
        data = pickle.load(f)

    all_mu = extract_all_mu(data)
    all_subj = load_subject_ids(data_name)

    if all_mu.shape[0] != len(all_subj):
        raise ValueError(
            f"Subject count mismatch in {data_name}: all_mu={all_mu.shape[0]}, all_subj={len(all_subj)}"
        )

    z_vars = np.var(all_mu, axis=0)
    rank_desc = np.argsort(z_vars)[::-1]
    n_use = int(max(1, min(n_z, all_mu.shape[1])))
    pick_idx = rank_desc[:n_use]

    print(f"{data_name} var(mu) per dim: {np.round(z_vars, 4)}")
    print(f"{data_name} picked top-{n_use} z dims: {pick_idx.tolist()}")
    print(f"{data_name} active dims (var >= {threshold}): {np.where(z_vars >= threshold)[0].tolist()}")

    df_task = pd.DataFrame({"subj": all_subj})
    for k, z_idx in enumerate(pick_idx, start=1):
        df_task[f"z{k}_{data_name}"] = all_mu[:, z_idx]

    behavior_metrics = extract_behavior_metrics(data_name, data)
    for key, vals in behavior_metrics.items():
        if len(vals) != len(df_task):
            raise ValueError(f"Behavior length mismatch for {data_name}:{key}")
        df_task[key] = vals

    task_frames[data_name] = df_task

df_merged = task_frames[all_task_data_name[0]]
for data_name in all_task_data_name[1:]:
    df_merged = df_merged.merge(task_frames[data_name], on="subj", how="inner")

print(f"\nCommon subjects across 4 tasks: {len(df_merged)}")

z_cols = []
for data_name, n_z in zip(all_task_data_name, all_task_n_z):
    for k in range(1, n_z + 1):
        z_cols.append(f"z{k}_{data_name}")

X2 = df_merged[z_cols].values
X2_std = StandardScaler().fit_transform(X2)
node_names = z_cols.copy()

n_factors = 3
fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax", method="principal")
fa.fit(X2_std)

loadings = fa.loadings_
communalities = fa.get_communalities()
uniqueness = fa.get_uniquenesses()
variance_info = fa.get_factor_variance()

print("\nFactor Loadings Matrix (Rotated):")
print(np.round(loadings, 3))
print("\nCommunalities:")
print(np.round(communalities, 3))
print("\nUniquenesses:")
print(np.round(uniqueness, 3))
print("\nVariance Explained by Factors:")
print("Variance per factor:", np.round(variance_info[0], 3))
print("Proportion per factor:", np.round(variance_info[1], 3))
print("Cumulative proportion:", np.round(variance_info[2], 3))

extra_names = [
    "risky_choice",
    "risky_price1",
    "risky_price2",
    "risky_price3",
    "delay_choice",
    "delay_price1",
    "delay_price2",
    "delay_price3",
]

for key in extra_names:
    if key not in df_merged.columns:
        raise KeyError(f"Missing behavior column after merge: {key}")

color_tasks = ["#b2dfdb", "#c5e1a5", "#ffe082", "#ffab91"]
color_bhs = [
    color_tasks[0],
    color_tasks[1],
    color_tasks[1],
    color_tasks[1],
    color_tasks[2],
    color_tasks[3],
    color_tasks[3],
    color_tasks[3],
]

label_bhs = [
    r"$P_{risky}$",
    r"$\beta_{money}$",
    r"$\beta_{prob}$",
    r"$\beta_{intercept}$",
    r"$P_{delay}$",
    r"$\beta_{money}$",
    r"$\beta_{time}$",
    r"$\beta_{intercept}$",
]

TASK_FACTOR_LABELS = {
    "RiskC": "Risky Choice",
    "DelayC": "Intertemporal Choice",
    "RiskP": "Risky Pricing",
    "DelayP": "Intertemporal Pricing",
}

def make_mid_labels(z_columns):

    labels = []
    for c in z_columns:
        part = c.split("_")
        if len(part) >= 2:
            z_tag = part[0]
            task = "_".join(part[1:])
            k = z_tag[1:] if z_tag.startswith("z") else z_tag
            task_name = TASK_FACTOR_LABELS.get(task, task)
            labels.append(f"{task_name}\nFactor {k}")
        else:
            labels.append(c)
    return labels

label_mid = make_mid_labels(node_names)
task_color_map = {
    "RiskC": color_tasks[0],
    "DelayC": color_tasks[2],
    "RiskP": color_tasks[1],
    "DelayP": color_tasks[3],
}
color_mid = []
for c in node_names:
    task = c.split("_", 1)[1]
    color_mid.append(task_color_map.get(task, "#dddddd"))

def plot_factor_analysis_3layer(
    loadings_in,
    data_matrix,
    df_extra,
    threshold=0.4,
    radius=0.25,
    fac_gap=3.0,
    mid_gap=4.0,
    bot_gap=2.5,
    depth_scale=0.1,
    alpha=0.05,
    show_bottom_arcs=False,
):
    n_mid, n_fac = loadings_in.shape
    n_bot = len(extra_names)

    fig, ax = plt.subplots(figsize=(max(12, n_mid * 1.2), 9))

    mid_width = (n_mid - 1) * mid_gap
    fac_width = (n_fac - 1) * fac_gap
    bot_width = (n_bot - 1) * bot_gap
    center = max(mid_width, fac_width, bot_width) / 2
    mid_offset = center - mid_width / 2
    fac_offset = center - fac_width / 2
    bot_offset = center - bot_width / 2

    fac_pos = {i: (fac_offset + i * fac_gap, 5.0) for i in range(n_fac)}
    for i in range(n_fac):
        x, y = fac_pos[i]
        circ = Circle((x, y), radius, edgecolor="black", facecolor="skyblue", zorder=3)
        ax.add_patch(circ)
        ax.text(x, y + 0.35, f"F{i+1}", ha="center", va="bottom", fontsize=12, weight="bold")

    feat_pos = {j: (mid_offset + j * mid_gap, 2.5) for j in range(n_mid)}
    for j in range(n_mid):
        x, y = feat_pos[j]
        circ = Circle((x, y), radius, edgecolor="black", facecolor=color_mid[j], zorder=3)
        ax.add_patch(circ)
        ax.text(x, y - 0.55, label_mid[j], ha="center", va="top", fontsize=12)

    rect_w, rect_h = 2.0, 0.6
    bot_pos = {}
    for k, name in enumerate(extra_names):
        x, y = bot_offset + k * bot_gap, 0.0
        bot_pos[k] = (x, y)
        rect = Rectangle(
            (x - rect_w / 2, y - rect_h / 2),
            rect_w,
            rect_h,
            edgecolor="black",
            facecolor=color_bhs[k],
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x, y, label_bhs[k], ha="center", va="center", fontsize=12, color="black")

    for j in range(n_mid):
        max_idx = np.argmax(np.abs(loadings_in[j, :]))
        for i in range(n_fac):
            w = loadings_in[j, i]
            if abs(w) >= threshold or i == max_idx:
                x1, y1 = fac_pos[i]
                x2, y2 = feat_pos[j]
                ax.plot([x1, x2], [y1, y2], color="gray", linewidth=abs(w) * 5, alpha=0.7, zorder=1)

    for j in range(n_mid):
        for k, name in enumerate(extra_names):
            r, p = pearsonr(data_matrix[:, j], df_extra[name].values)
            if np.isnan(r) or p >= alpha:
                continue
            x1, y1 = feat_pos[j]
            x2, y2 = bot_pos[k]
            lw = 0.5 + abs(r) * 4
            ax.plot([x1, x2], [y1, y2], color="steelblue", linewidth=lw, alpha=0.6, zorder=1.5)

    def arcs_mid_above(pos_dict, values_matrix, color="darkgreen"):
        keys = list(pos_dict.keys())
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                ia, ib = keys[a], keys[b]
                r, p = pearsonr(values_matrix[:, a], values_matrix[:, b])
                if np.isnan(r) or p >= alpha:
                    continue
                (x1, y1), (x2, y2) = pos_dict[ia], pos_dict[ib]
                dx = x2 - x1
                lift = depth_scale * abs(dx)
                verts = [
                    (x1, y1 + radius),
                    (x1 + 0.25 * dx, y1 + radius + lift),
                    (x2 - 0.25 * dx, y2 + radius + lift),
                    (x2, y2 + radius),
                ]
                path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
                ax.add_patch(
                    PathPatch(
                        path,
                        lw=0.5 + abs(r) * 4,
                        edgecolor=color,
                        facecolor="none",
                        alpha=0.6,
                        zorder=2.5,
                    )
                )
                ax.text(
                    (x1 + x2) / 2,
                    y1 + radius + lift * 0.45,
                    f"{r:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color=color,
                    zorder=3,
                )

    def arcs_bot_below(pos_dict, values_matrix, color="darkblue"):
        keys = list(pos_dict.keys())
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                ia, ib = keys[a], keys[b]
                r, p = pearsonr(values_matrix[:, a], values_matrix[:, b])
                if np.isnan(r) or p >= alpha:
                    continue
                (x1, y1), (x2, y2) = pos_dict[ia], pos_dict[ib]
                dx = x2 - x1
                drop = depth_scale * abs(dx)
                verts = [
                    (x1, y1 - rect_h / 2),
                    (x1 + 0.25 * dx, y1 - rect_h / 2 - drop),
                    (x2 - 0.25 * dx, y2 - rect_h / 2 - drop),
                    (x2, y2 - rect_h / 2),
                ]
                path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
                ax.add_patch(
                    PathPatch(
                        path,
                        lw=0.5 + abs(r) * 4,
                        edgecolor=color,
                        facecolor="none",
                        alpha=0.6,
                        zorder=0.5,
                    )
                )
                ax.text(
                    (x1 + x2) / 2,
                    y1 - rect_h / 2 - drop * 0.9,
                    f"{r:.2f}",
                    ha="center",
                    va="top",
                    fontsize=10,
                    color=color,
                    zorder=1,
                )

    arcs_mid_above(feat_pos, data_matrix, color="darkgreen")
    if show_bottom_arcs:
        arcs_bot_below(bot_pos, df_extra[extra_names].values, color="darkblue")

    ax.set_xlim(-1, center * 2 + 1)
    ax.set_ylim(-3, 6.0)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.show()

plot_factor_analysis_3layer(
    loadings,
    X2_std,
    df_merged,
    threshold=0.4,
    alpha=0.05,
    show_bottom_arcs=False,
)

n_features = X2_std.shape[1]

fa_init = FactorAnalyzer(rotation=None, n_factors=n_features, method="principal")
fa_init.fit(X2_std)

ev, _ = fa_init.get_eigenvalues()
prop_var = ev / np.sum(ev)
cum_var = np.cumsum(prop_var)

n_subjects = X2_std.shape[0]
n_rep = 1000
rand_ev = np.zeros((n_rep, n_features))

for i in range(n_rep):
    rand_data = np.random.normal(size=(n_subjects, n_features))
    fa_rand = FactorAnalyzer(rotation=None, n_factors=n_features, method="principal")
    fa_rand.fit(rand_data)
    rand_ev[i, :] = fa_rand.get_eigenvalues()[0]

mean_rand_ev = np.mean(rand_ev, axis=0)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

ax = axes[0]
ax.scatter(range(1, n_features + 1), ev, marker="o", color="black")
ax.plot(range(1, n_features + 1), ev, linestyle="--", color="black")
ax.axhline(1, color="gray", linestyle="--", linewidth=1)
ax.set_title("Scree Plot (EFA)")
ax.set_xlabel("Factor Number")
ax.set_ylabel("Eigenvalue")

ax = axes[1]
ax.plot(range(1, n_features + 1), cum_var, marker="o", color="black")
ax.axhline(0.7, color="gray", linestyle="--", linewidth=1)
ax.set_title("Cumulative Variance Explained")
ax.set_xlabel("Factor Number")
ax.set_ylabel("Cumulative Variance")

ax = axes[2]
ax.plot(range(1, n_features + 1), ev, marker="o", color="black", label="Real Data")
ax.plot(range(1, n_features + 1), mean_rand_ev, marker="x", color="dimgray", label="Random Data")
ax.axhline(1, color="gray", linestyle="--", linewidth=1)
ax.set_title("Parallel Analysis")
ax.set_xlabel("Factor Number")
ax.set_ylabel("Eigenvalue")
ax.legend()

plt.tight_layout()
plt.show()

var_exp = []
for n in range(1, 6):
    fa_tmp = FactorAnalyzer(n_factors=n, rotation=None, method="principal")
    fa_tmp.fit(X2_std)
    _, _, cum_var_tmp = fa_tmp.get_factor_variance()
    var_exp.append(cum_var_tmp[-1])

var_exp = np.array(var_exp)

plt.figure(figsize=(4, 2.5))
plt.plot(range(1, 6), var_exp, marker="o", color="black", linestyle="-")
plt.axhline(0.70, color="gray", linestyle="--", linewidth=1)
plt.xticks(range(1, 6))
plt.ylim(0, 1)
plt.xlabel("Number of Factors")
plt.ylabel("Variance Explained")
plt.show()

for n_factors in range(1, 6):
    fa_tmp = FactorAnalyzer(n_factors=n_factors, rotation="varimax", method="principal")
    fa_tmp.fit(X2_std)
    loadings_tmp = fa_tmp.loadings_
    print(f"\n===== {n_factors} Factors (Varimax rotation) =====")
    print(np.round(loadings_tmp, 3))
