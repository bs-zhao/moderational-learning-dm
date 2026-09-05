"""
Simulation 2 (Hete4bn4): θ recovery scatter (first 3 panels) and top-3 μ histograms.
Mirrors figure_bc1.py panels without the 6-panel combined layout.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
from neural_processes import *
from loss_func import *
from tools import *

import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

np.random.seed(31)
torch.manual_seed(31)
torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    torch.cuda.manual_seed(31)
    torch.cuda.manual_seed_all(31)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH = 1
data_name = "Hete4bn4"

n_subjects = 2000
min_trials = 100
max_trials = 200
pick_min = 50
pick_max = 150
emb_size = 32
l1_lambda = 0
l2_lambda = 0
beta = 1
input_dim = 4
latent_dim = 16
repeat = 1
n_panels = 3

if BATCH == 1:
    name_save = (
        f"save/train_batch/{data_name}_latent_{latent_dim}_emb_{emb_size}_"
        f"l1_{l1_lambda}_l2_{l2_lambda}_beta_{beta}_nsubj_{n_subjects}_"
        f"min_{min_trials}_max_{max_trials}_input_{input_dim}_"
        f"minpick_{pick_min}_maxpick_{pick_max}_repeat_{repeat}.pkl"
    )
else:
    name_save = (
        f"save/train/{data_name}_latent_{latent_dim}_emb_{emb_size}_"
        f"l1_{l1_lambda}_l2_{l2_lambda}_beta_{beta}_nsubj_{n_subjects}_"
        f"min_{min_trials}_max_{max_trials}_input_{input_dim}_"
        f"minpick_{pick_min}_maxpick_{pick_max}_repeat_{repeat}.pkl"
    )

print(name_save)
with open(name_save, 'rb') as f:
    data = pickle.load(f)

if BATCH == 1:
    model = data['model']
else:
    model = oldBP(x_dim=input_dim, y_dim=1, latent_dim=latent_dim, emb_size=emb_size).to(device)

model.load_state_dict(data['model_state_dict'])
model.eval()
test_loader = data['test_loader']

co_test = []
mus_test = []

max_len_trials = data['max_trials']
max_len_trials_outer = max_len_trials
x_dim = 4
y_dim = 1
outer_dim = x_dim + y_dim

with torch.no_grad():
    for X_Y_batch, X_Y_batch_outer, labels_batch in test_loader:
        B = len(X_Y_batch)
        XY_context_batch = torch.zeros(B, max_len_trials, x_dim + y_dim, device=device, dtype=torch.float64)
        XY_outer_batch = torch.zeros(B, max_len_trials_outer, outer_dim, device=device, dtype=torch.float64)
        XY_target_batch = torch.zeros(B, max_len_trials, x_dim + y_dim, device=device, dtype=torch.float64)
        trial_mask_context = torch.zeros(B, max_len_trials, dtype=torch.bool, device=device)
        trial_mask_outer = torch.zeros(B, max_len_trials_outer, dtype=torch.bool, device=device)
        trial_mask_target = torch.zeros(B, max_len_trials, dtype=torch.bool, device=device)

        for i in range(B):
            X_Y = X_Y_batch[i].to(device=device, dtype=torch.float64)
            X_Y_outer = X_Y_batch_outer[i].to(device=device, dtype=torch.float64)
            X_Y_context = X_Y[:, :]
            X_Y_target = X_Y.clone()

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
        co_test += labels_batch
        mus_test += mu.tolist()

co_test = np.vstack(co_test)
mus_test = np.vstack(mus_test)

X = co_test
Y = mus_test
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)
Y_sorted = mus_test[:, mus_test.std(axis=0).argsort()[::-1]]

axis_label_size = 14
tick_label_size = 12
title_size = 14
suptitle_size = 16
wspace = 0.55
figsize = (6.5, 2.2)
figsize2 = (7, 2.5)
scatter_alpha = 0.4
scatter_s = 20

def _style_axes(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size)

def _annotate_r(ax, y, yh):
    """Same placement/style as draw_sim1_recovery_v2 in tools.py."""
    xy_min = min(y.min(), yh.min())
    xy_max = max(y.max(), yh.max())
    ax.text(
        xy_min + 0.55 * (xy_max - xy_min),
        xy_min + 0.05 * (xy_max - xy_min),
        f"$r$ = {np.corrcoef(y, yh)[0, 1]:.3f}",
        fontsize=axis_label_size,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

fig1, axes1 = plt.subplots(1, n_panels, figsize=figsize2)
axes1 = np.atleast_1d(axes1).flatten()

for i in range(n_panels):
    y = X_scaled[:, i]
    x = Y_scaled
    ols = sm.OLS(y, x).fit()
    yh = ols.predict(x)

    axes1[i].scatter(
        y, yh, alpha=scatter_alpha, color="#4C72B0", s=scatter_s, linewidths=0
    )

    xy_min = min(y.min(), yh.min())
    xy_max = max(y.max(), yh.max())
    pad = 0.05 * (xy_max - xy_min)
    xy_min -= pad
    xy_max += pad

    axes1[i].set_xlim(xy_min, xy_max)
    axes1[i].set_ylim(xy_min, xy_max)
    axes1[i].set_aspect("equal", adjustable="box")
    axes1[i].plot([xy_min, xy_max], [xy_min, xy_max], color="black", linewidth=1, zorder=0)

    axes1[i].set_xlabel("True", fontsize=axis_label_size)
    if i == 0:
        axes1[i].set_ylabel("Predicted", fontsize=axis_label_size)
    axes1[i].set_title(f"$\\theta_{{{i+1}}}$", fontsize=title_size)
    _annotate_r(axes1[i], y, yh)
    _style_axes(axes1[i])

plt.subplots_adjust(wspace=wspace)
fig1.text(0.00, 1.00, "", fontsize=suptitle_size, fontweight="bold", ha="left", va="top")
plt.show()

fig2, axes2 = plt.subplots(1, n_panels, figsize=figsize)
axes2 = np.atleast_1d(axes2).flatten()

for i in range(n_panels):
    axes2[i].hist(
        Y_sorted[:, i],
        bins=35,
        alpha=0.7,
        color="#55A868",
        density=True,
    )
    axes2[i].set_title(f"$\\mu_{{{i+1}}}$", fontsize=title_size)
    axes2[i].set_xlabel("Value", fontsize=axis_label_size)
    if hasattr(axes2[i], "set_box_aspect"):
        axes2[i].set_box_aspect(1)
    else:
        axes2[i].set_aspect("equal", adjustable="box")
    if i == 0:
        axes2[i].set_ylabel("Density", fontsize=axis_label_size)
    _style_axes(axes2[i])

plt.subplots_adjust(wspace=wspace)
fig2.text(0.00, 1.00, "", fontsize=suptitle_size, fontweight="bold", ha="left", va="top")
plt.show()
