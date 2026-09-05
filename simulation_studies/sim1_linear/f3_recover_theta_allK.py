import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
from neural_processes import *
from loss_func import *
from tools import *

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.autolayout'] = False
matplotlib.rcParams['figure.constrained_layout.use'] = False
matplotlib.rcParams['savefig.bbox'] = 'standard'
matplotlib.rcParams['savefig.pad_inches'] = 0.02
matplotlib.rcParams['savefig.dpi'] = 300
from torch.utils.data import DataLoader
from neural_processes_batch_new import basicNP_D2_outer

import torch

from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

np.random.seed(31)
torch.manual_seed(31)
torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    torch.cuda.manual_seed(31)
    torch.cuda.manual_seed_all(31)
import statsmodels.api as sm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH = 1

data_name = "Flex6Lv2b"

n_subjects = 2000
min_trials = 100
max_trials = 200

pick_min = 50
pick_max = 150

emb_size = 32

l1_lambda = 0
l2_lambda = 0

p_valid = 0.15
p_test = 0.15
patience = 50
epochs = 1000
learning_rate = 5e-4

norm = 0

input_dim = 6
beta = 1
latent_dim = 16
repeat = 1

if BATCH == 1:
    name_save = f"save/train_batch/{data_name}_latent_{latent_dim}_emb_{emb_size}_l1_{l1_lambda}_l2_{l2_lambda}_beta_{beta}_nsubj_{n_subjects}_min_{min_trials}_max_{max_trials}_input_{input_dim}_minpick_{pick_min}_maxpick_{pick_max}_repeat_{repeat}.pkl"
else:
    name_save = f"save/train/{data_name}_latent_{latent_dim}_emb_{emb_size}_l1_{l1_lambda}_l2_{l2_lambda}_beta_{beta}_nsubj_{n_subjects}_min_{min_trials}_max_{max_trials}_input_{input_dim}_minpick_{pick_min}_maxpick_{pick_max}_repeat_{repeat}.pkl"

print(name_save)

with open(name_save, 'rb') as f:
    data = pickle.load(f)

if BATCH == 1:
    model = data['model']
else:
    if "Flex6" in data_name:
        model = oldBP(x_dim=6, y_dim=1, latent_dim=latent_dim, emb_size=emb_size).to(device)
    else:
        model = oldBP(x_dim=input_dim, y_dim=1, latent_dim=latent_dim, emb_size=emb_size).to(device)

model.load_state_dict(data['model_state_dict'])
model.eval()

train_loader = data['train_loader']
val_loader = data['val_loader']
test_loader = data['test_loader']
all_loader = data['all_loader']

all_Y_real = []
all_Y_pred = []

model.eval()

max_len_trials = data['max_trials']
max_len_trials_outer = max_len_trials
if "Flex6" in data_name:
    x_dim = 6
if "Hete" in data_name:
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
            total_trials = int(X_Y.shape[0])

            X_Y_context = X_Y[50:, :]
            X_Y_target = X_Y[:50, :]
            Y_target = X_Y_target[:,-y_dim:]

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

        Yh, mu, log_var = model(
            XY_context_batch,
            XY_target_batch,
            XY_outer=XY_outer_batch,
            trial_mask_context=trial_mask_context,
            trial_mask_outer=trial_mask_outer,
            trial_mask_target=trial_mask_target,
        )
        Y_target_batch = XY_target_batch[:, :, -y_dim:]
        if "beta" == "beta":
            loss, components = beta_mse_loss_batch(
            Yh,
            Y_target_batch,
            mu=mu, log_var=log_var, beta=beta,
            trial_mask=trial_mask_target,
            return_components=True
        )
        elif loss_type == "info":
            pass

        for b in range(B):
            mask_b = trial_mask_target[b]
            n_trial_b = mask_b.sum()
            all_Y_real.append(Y_target_batch[b, :n_trial_b].cpu().numpy())
            all_Y_pred.append(Yh[b, :n_trial_b].cpu().numpy())

co_test = []
mus_test = []
lvs_test = []

model.eval()

max_len_trials = data['max_trials']
max_len_trials_outer = max_len_trials
if "Flex6" in data_name:
    x_dim = 6
if "Hete" in data_name:
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
            total_trials = int(X_Y.shape[0])

            X_Y_context = X_Y[:, :]
            X_Y_target = X_Y.clone()
            Y_target = X_Y_target[:,-y_dim:]

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

        Yh, mu, log_var = model(
            XY_context_batch,
            XY_target_batch,
            XY_outer=XY_outer_batch,
            trial_mask_context=trial_mask_context,
            trial_mask_outer=trial_mask_outer,
            trial_mask_target=trial_mask_target,
        )
        Y_target_batch = XY_target_batch[:, :, -y_dim:]
        if "beta" == "beta":
            loss, components = beta_mse_loss_batch(
            Yh,
            Y_target_batch,
            mu=mu, log_var=log_var, beta=beta,
            trial_mask=trial_mask_target,
            return_components=True
        )
        elif loss_type == "info":
            pass

        co_test += labels_batch
        mus_test += mu.tolist()
        lvs_test += log_var.tolist()

co_test = np.vstack(co_test)
mus_test = np.vstack(mus_test)
lvs_test = np.vstack(lvs_test)

X = co_test
Y = mus_test

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

base_size = 2
ax_w_in = base_size
ax_h_in = base_size
left_in = 0.70
right_in = 0.20
bottom_in = 0.70
top_in = 0.45
wspace_in = 0.40

fig_w_in = input_dim * ax_w_in + max(0, input_dim - 1) * wspace_in + left_in + right_in
fig_h_in = ax_h_in + top_in + bottom_in

fig, axes = plt.subplots(1, input_dim, figsize=(fig_w_in, fig_h_in))
axes = np.atleast_1d(axes)

for i in range(input_dim):
    y = X_scaled[:, i]
    x = Y_scaled[:, :]

    model = sm.OLS(y, x).fit()
    print(model.summary())
    yh = model.predict(x)
    axes[i].scatter(y, yh, alpha=0.6, color="#C44E52", s=25, linewidths=0)
    axes[i].set_xlabel("True", fontsize=15)
    if i == 0:
        axes[i].set_ylabel("Predicted", fontsize=16)
    axes[i].set_title(f"θ{i+1}", fontsize=16)
    try:
        axes[i].set_box_aspect(1)
    except Exception:
        axes[i].set_aspect('equal', adjustable='box')
    r = np.corrcoef(y, yh)[0, 1]
    axes[i].text(0.95, 0.05, f"$r$ = {r:.3f}",
                 transform=axes[i].transAxes,
                 ha="right", va="bottom",
                 fontsize=16)

    axes[i].tick_params(axis='x', labelsize=14)
    axes[i].tick_params(axis='y', labelsize=14)

    axes[i].spines['right'].set_visible(False)
    axes[i].spines['top'].set_visible(False)

    xmin, xmax = axes[i].get_xlim()
    ymin, ymax = axes[i].get_ylim()
    lo = min(xmin, ymin)
    hi = max(xmax, ymax)
    axes[i].plot([lo, hi], [lo, hi], color='black', linewidth=1.0)
    axes[i].set_xlim(xmin, xmax)
    axes[i].set_ylim(ymin, ymax)

left_frac = left_in / fig_w_in
right_frac = 1 - right_in / fig_w_in
bottom_frac = bottom_in / fig_h_in
top_frac = 1 - top_in / fig_h_in
wspace_frac = wspace_in / ax_w_in

fig.subplots_adjust(left=left_frac, right=right_frac, bottom=bottom_frac, top=top_frac, wspace=wspace_frac)

supersuper = f"k = {input_dim}"
supersuper_x = 0.00
supersuper_y = 0.98
fig.text(
    supersuper_x, supersuper_y, supersuper,
    fontsize=16, fontweight="bold", ha="left", va="top"
)

SAVE_FIG = False
if SAVE_FIG:
    out_path = f"save/figs/recover_theta_k{input_dim}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='standard', pad_inches=0.02)

print("Figure size (inches):", fig.get_size_inches())

plt.show()
