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
from torch.utils.data import DataLoader
from neural_processes_batch_new import basicNP_D2_outer

import torch

from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN; from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN; from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering

np.random.seed(31)
torch.manual_seed(31)
torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    torch.cuda.manual_seed(31)
    torch.cuda.manual_seed_all(31)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH = 1

data_name = ["Flex6Lv2", "Flex6Lv2b", "Hete4", "Hete4bn4"][3]

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

norm = 1

input_dim = 4
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

co_test = []
mus_test = []
lvs_test = []

if BATCH != 1:
    model.eval()
    with torch.no_grad():

        for X_Y, labels in test_loader:
            X_Y = X_Y.to(device)

            X_Y_context = X_Y[:, :, :]
            X_Y_target = X_Y[:, :, :]
            Y_target = X_Y_target[0,:,-1]

            Yh, mu, log_var = model(X_Y_context, X_Y_target)

            co_test.append(labels.cpu().numpy())
            mus_test.append(mu.cpu().numpy())
            lvs_test.append(log_var.squeeze(0).detach().cpu().numpy())

else:

    max_len_trials = data['max_trials']
    max_len_trials_outer = max_len_trials
    if "Flex6" in data_name:
        x_dim = 6
    if "Hete" in data_name:
        x_dim = 4

    y_dim = 1
    outer_dim = x_dim + y_dim

    with torch.no_grad():

        for X_Y_batch, X_Y_batch_outer, labels_batch in all_loader:
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

mus_test_sorted = mus_test[:, mus_test.std(axis=0).argsort()[::-1]]
for i in range(3):
    plt.hist(mus_test_sorted[:, i], bins=50, alpha=0.7, color="blue")
    plt.show()

X = co_test
Y = mus_test

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

Y_scaled = PCA(n_components=3).fit_transform(Y_scaled)

nc = min(X.shape[1], Y.shape[1])
cca = CCA(n_components=nc)

cca.fit(X_scaled, Y_scaled)

X_c, Y_c = cca.transform(X_scaled, Y_scaled)

canonical_correlations = []
for i in range(nc):
    corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
    canonical_correlations.append(corr)

print(":", canonical_correlations)

fig, axes = plt.subplots(1, nc, figsize=(12, 3))

for i in range(nc):

    axes[i].scatter(X_c[:, i], Y_c[:, i], alpha=0.7)

    axes[i].set_title(f"Canonical Component {i}")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, nc, figsize=(12, 8))

for i in range(nc):

    axes[0, i].hist(X_c[:, i], bins=30, alpha=0.7, color='blue')
    axes[0, i].set_title(f"X_c Canonical Component {i}")

    axes[1, i].hist(Y_c[:, i], bins=30, alpha=0.7, color='red')
    axes[1, i].set_title(f"Y_c Canonical Component {i}")

plt.tight_layout()
plt.show()

n_co = 3

fig, axes = plt.subplots(nc, n_co, figsize=(12, 10))

if nc == 1:

    axes = [axes]
if input_dim == 1:

    axes = [[ax] for ax in axes]

for i in range(nc):
    for j in range(n_co):

        axes[i][j].scatter(X[:, j], X_c[:, i], alpha=0.7)

        axes[i][j].set_title(f"Component {i} vs Feature X{j}")

plt.tight_layout()
plt.show()

L = []
for l in co_test:
    t = l[-1]
    L.append(int(t.item()))

scaler_cluster = StandardScaler()
mus_test_scaled = scaler_cluster.fit_transform(mus_test)
mus_test_scaled = mus_test.copy()

for_cluster_3d = PCA(n_components=3).fit_transform(mus_test_scaled)

x = for_cluster_3d[:, 0]
y = for_cluster_3d[:, 1]
z = for_cluster_3d[:, 2]

X_cluster = np.column_stack([x, y, z])

L2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_cluster)

sil_score = silhouette_score(X_cluster, L2)

if len(L) > 0:
    ari_score = adjusted_rand_score(L, L2)
    nmi_score = normalized_mutual_info_score(L, L2)

    print(f"K-means:")
    print(f"  : {sil_score:.4f}")
    print(f"   (): {ari_score:.4f}")
    print(f"   (): {nmi_score:.4f}")
else:
    print(f"K-means:")
    print(f"  : {sil_score:.4f}")

fig2 = plt.figure(figsize=(7.5, 5.5))

ax1 = fig2.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(x, y, z, c=L, cmap='viridis', alpha=0.4, s=5)
ax1.set_xlabel('PC1', fontsize=10, labelpad=-8)
ax1.set_ylabel('PC2', fontsize=10, labelpad=-2)
ax1.set_zlabel('PC3', fontsize=10, labelpad=-2)
ax1.set_title('Grouped by True Label', fontsize=16, y=1.1)
ax1.view_init(elev=20, azim=95)

ax2 = fig2.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(x, y, z, c=L2, cmap='tab10', alpha=0.4, s=5)
ax2.set_xlabel('PC1', fontsize=10, labelpad=-8)
ax2.set_ylabel('PC2', fontsize=10, labelpad=-2)
ax2.set_zlabel('PC3', fontsize=10, labelpad=-2)
ax2.set_title(f'Grouped by K-means Clustering', fontsize=16, y=1.1)
ax2.view_init(elev=20, azim=95)

plt.subplots_adjust(wspace=0.0)

for ax in [ax1, ax2]:
    ax.tick_params(axis='x', pad=0)

for ax in [ax1, ax2]:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

plt.show()

mus_test_scaled = scaler_Y.transform(mus_test)

for_cluster = PCA(n_components=2).fit_transform(mus_test_scaled)

x = for_cluster[:, 0]
y = for_cluster[:, 1]

X_cluster = np.column_stack([x, y])

from sklearn.cluster import SpectralClustering
L2 = SpectralClustering(n_clusters=2, affinity="rbf", gamma=10, random_state=0).fit_predict(X_cluster)

sil_score = silhouette_score(X_cluster, L2)

if len(L) > 0:
    ari_score = adjusted_rand_score(L, L2)
    nmi_score = normalized_mutual_info_score(L, L2)

    print(f"K-means:")
    print(f"  : {sil_score:.4f}")
    print(f"   (): {ari_score:.4f}")
    print(f"   (): {nmi_score:.4f}")
else:
    print(f"K-means:")
    print(f"  : {sil_score:.4f}")

fig2 = plt.figure(figsize=(15, 6))

ax1 = fig2.add_subplot(121)
scatter1 = ax1.scatter(x, y, c=L, cmap='viridis', alpha=0.7)
ax1.set_xlabel('M1')
ax1.set_ylabel('M2')
ax1.set_title('μs output by the encoder')

ax2 = fig2.add_subplot(122)
scatter2 = ax2.scatter(x, y, c=L2, cmap='tab10', alpha=0.7)
ax2.set_xlabel('M1')
ax2.set_ylabel('M2')
ax2.set_title(f'K-means Clustering\nSilhouette Score: {sil_score:.3f}')

plt.subplots_adjust(wspace=0.01)
plt.show()

L = []
for l in co_test:
    t = l[-1]
    L.append(int(t.item()))

scaler_cluster = StandardScaler()
mus_test_scaled = scaler_cluster.fit_transform(mus_test)
mus_test_scaled = mus_test.copy()

for_cluster = PCA(n_components=2).fit_transform(mus_test_scaled)

x = for_cluster[:, 0]
y = for_cluster[:, 1]

X_cluster = for_cluster[:, :3]

L2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_cluster)

sil_score = silhouette_score(X_cluster, L2)

if len(L) > 0:
    ari_score = adjusted_rand_score(L, L2)
    nmi_score = normalized_mutual_info_score(L, L2)

    print(f"K-means:")
    print(f"  : {sil_score:.4f}")
    print(f"   (): {ari_score:.4f}")
    print(f"   (): {nmi_score:.4f}")
else:
    print(f"K-means:")
    print(f"  : {sil_score:.4f}")

fig2, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))

axes[0].scatter(
    x, y,
    c=L, cmap='viridis',
    alpha=0.5, s=8, linewidths=0
)
axes[0].set_title("Grouped by True Label", fontsize=14)
axes[0].set_xlabel("PC1", fontsize=11)
axes[0].set_ylabel("PC2", fontsize=11)

axes[1].scatter(
    x, y,
    c=L2, cmap='tab10',
    alpha=0.5, s=8, linewidths=0
)
axes[1].set_title("Grouped by K-means Clustering", fontsize=14)
axes[1].set_xlabel("PC1", fontsize=11)
axes[1].set_ylabel("PC2", fontsize=11)

for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=10)

plt.subplots_adjust(wspace=0.25)
plt.show()
