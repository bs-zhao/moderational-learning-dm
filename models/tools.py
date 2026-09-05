from torch.utils.data import Dataset

class regDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ListDataset(Dataset):
    def __init__(self, data_list, data_list_outer, label_list):
        assert len(data_list) == len(label_list), "data label"
        self.data_list = data_list
        self.data_list_outer = data_list_outer
        self.label_list = label_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.data_list_outer[idx], self.label_list[idx]

def collate_to_lists(batch):
    """batch data_list label_list
 Python /"""
    if not batch:
        return [], [], []
    data_batch, data_batch_outer, label_batch = zip(*batch)
    return list(data_batch), list(data_batch_outer), list(label_batch)

import numpy as np
import pickle
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from numpy.random import default_rng
from sklearn.decomposition import PCA

def checking_nlatent(X, norm=0, max_factors=8):

    max_factors = min(max_factors, X.shape[1])

    X = X.astype(float)
    if norm == 1:
        Xz = StandardScaler().fit_transform(X)
    else:
        Xz = X

    all_var = []
    for i in range(Xz.shape[1]):
        all_var.append(np.std(Xz[:,i]))
    all_var = np.array(all_var)
    all_var = np.sort(all_var)[::-1]

    efa_explained = []
    for k in range(1, max_factors + 1):
        fa = FactorAnalysis(n_components=k)
        Z = fa.fit_transform(Xz, y=None)
        L = fa.components_.T
        communalities = np.sum(L**2, axis=1)
        explained_ratio = communalities.sum() / Xz.shape[1]
        efa_explained.append(explained_ratio)

    def parallel_analysis(X, n_iter=500, random_state=0):
        rng = default_rng(random_state)
        n, d = X.shape

        _, s, _ = np.linalg.svd(X, full_matrices=False)
        eig_real = (s**2) / (n-1)

        rand_eigs = np.zeros((n_iter, d))
        for i in range(n_iter):
            X_rand = rng.standard_normal(size=X.shape)
            _, s_rand, _ = np.linalg.svd(X_rand, full_matrices=False)
            rand_eigs[i,:] = (s_rand**2) / (n-1)
        eig_rand = rand_eigs.mean(axis=0)
        return eig_real, eig_rand

    eig_real, eig_rand = parallel_analysis(Xz)

    n_parallel = np.sum(eig_real > eig_rand)
    print("Parallel Analysis suggests:", n_parallel, "factors")

    pca = PCA().fit(Xz)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    n_pca90 = np.searchsorted(cum_var, 0.9) + 1
    print("PCA 90% cumulative variance suggests:", n_pca90, "factors")

    def velicer_map(X, max_factors=8):
        n, d = X.shape
        R = np.corrcoef(X, rowvar=False)
        partials = []
        for k in range(1, max_factors+1):
            pca = PCA(n_components=k).fit(X)
            X_recon = pca.inverse_transform(pca.transform(X))
            resid = X - X_recon
            resid_corr = np.corrcoef(resid, rowvar=False)
            off_diag = resid_corr - np.diag(np.diag(resid_corr))
            partials.append((off_diag**2).mean())
        return partials

    partials = velicer_map(Xz, max_factors=max_factors)

    fig15, axes15 = plt.subplots(2, 3, figsize=(8, 10))
    ax00, ax01, ax02, ax10, ax11, ax12 = axes15[0,0], axes15[0,1], axes15[0,2], axes15[1,0], axes15[1,1], axes15[1,2]

    for ax in axes15.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax00.plot(range(1, len(all_var)+1), all_var, marker="o", color="black")
    ax00.set_xlabel("Latent")
    ax00.set_ylabel("Variance")
    ax00.set_title("Variance of Latents")

    ax01.plot(range(1, len(efa_explained)+1), efa_explained, marker="o", color="black")
    ax01.set_xlabel("Factors")
    ax01.set_ylabel("Cumulative variance explained")
    ax01.set_title("EFA (sklearn)")

    ax02.plot(range(1, len(eig_real)+1), eig_real, marker="o", color="black", label="Real eigenvalues")
    ax02.plot(range(1, len(eig_rand)+1), eig_rand, marker="s", color="gray", label="Random mean eigenvalues")
    ax02.axhline(1, color="gray", linestyle="--", linewidth=1, label="Kaiser λ>1")
    ax02.set_xlabel("Component")
    ax02.set_ylabel("Eigenvalue")
    ax02.set_title("Parallel Analysis")
    ax02.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1)

    ax10.plot(range(1, len(cum_var)+1), cum_var, marker="o", color="black", label="Cumulative variance")
    ax10.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="90% threshold")
    ax10.set_xlabel("Number of components")
    ax10.set_ylabel("Cumulative variance explained")
    ax10.set_title("PCA Scree Plot")
    ax10.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1)

    ax11.plot(range(1, len(partials)+1), partials, marker="o", color="black")
    ax11.set_xlabel("Number of factors")
    ax11.set_ylabel("Mean squared partial correlation")
    ax11.set_title("Velicer’s MAP Test")

    ax12.axis("off")

    suptitle_text_15 = "Component Selection Diagnostics"
    fig15.suptitle(suptitle_text_15, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.07, 1, 0.98])
    plt.show()

    k_map = np.argmin(partials) + 1
    print("MAP Test suggests:", k_map, "factors")

    print("\n==== Suggested number of factors/components ====")
    print("EFA cumulative variance (>=70%):", np.searchsorted(efa_explained, 0.7)+1)
    print("Parallel Analysis:", n_parallel)
    print("PCA 90% cumulative variance:", n_pca90)
    print("MAP Test:", k_map)

    eig_real2, eig_rand2 = parallel_analysis(Xz)
    pca2 = PCA().fit(Xz)
    explained_var2 = pca2.explained_variance_ratio_
    cum_var2 = np.cumsum(explained_var2)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 4.0))

    axes[0].plot(range(1, len(eig_real2)+1), eig_real2,
                marker="o", color="black", label="Real eigenvalues")
    if norm == 1:
        axes[0].plot(range(1, len(eig_rand2)+1), eig_rand2,
                    marker="s", color="gray", label="Random mean eigenvalues")
        axes[0].axhline(1, color="gray", linestyle="--", linewidth=1, label="Kaiser λ>1")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Eigenvalue")

    axes[0].legend(frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, -0.25), ncol=1)
    axes[0].set_title("PCA Scree Plot", fontsize=14)

    axes[1].plot(range(1, len(explained_var2)+1), explained_var2,
                marker="o", color="black", label="Proportion")
    axes[1].plot(range(1, len(cum_var2)+1), cum_var2,
                marker="o", linestyle="--", color="gray", label="Cumulative")

    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Variance explained")

    axes[1].legend(frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, -0.25), ncol=1)
    axes[1].set_title("PCA Variance Explained", fontsize=14)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    suptitle_text = "PCA Component Selection"
    title_top = 1

    plt.tight_layout(rect=[0, 0.05, 1, title_top])
    plt.show()

def plot_pca_variance(
    X, norm=0, max_factors=8,
    title="PCA Variance Explained",
    xlabel="Component",
    ylabel="Variance explained"
):
    "PCA +"
    X = X.astype(float)
    Xz = StandardScaler().fit_transform(X) if norm == 1 else X

    max_factors = min(max_factors, Xz.shape[1])

    pca = PCA(n_components=max_factors).fit(Xz)
    explained = pca.explained_variance_ratio_
    cum = np.cumsum(explained)

    fig, ax = plt.subplots(1, 1, figsize=(2.2, 3.2))

    ax.plot(range(1, len(explained)+1), explained,
            marker="o", color="black", label="Proportion")
    ax.plot(range(1, len(cum)+1), cum,
            marker="o", linestyle="--", color="gray", label="Cumulative")

    ax.set_xticks(range(1, len(explained)+1))

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.show()

    return {"explained_var": explained, "cum_var": cum}

def check_prediction(all_Y_real, all_Y_pred, cmap="tab20"):
    plt.figure(figsize=(4,4))

    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(all_Y_real)))

    for i in range(len(all_Y_real)):
        all_Y_real[i] = all_Y_real[i].squeeze()
        all_Y_pred[i] = all_Y_pred[i].squeeze()
        plt.scatter(all_Y_real[i], all_Y_pred[i],
                    alpha=0.4, s=20,
                    color=colors[i], label=f"Subj {i+1}")

    y_true = np.concatenate(all_Y_real)
    y_pred = np.concatenate(all_Y_pred)
    r = np.corrcoef(y_true, y_pred)[0,1]

    lims = [min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("Actual Y", fontsize=14)
    plt.ylabel("Predicted Y", fontsize=14)

    plt.text(lims[0] + 0.65*(lims[1]-lims[0]),
            lims[0] + 0.05*(lims[1]-lims[0]),
            f"r = {r:.3f}",
            fontsize=14,
            ha="left", va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.show()

def draw_sim1_recovery(
    all_Y_real, all_Y_pred, X,
    norm=0, max_factors=8, cmap="tab20",
    width_ratios=[1.2, 1, 1, 1.5], wspace=0.35,
    axis_label_size=12, tick_label_size=10, title_size=12, suptitle_size=14,
    suptitle_text="Simulation 1: Recovery and Component Diagnostics", supersuper=None
):
    "1 +"

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from numpy.random import default_rng

    max_factors = min(max_factors, X.shape[1])
    X = X.astype(float)
    if norm == 1:
        Xz = StandardScaler().fit_transform(X)
    else:
        Xz = X

    all_var = []
    for i in range(Xz.shape[1]):
        all_var.append(np.std(Xz[:, i]))
    all_var = np.array(all_var)
    all_var = np.sort(all_var)[::-1]

    def parallel_analysis(X, n_iter=500, random_state=0):
        rng = default_rng(random_state)
        n, d = X.shape
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        eig_real = (s ** 2) / (n - 1)
        rand_eigs = np.zeros((n_iter, d))
        for i in range(n_iter):
            X_rand = rng.standard_normal(size=X.shape)
            _, s_rand, _ = np.linalg.svd(X_rand, full_matrices=False)
            rand_eigs[i, :] = (s_rand ** 2) / (n - 1)
        eig_rand = rand_eigs.mean(axis=0)
        return eig_real, eig_rand

    eig_real, eig_rand = parallel_analysis(Xz)

    pca = PCA().fit(Xz)
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)

    def velicer_map(X, max_factors=8):
        partials = []
        for k in range(1, max_factors + 1):
            pca = PCA(n_components=k).fit(X)
            X_recon = pca.inverse_transform(pca.transform(X))
            resid = X - X_recon
            resid_corr = np.corrcoef(resid, rowvar=False)
            off_diag = resid_corr - np.diag(np.diag(resid_corr))
            partials.append((off_diag ** 2).mean())
        return partials

    partials = velicer_map(Xz, max_factors=max_factors)

    fig, axes = plt.subplots(
        1, 4, figsize=(11, 2.2),
        gridspec_kw={'width_ratios': width_ratios}
    )
    ax_scatter, ax0, ax1, ax2 = axes

    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(all_Y_real)))

    for i in range(len(all_Y_real)):
        all_Y_real[i] = all_Y_real[i].squeeze()
        all_Y_pred[i] = all_Y_pred[i].squeeze()
        ax_scatter.scatter(
            all_Y_real[i], all_Y_pred[i],
            alpha=0.4, s=20, color=colors[i]
        )

    y_true = np.concatenate(all_Y_real)
    y_pred = np.concatenate(all_Y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1]

    data_min = min(y_true.min(), y_pred.min())
    data_max = max(y_true.max(), y_pred.max())
    pad = 0.05 * (data_max - data_min)
    lims = [data_min - pad, data_max + pad]

    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_aspect('equal', adjustable='box')

    ax_scatter.plot(
        lims, lims,
        color="black",
        linewidth=1,
        linestyle="-",
        zorder=0
    )

    ax_scatter.spines['right'].set_visible(False)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.set_xlabel("Actual Y", fontsize=axis_label_size)
    ax_scatter.set_ylabel("Predicted Y", fontsize=axis_label_size)

    ax_scatter.text(
        lims[0] + 0.55 * (lims[1] - lims[0]),
        lims[0] + 0.05 * (lims[1] - lims[0]),
        f"$r$ = {r:.3f}",
        fontsize=axis_label_size,
        ha="left", va="bottom",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

    ax0.plot(range(1, len(all_var) + 1), all_var, marker="o", color="black")
    ax0.set_xlabel("Latent Dimension", fontsize=axis_label_size)
    ax0.set_ylabel("Variance", fontsize=axis_label_size)
    ax0.set_title("Variance of Latent\nDimensions (Sorted)", fontsize=title_size)

    ax1.plot(range(1, len(eig_real) + 1), eig_real,
             marker="o", color="black", label="Real eigenvalues")
    if norm == 1:
        ax1.plot(range(1, len(eig_rand) + 1), eig_rand,
                 marker="s", color="gray", label="Random mean eigenvalues")
        ax1.axhline(1, color="gray", linestyle="--", linewidth=1)

    ax1.set_xlabel("Principal Component", fontsize=axis_label_size)
    ax1.set_ylabel("Eigenvalue", fontsize=axis_label_size)
    ax1.set_title("PCA Scree Plot\n", fontsize=title_size)

    ax2.plot(range(1, len(explained_var) + 1), explained_var,
             marker="o", color="black", label="Proportion")
    ax2.plot(range(1, len(cum_var) + 1), cum_var,
             marker="o", linestyle="--", color="gray", label="Cumulative")

    ax2.set_xlabel("Principal Component", fontsize=axis_label_size)
    ax2.set_ylabel("Variance explained", fontsize=axis_label_size)
    ax2.set_title("PCA Variance Explained\n(Including Cumulative)", fontsize=title_size)

    for ax in [ax0, ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=tick_label_size)

    ax_scatter.tick_params(axis="both", which="major", labelsize=tick_label_size)

    plt.subplots_adjust(wspace=wspace)

    fig.suptitle(
        suptitle_text,
        fontsize=suptitle_size,
        fontweight="bold",
        y=1.18
    )

    if supersuper is not None:
        fig.text(
            0.05, 1.22, supersuper,
            fontsize=suptitle_size,
            fontweight="bold",
            ha="left", va="top"
        )

    plt.show()

def draw_sim1_recovery_v2(
    all_Y_real, all_Y_pred, X,
    log_var=None,
    norm=0, max_factors=8, cmap="tab20",
    width_ratios=[1.5, 1], wspace=0.55,
    axis_label_size=12, tick_label_size=10, title_size=12, suptitle_size=14,
    suptitle_text="Simulation 1: Recovery and Component Diagnostics", supersuper=None,
):
    """1v2 +
 log_var log log σ = 0.5 * log_var
 muX"""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    X = X.astype(float)

    if norm == 1:
        Xz = StandardScaler().fit_transform(X)
    else:
        Xz = X

    all_var = np.array([np.std(Xz[:, i]) for i in range(Xz.shape[1])])

    fig, axes = plt.subplots(
        1, 2, figsize=(6.5, 2.2),
        gridspec_kw={'width_ratios': width_ratios}
    )
    ax_scatter, ax0 = axes

    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(all_Y_real)))

    for i in range(len(all_Y_real)):
        all_Y_real[i] = all_Y_real[i].squeeze()
        all_Y_pred[i] = all_Y_pred[i].squeeze()
        ax_scatter.scatter(
            all_Y_real[i], all_Y_pred[i],
            alpha=0.4, s=20, color=colors[i]
        )

    y_true = np.concatenate(all_Y_real)
    y_pred = np.concatenate(all_Y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1]

    data_min = min(y_true.min(), y_pred.min())
    data_max = max(y_true.max(), y_pred.max())
    pad = 0.05 * (data_max - data_min)
    lims = [data_min - pad, data_max + pad]

    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_aspect('equal', adjustable='box')
    ax_scatter.plot(lims, lims, color="black", linewidth=1, linestyle="-", zorder=0)
    ax_scatter.spines['right'].set_visible(False)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.set_xlabel("Actual Y", fontsize=axis_label_size)
    ax_scatter.set_ylabel("Predicted Y", fontsize=axis_label_size)
    ax_scatter.text(
        lims[0] + 0.55 * (lims[1] - lims[0]),
        lims[0] + 0.05 * (lims[1] - lims[0]),
        f"$r$ = {r:.3f}",
        fontsize=axis_label_size,
        ha="left", va="bottom",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

    if log_var is not None:
        log_sigma = 0.5 * np.asarray(log_var, dtype=float)
        per_dim = np.mean(log_sigma, axis=0)
        vals_sorted = np.sort(per_dim)[::-1]
        y_label = r"$\log\sigma$"
        panel_title = r"$\log\sigma$ of Latent" + "\nDimensions (Sorted)"
    else:
        vals_sorted = np.sort(all_var)[::-1]
        y_label = "Variance"
        panel_title = "Variance of Latent\nDimensions (Sorted)"

    ax0.plot(range(1, len(vals_sorted) + 1), vals_sorted, marker="o", color="black")
    ax0.set_xlabel("Latent Dimension", fontsize=axis_label_size)
    ax0.set_ylabel(y_label, fontsize=axis_label_size)
    ax0.set_title(panel_title, fontsize=title_size)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.tick_params(axis="both", which="major", labelsize=tick_label_size)

    ax_scatter.tick_params(axis="both", which="major", labelsize=tick_label_size)
    plt.subplots_adjust(wspace=wspace)

    fig.suptitle(
        suptitle_text,
        fontsize=suptitle_size,
        fontweight="bold",
        y=1.18
    )

    if supersuper is not None:
        fig.text(
            0.05, 1.22, supersuper,
            fontsize=suptitle_size,
            fontweight="bold",
            ha="left", va="top"
        )

    plt.show()
