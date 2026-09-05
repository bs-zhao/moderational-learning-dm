import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
from neural_processes_batch_new import basicNP_D2_outer
from loss_func import beta_bce_loss_batch, info_bce_loss_batch, beta_mse_loss_batch, info_mse_loss_batch
from tools import *

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.widgets import Cursor
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

seed = 2026
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_name = ["RiskC", "DelayC", "RiskP", "DelayP", "BB"][3]

latent_dim = 8

all_beta = [0.8]

all_sparse_lambda = [0.002]

with_outer = 0
an_beta = 0
loss_type = "beta"
nrpeat = 1
alpha = 0
lambda_mmd = 0

batch_size = 64
epochs = 10000
learning_rate = 5e-4
patience = 50
show_plot = 0

if data_name == "BB":
    x_dim = 6
else:
    x_dim = 4
y_dim = 1

if data_name == "DelayP" or data_name == "RiskP":
    x_dim = 2
    is_price = 1
    loss_func_beta = beta_mse_loss_batch
    loss_func_info = info_mse_loss_batch
else:
    is_price = 0
    loss_func_beta = beta_bce_loss_batch
    loss_func_info = info_bce_loss_batch

for beta in all_beta:
    for sparse_lambda in all_sparse_lambda:

        if an_beta != 1:
            dir_save = f"save/s2_train/MLP_mod/{data_name}_{loss_type}_b_{beta}_lat_{latent_dim}_outer_{with_outer}_sp_{sparse_lambda}_rp_{nrpeat}/"
        else:
            dir_save = f"save/s2_train/MLP_mod/{data_name}_{loss_type}_b_{beta}_la_{lambda_mmd}_lat_{latent_dim}_outer_{with_outer}_sp_{sparse_lambda}_rp_{nrpeat}_an_{an_beta}/"

        os.makedirs(dir_save, exist_ok=True)
        os.makedirs(f"{dir_save}/states", exist_ok=True)

        save_name_tmp = f"{dir_save}/tmp.pkl"
        save_name = f"{dir_save}/all.pkl"

        dir_in = f"save/s1_data/"
        with open(f"{dir_in}/{data_name}.pkl", "rb") as f:
            data = pickle.load(f)

        all_subj = data["all_subj"]
        n_subj = data["n_subj"]
        all_len_trial = data["all_len_trial"]
        data_list_all = data["data_list_all"]

        n_train_subjs = data["n_train_subjs"]
        n_valid_subjs = data["n_valid_subjs"]
        n_test_subjs = data["n_test_subjs"]

        pick_min_train = data["pick_min_train"]
        pick_max_train = data["pick_max_train"]

        n_trial_inference_valid = data["n_trial_inference_valid"]
        n_trial_inference_test = data["n_trial_inference_test"]

        max_len_trials = data["max_len_trials"]

        outer_dim = x_dim + y_dim
        max_len_trials_outer = max_len_trials
        data_list_all_outer = data_list_all

        train_dataset = ListDataset(data_list_all[:n_train_subjs], data_list_all_outer[:n_train_subjs], data_list_all[:n_train_subjs])
        val_dataset = ListDataset(data_list_all[n_train_subjs:n_train_subjs+n_valid_subjs], data_list_all[n_train_subjs:n_train_subjs+n_valid_subjs], data_list_all[n_train_subjs:n_train_subjs+n_valid_subjs])
        test_dataset = ListDataset(data_list_all[n_train_subjs+n_valid_subjs:], data_list_all[n_train_subjs+n_valid_subjs:], data_list_all[n_train_subjs+n_valid_subjs:])
        all_data = ListDataset(data_list_all, data_list_all_outer, data_list_all)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_to_lists)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_to_lists)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_to_lists)
        all_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, collate_fn=collate_to_lists)

        model = basicNP_D2_outer(x_dim=x_dim, y_dim=y_dim, outer_dim=outer_dim, emb_size=32, latent_dim=latent_dim, with_outer=with_outer)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_loss_history = []
        train_recon_loss_history = []
        train_kl_loss_history = []
        train_mmd_loss_history = []
        train_sparse_loss_history = []
        val_loss_history = []
        val_recon_loss_history = []
        val_kl_loss_history = []
        val_mmd_loss_history = []
        val_sparse_loss_history = []
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        for epoch in range(epochs):

            if an_beta != 1:
                beta_now = beta
            else:
                if epoch < 100:
                    beta_now = beta * (epoch / 100)
                else:
                    beta_now = beta

            model.train()
            train_loss = 0
            train_recon_loss = 0
            train_kl_loss = 0
            train_mmd_loss = 0
            train_sparse_loss = 0
            for X_Y_batch, X_Y_batch_outer, labels_batch in train_loader:

                B = len(X_Y_batch)

                for rp in range(nrpeat):

                    XY_context_batch = torch.zeros(B, max_len_trials, x_dim + y_dim, device=device, dtype=torch.float32)
                    XY_outer_batch = torch.zeros(B, max_len_trials_outer, outer_dim, device=device, dtype=torch.float32)
                    XY_target_batch = torch.zeros(B, max_len_trials, x_dim + y_dim, device=device, dtype=torch.float32)

                    trial_mask_context = torch.zeros(B, max_len_trials, dtype=torch.bool, device=device)
                    trial_mask_outer = torch.zeros(B, max_len_trials_outer, dtype=torch.bool, device=device)
                    trial_mask_target = torch.zeros(B, max_len_trials, dtype=torch.bool, device=device)

                    for i in range(B):
                        X_Y = torch.tensor(X_Y_batch[i], device=device, dtype=torch.float32)

                        X_Y_outer = torch.tensor(X_Y_batch_outer[i], device=device, dtype=torch.float32)
                        total_trials = int(X_Y.shape[0])

                        ctx_cap = int(total_trials * 1)
                        high = max(2, min(int(pick_max_train), ctx_cap))
                        low = max(1, min(int(pick_min_train), high - 1))
                        if low >= high:
                            low = max(1, high - 1)

                        num_trial_pick = torch.randint(low=low, high=high, size=(1,), device=device).item()
                        pick_indices = torch.randperm(total_trials, device=device)[:num_trial_pick]

                        X_Y_context = X_Y[pick_indices, :]
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

                    Yh, mu, log_var = model(
                        XY_context_batch,
                        XY_target_batch,
                        XY_outer=XY_outer_batch,
                        trial_mask_context=trial_mask_context,
                        trial_mask_outer=trial_mask_outer,
                        trial_mask_target=trial_mask_target,
                    )
                    Y_target_batch = XY_target_batch[:, :, -y_dim:]

                    if loss_type == "beta":
                        loss, components = loss_func_beta(
                            Yh,
                            target=Y_target_batch,
                            mu=mu, log_var=log_var, beta=beta_now,
                            trial_mask=trial_mask_target,
                            return_components=True,
                            lambda_sparse=sparse_lambda
                        )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_recon_loss += components['reconstruction']
                    train_kl_loss += components['kl_divergence']
                    train_sparse_loss += components.get('sparse_loss', 0.0)

            path_save_state = f"{dir_save}/states/epoch_{epoch}.pth"
            torch.save(model.state_dict(), path_save_state)

            train_loss /= (nrpeat*len(train_loader))
            train_recon_loss /= (nrpeat*len(train_loader))
            train_kl_loss /= (nrpeat*len(train_loader))
            train_mmd_loss /= (nrpeat*len(train_loader))
            train_sparse_loss /= (nrpeat*len(train_loader))
            train_loss_history.append(train_loss)
            train_recon_loss_history.append(train_recon_loss)
            train_kl_loss_history.append(train_kl_loss)
            train_mmd_loss_history.append(train_mmd_loss)
            train_sparse_loss_history.append(train_sparse_loss)

            model.eval()
            val_loss = 0
            val_recon_loss = 0
            val_kl_loss = 0
            val_mmd_loss = 0
            val_sparse_loss = 0
            i_in_val = 0
            with torch.no_grad():

                for X_Y_batch, X_Y_batch_outer, labels_batch in val_loader:
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
                        total_trials = int(X_Y.shape[0])

                        num_trial_pick = min(n_trial_inference_valid[i_in_val], total_trials*1)
                        i_in_val += 1
                        pick_indices = torch.arange(int(num_trial_pick), device=device)

                        X_Y_context = X_Y[pick_indices, :]
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

                    if loss_type == "beta":
                        loss, components = loss_func_beta(
                            Yh,
                            Y_target_batch,
                            mu=mu, log_var=log_var, beta=beta_now,
                            trial_mask=trial_mask_target,
                            return_components=True,
                            lambda_sparse=sparse_lambda
                        )

                    val_loss += loss.item()
                    val_recon_loss += components['reconstruction']
                    val_kl_loss += components['kl_divergence']
                    val_sparse_loss += components.get('sparse_loss', 0.0)

            val_loss /= len(val_loader)
            val_recon_loss /= len(val_loader)
            val_kl_loss /= len(val_loader)
            val_mmd_loss /= len(val_loader)
            val_sparse_loss /= len(val_loader)
            val_loss_history.append(val_loss)
            val_recon_loss_history.append(val_recon_loss)
            val_kl_loss_history.append(val_kl_loss)
            val_mmd_loss_history.append(val_mmd_loss)
            val_sparse_loss_history.append(val_sparse_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = model.state_dict()
            else:
                epochs_without_improvement += 1

            print(f"Epoch {epoch+1}/{epochs}, tLoss: {train_loss:.4f}, vLoss: {val_loss:.4f}, N: {epochs_without_improvement}")

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}, lowest validation loss: {best_val_loss}")
                break

        model.load_state_dict(best_model_state)

        model.eval()
        test_loss = 0
        test_recon_loss = 0
        test_kl_loss = 0
        test_mmd_loss = 0
        test_sparse_loss = 0
        test_correct = 0
        test_total = 0

        all_mu_test = []
        all_log_var_test = []
        i_in_test = 0
        with torch.no_grad():

            for X_Y_batch, X_Y_batch_outer, labels_batch in test_loader:
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
                    total_trials = int(X_Y.shape[0])

                    num_trial_pick = min(n_trial_inference_test[i_in_test], total_trials*1)
                    i_in_test += 1
                    pick_indices = torch.arange(int(num_trial_pick), device=device)

                    X_Y_context = X_Y[pick_indices, :]
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
                if loss_type == "beta":
                    loss, components = loss_func_beta(
                    Yh,
                    Y_target_batch,
                    mu=mu, log_var=log_var, beta=beta,
                    trial_mask=trial_mask_target,
                    return_components=True,
                    lambda_sparse=sparse_lambda
                )

                test_recon_loss += components['reconstruction']
                test_kl_loss += components['kl_divergence']
                test_sparse_loss += components.get('sparse_loss', 0.0)
                test_loss += loss.item()

                preds = Yh.squeeze(-1)
                target = Y_target_batch.squeeze(-1)
                if loss_type == "beta":
                    preds_bin = (preds > 0)
                else:
                    preds_bin = (preds > 0.5)
                target_bin = (target > 0.5)

                correct_masked = (preds_bin == target_bin) & trial_mask_target
                test_correct += correct_masked.sum().item()
                test_total += trial_mask_target.sum().item()

                all_mu_test.append(mu)
                all_log_var_test.append(log_var)

        test_loss /= len(test_loader)
        test_recon_loss /= len(test_loader)
        test_kl_loss /= len(test_loader)
        test_mmd_loss /= len(test_loader)
        test_sparse_loss /= len(test_loader)
        test_acc = (test_correct / test_total) if test_total > 0 else 0.0

        print(f"TestLoss: {test_loss:.4f}, TestReconLoss: {test_recon_loss:.4f}, TestKLLoss: {test_kl_loss:.4f}, TestSparseLoss: {test_sparse_loss:.4f}, TestAcc: {test_acc:.4f}")

        all_mu_test = torch.cat(all_mu_test, dim=0).cpu()
        all_log_var_test = torch.cat(all_log_var_test, dim=0).cpu()

        plt.hist(all_mu_test.cpu()[:,0], bins=50, color="skyblue", alpha=0.2, density=True)
        plt.hist(all_log_var_test.cpu()[:,0], bins=50, color="orange", alpha=0.2, density=True)
        plt.show()

        data_to_save = {
            "data_name": data_name,
            "loss_type": loss_type,
            "beta": beta,
            "alpha": alpha,
            "lambda_mmd": lambda_mmd,
            "sparse_lambda": sparse_lambda,
            "latent_dim": latent_dim,
            "with_outer": with_outer,
            "nrpeat": nrpeat,
            "dats_list_all": data_list_all,
            "dats_list_all_outer": data_list_all_outer,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "all_data": all_data,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "all_loader": all_loader,
            "model": model,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer,
            "train_loss_history": train_loss_history,
            "train_recon_loss_history": train_recon_loss_history,
            "train_kl_loss_history": train_kl_loss_history,
            "train_mmd_loss_history": train_mmd_loss_history,
            "train_sparse_loss_history": train_sparse_loss_history,
            "val_loss_history": val_loss_history,
            "val_recon_loss_history": val_recon_loss_history,
            "val_kl_loss_history": val_kl_loss_history,
            "val_mmd_loss_history": val_mmd_loss_history,
            "val_sparse_loss_history": val_sparse_loss_history,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "best_model_state": best_model_state,
            "test_loss": test_loss,
            "test_recon_loss": test_recon_loss,
            "test_kl_loss": test_kl_loss,
            "test_mmd_loss": test_mmd_loss,
            "test_sparse_loss": test_sparse_loss,
            "test_acc": test_acc,
            "all_mu_test": all_mu_test,
            "all_log_var_test": all_log_var_test,
            "max_len_trials": max_len_trials,
            "max_len_trials_outer": max_len_trials_outer,
            "x_dim": x_dim,
            "y_dim": y_dim,
            "outer_dim": outer_dim,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "patience": patience,
            "show_plot": show_plot,
            "n_train_subjs": n_train_subjs,
            "n_valid_subjs": n_valid_subjs,

            "n_trial_inference_valid": n_trial_inference_valid,
            "n_trial_inference_test": n_trial_inference_test,
            "pick_min_train": pick_min_train,
            "pick_max_train": pick_max_train,
        }

        pickle.dump(data_to_save, open(f"{save_name_tmp}", "wb"))
        pickle.dump(data_to_save, open(f"{save_name}", "wb"))
        print(f"Saved to {save_name}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax1.plot(train_loss_history, label="train loss")
        ax1.plot(train_recon_loss_history, label="train recon loss")
        ax1.plot(train_kl_loss_history, label="train kl loss")
        ax1.plot(train_mmd_loss_history, label="train mmd loss")
        ax1.plot(train_sparse_loss_history, label="train sparse loss")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training losses")
        ax1.legend()
        cursor1 = Cursor(ax1, useblit=True, color='red', linewidth=1)

        ax2.plot(val_loss_history, label="val loss")
        ax2.plot(val_recon_loss_history, label="val recon loss")
        ax2.plot(val_kl_loss_history, label="val kl loss")
        ax2.plot(val_mmd_loss_history, label="val mmd loss")
        ax2.plot(val_sparse_loss_history, label="val sparse loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Validation losses")
        ax2.legend()
        cursor2 = Cursor(ax2, useblit=True, color='blue', linewidth=1)

        plt.tight_layout()
        plt.show()
