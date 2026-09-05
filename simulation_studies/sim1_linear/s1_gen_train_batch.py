import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
from neural_processes import *
from loss_func import *
from tools import *
from neural_processes_batch_new import basicNP_D2_outer
from loss_func import beta_bce_loss_batch, info_bce_loss_batch, beta_mse_loss_batch, info_mse_loss_batch

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle

np.random.seed(31)
torch.manual_seed(31)
torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_data(n_subjects, min_trials, max_trials, input_dim, data_name, seed=None):
    if seed is not None:
        np.random.seed(seed)

    data = []
    labels = []

    if data_name == "Gamma1":
        for _ in range(n_subjects):
            coefficients = np.random.gamma(shape=20, scale=0.2, size=(input_dim,))
            n_trials = np.random.randint(min_trials, max_trials + 1)
            X = np.random.uniform(-2, 2, size=(n_trials, input_dim))
            noise = np.random.normal(0, 0.1, size=(n_trials,))
            Y = X @ coefficients + noise

            XY = np.concatenate([X, Y[:, None]], axis=1)
            data.append(torch.tensor(XY, dtype=torch.float64))
            labels.append(torch.tensor(coefficients, dtype=torch.float64))

    elif data_name == "Normal1":
        for _ in range(n_subjects):
            coefficients = np.random.normal(4, 1, size=(input_dim,))
            n_trials = np.random.randint(min_trials, max_trials + 1)
            X = np.random.uniform(-2, 2, size=(n_trials, input_dim))
            noise = np.random.normal(0, 1.5, size=(n_trials,))
            Y = X @ coefficients + noise

            XY = np.concatenate([X, Y[:, None]], axis=1)
            data.append(torch.tensor(XY, dtype=torch.float64))
            labels.append(torch.tensor(coefficients, dtype=torch.float64))

    elif data_name == "Flex6":
        for _ in range(n_subjects):
            coefficients = np.random.normal(4, 1, size=(6,))
            coefficients[input_dim:] = 0

            n_trials = np.random.randint(min_trials, max_trials + 1)
            X = np.random.uniform(-2, 2, size=(n_trials, 6))
            noise = np.random.normal(0, 1.5, size=(n_trials,))
            Y = (
                coefficients[0] * (X[:,0]) +
                coefficients[1] * (X[:,1]**2) +
                coefficients[2] * np.exp(X[:,2]) +
                coefficients[3] * np.abs(X[:,3]) +
                coefficients[4] * (X[:,4]**3) +
                coefficients[5] * np.abs(X[:,5]) +
                noise
            )

            XY = np.concatenate([X, Y[:, None]], axis=1)
            data.append(torch.tensor(XY, dtype=torch.float64))
            labels.append(torch.tensor(coefficients, dtype=torch.float64))

    elif data_name == "Flex6n2":
        for _ in range(n_subjects):
            coefficients = np.random.normal(4, 1, size=(6,))
            coefficients[input_dim:] = 0

            n_trials = np.random.randint(min_trials, max_trials + 1)
            X = np.random.uniform(-2, 2, size=(n_trials, 6))
            noise = np.random.normal(0, 3.5, size=(n_trials,))
            Y = (
                coefficients[0] * (X[:,0]) +
                coefficients[1] * (X[:,1]**2) +
                coefficients[2] * np.exp(X[:,2]) +
                coefficients[3] * (X[:,3] - X[:,1]) +
                coefficients[4] * (X[:,4]**3) +
                coefficients[5] * np.abs(X[:,5] - X[:,2])**2 +
                noise
            )

            XY = np.concatenate([X, Y[:, None]], axis=1)
            data.append(torch.tensor(XY, dtype=torch.float64))
            labels.append(torch.tensor(coefficients, dtype=torch.float64))

    elif data_name == "Flex6Lv2" or data_name == "Flex6Lv2b":
        for _ in range(n_subjects):
            coefficients = np.random.normal(4, 1, size=(6,))
            coefficients[input_dim:] = 1

            n_trials = np.random.randint(min_trials, max_trials + 1)
            X = np.random.uniform(-2, 2, size=(n_trials, 6))

            target_R2 = 0.8
            snr = target_R2 / (1 - target_R2)
            var_signal = (4.0/3.0) * float((coefficients**2).sum())
            noise_std = (var_signal / snr) ** 0.5
            noise = np.random.normal(0, noise_std, size=(n_trials,))

            Y = (
                coefficients[0] * (X[:,0]) +
                coefficients[1] * (X[:,1]) +
                coefficients[2] * (X[:,2]) +
                coefficients[3] * (X[:,3]) +
                coefficients[4] * (X[:,4]) +
                coefficients[5] * (X[:,5]) +
                noise
            )

            XY = np.concatenate([X, Y[:, None]], axis=1)
            data.append(torch.tensor(XY, dtype=torch.float64))
            labels.append(torch.tensor(coefficients, dtype=torch.float64))

    elif data_name == "Hete4n2" or data_name == "Hete4bn2":
        for _ in range(n_subjects):
            coefficients = np.random.normal(3, 1, size=(3,))
            n_trials = np.random.randint(min_trials, max_trials + 1)
            X = np.random.uniform(0, 2, size=(n_trials, 4))

            noise = np.random.normal(0, 6, size=(n_trials,))

            cata = np.random.randint(0, 2)
            coefficients[-1] = cata
            if cata == 0:
                Y = coefficients[0]*(X[:,0]*X[:,1]) - coefficients[1]*(X[:,2] + X[:,3])**2 + noise
            elif cata == 1:
                Y = coefficients[0]*(X[:,0]*X[:,2]) - coefficients[1]*(X[:,1] + X[:,3])**2 + noise

            XY = np.concatenate([X, Y[:, None]], axis=1)
            data.append(torch.tensor(XY, dtype=torch.float64))
            labels.append(torch.tensor(coefficients, dtype=torch.float64))

    elif data_name == "Hete4n3" or data_name == "Hete4bn3":
        for _ in range(n_subjects):
            coefficients = np.random.normal(1.5, 0.5, size=(3,))
            n_trials = np.random.randint(min_trials, max_trials + 1)
            X = np.random.uniform(0, 2, size=(n_trials, 4))

            noise = np.random.normal(0, 3, size=(n_trials,))

            cata = np.random.randint(0, 2)
            coefficients[-1] = cata
            if cata == 0:
                Y = coefficients[0]*(X[:,0]*X[:,1]) - coefficients[1]*(X[:,2] + X[:,3])**2 + noise
            elif cata == 1:
                Y = coefficients[0]*(X[:,0]*X[:,2]) - coefficients[1]*(X[:,1] + X[:,3])**2 + noise

            XY = np.concatenate([X, Y[:, None]], axis=1)
            data.append(torch.tensor(XY, dtype=torch.float64))
            labels.append(torch.tensor(coefficients, dtype=torch.float64))

    elif data_name == "Hete4n4" or data_name == "Hete4bn4":
        for _ in range(n_subjects):
            coefficients = np.random.normal(1.5, 0.5, size=(3,))
            n_trials = np.random.randint(min_trials, max_trials + 1)
            X = np.random.uniform(0, 2, size=(n_trials, 4))

            noise = np.random.normal(0, 5, size=(n_trials,))

            cata = np.random.randint(0, 2)
            coefficients[-1] = cata
            if cata == 0:
                Y = coefficients[0]*(X[:,0]*X[:,1]) - coefficients[1]*(X[:,2] + X[:,3])**2 + noise
            elif cata == 1:
                Y = coefficients[0]*(X[:,0]*X[:,2]) - coefficients[1]*(X[:,1] + X[:,3])**2 + noise

            XY = np.concatenate([X, Y[:, None]], axis=1)
            data.append(torch.tensor(XY, dtype=torch.float64))
            labels.append(torch.tensor(coefficients, dtype=torch.float64))

    return data, labels

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
patience = 30
epochs = 1000
learning_rate = 5e-4

show_plot = 0

all_input_dim = [4]

all_beta = [1]

all_latent_dim = [16]

all_n_repeat = [1]

for input_dim in all_input_dim:

    for beta in all_beta:

        for latent_dim in all_latent_dim:

            for repeat in all_n_repeat:

                beta_start = beta
                beta_end = beta
                warmup_epochs = 100

                print(f"Repeat {repeat}")

                all_data, all_labels = generate_data(n_subjects, min_trials, max_trials, input_dim, data_name, seed=31)

                idx1 = int(n_subjects * (1 - p_valid - p_test))
                idx2 = int(n_subjects * (1 - p_test))
                train_data, train_labels = all_data[:idx1], all_labels[:idx1]
                val_data, val_labels = all_data[idx1:idx2], all_labels[idx1:idx2]
                test_data, test_labels = all_data[idx2:], all_labels[idx2:]

                batch_size = 32
                loss_type = "beta"
                max_len_trials = max_trials
                max_len_trials_outer = max_len_trials
                if "Flex6" in data_name:
                    x_dim = 6
                else:
                    x_dim = input_dim
                y_dim = 1
                outer_dim = x_dim + y_dim

                loss_func_beta = beta_mse_loss_batch
                loss_func_info = info_mse_loss_batch

                train_dataset = ListDataset(train_data, train_data, train_labels)
                val_dataset = ListDataset(val_data, val_data, val_labels)
                test_dataset = ListDataset(test_data, test_data, test_labels)
                all_dataset = ListDataset(all_data, all_data, all_labels)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_to_lists)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_to_lists)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_to_lists)
                all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_to_lists)

                num_pick_valid = torch.randint(pick_min, pick_max, (batch_size*len(val_loader),))
                num_pick_test = torch.randint(pick_min, pick_max, (batch_size*len(test_loader),))

                if "Flex6" in data_name:
                    model = basicNP_D2_outer(x_dim=6, y_dim=1, outer_dim=1, emb_size=emb_size, latent_dim=latent_dim, with_outer=False)
                else:
                    model = basicNP_D2_outer(x_dim=input_dim, y_dim=1, outer_dim=1, emb_size=emb_size, latent_dim=latent_dim, with_outer=False)
                model = model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                train_loss_history = []
                train_rec_loss_history = []
                train_kl_loss_history = []
                val_loss_history = []
                val_rec_loss_history = []
                val_kl_loss_history = []
                best_val_loss = float('inf')
                epochs_without_improvement = 0
                for epoch in range(epochs):

                    if epoch < warmup_epochs:
                        beta = beta_start + (beta_end - beta_start) * epoch / warmup_epochs
                    else:
                        beta = beta_end

                    model.train()
                    train_loss = 0
                    train_rec_loss = 0
                    train_kl_loss = 0
                    train_mmd_loss = 0
                    for X_Y_batch, X_Y_batch_outer, labels_batch in train_loader:

                        B = len(X_Y_batch)

                        for rp in range(1):

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

                                ctx_cap = int(total_trials * 1)
                                high = max(2, min(int(pick_max), ctx_cap))
                                low = max(1, min(int(pick_min), high - 1))
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
                                    mu=mu, log_var=log_var, beta=beta,
                                    trial_mask=trial_mask_target,
                                    return_components=True
                                )
                            elif loss_type == "info":
                                pass

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()
                            train_rec_loss += components['reconstruction']
                            train_kl_loss += components['kl_divergence']

                    train_loss /= len(train_loader)
                    train_rec_loss /= len(train_loader)
                    train_kl_loss /= len(train_loader)
                    train_loss_history.append(train_loss)
                    train_rec_loss_history.append(train_rec_loss)
                    train_kl_loss_history.append(train_kl_loss)

                    model.eval()
                    val_loss = 0
                    val_rec_loss = 0
                    val_kl_loss = 0
                    with torch.no_grad():

                        i_in_val = 0
                        for X_Y_batch, X_Y_batch_outer, labels_batch in val_loader:
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

                                num_trial_pick = min(num_pick_valid[i_in_val], total_trials*1)
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
                                    mu=mu, log_var=log_var, beta=beta,
                                    trial_mask=trial_mask_target,
                                    return_components=True
                                )
                            elif loss_type == "info":
                                pass

                            val_loss += loss.item()
                            val_rec_loss += components['reconstruction']
                            val_kl_loss += components['kl_divergence']

                    val_loss /= len(val_loader)
                    val_rec_loss /= len(val_loader)
                    val_kl_loss /= len(val_loader)
                    val_loss_history.append(val_loss)
                    val_rec_loss_history.append(val_rec_loss)
                    val_kl_loss_history.append(val_kl_loss)
                    if show_plot:
                        plt.show()

                    print(f"Epoch {epoch+1}/{epochs}, tLoss: {train_loss:.4f}, vLoss: {val_loss:.4f}, vRecLoss: {val_rec_loss:.4f}, vKL: {val_kl_loss:.4f}, N: {epochs_without_improvement}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        best_model_state = model.state_dict()
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= patience:
                        print(f"Early stopping at epoch {epoch+1}, lowest validation loss: {best_val_loss}")
                        break

                model.load_state_dict(best_model_state)

                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.plot(train_loss_history, label='Train Loss')
                plt.plot(train_rec_loss_history, label='Train Reconstruction Loss')
                plt.plot(train_kl_loss_history, label='Train KL Divergence Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.title('Training Loss')

                plt.plot(val_loss_history, label='Validation Loss')
                plt.plot(val_rec_loss_history, label='Validation Reconstruction Loss')
                plt.plot(val_kl_loss_history, label='Validation KL Divergence Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.title('Validation Loss')

                plt.show()

                actual_coefficients = []
                predicted_coefficients = []

                model.eval()
                test_loss = 0
                test_rec_loss = 0
                test_kl_loss = 0
                test_mmd_loss = 0

                all_mu_test = []
                all_log_var_test = []
                i_in_test = 0
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

                            num_trial_pick = min(num_pick_test[i_in_test], total_trials*1)
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
                            return_components=True
                        )
                        elif loss_type == "info":
                            pass

                        test_rec_loss += components['reconstruction']
                        test_kl_loss += components['kl_divergence']
                        test_loss += loss.item()

                        all_mu_test.append(mu)
                        all_log_var_test.append(log_var)

                        actual_coefficients += labels_batch
                        predicted_coefficients += mu.tolist()

                test_loss /= len(test_loader)
                test_rec_loss /= len(test_loader)
                test_kl_loss /= len(test_loader)
                print(f"Test Loss: {test_loss:.4f}, Test RecLoss: {test_rec_loss:.4f}, Test KL: {test_kl_loss:.4f}")

                actual_coefficients = np.vstack(labels_batch)
                predicted_coefficients = np.vstack(predicted_coefficients)

                name_save = f"save/train_batch/{data_name}_latent_{latent_dim}_emb_{emb_size}_l1_{l1_lambda}_l2_{l2_lambda}_beta_{beta_end}_nsubj_{n_subjects}_min_{min_trials}_max_{max_trials}_input_{input_dim}_minpick_{pick_min}_maxpick_{pick_max}_repeat_{repeat}.pkl"

                dic_save = {}
                dic_save['model'] = model
                dic_save['train_loss_history'] = train_loss_history
                dic_save['val_loss_history'] = val_loss_history
                dic_save['train_rec_loss_history'] = train_rec_loss_history
                dic_save['train_kl_loss_history'] = train_kl_loss_history
                dic_save['val_rec_loss_history'] = val_rec_loss_history
                dic_save['val_kl_loss_history'] = val_kl_loss_history
                dic_save['actual_coefficients'] = actual_coefficients
                dic_save['predicted_coefficients'] = predicted_coefficients
                dic_save['test_loss'] = test_loss
                dic_save['test_rec_loss'] = test_rec_loss
                dic_save['test_kl_loss'] = test_kl_loss
                dic_save['num_pick_valid'] = num_pick_valid
                dic_save['num_pick_test'] = num_pick_test
                dic_save['model_state_dict'] = model.state_dict()
                dic_save['l1_lambda'] = l1_lambda
                dic_save['l2_lambda'] = l2_lambda
                dic_save['beta'] = beta
                dic_save['latent_dim'] = latent_dim
                dic_save['emb_size'] = emb_size
                dic_save['n_subjects'] = n_subjects
                dic_save['min_trials'] = min_trials
                dic_save['max_trials'] = max_trials
                dic_save['input_dim'] = input_dim
                dic_save['repeat'] = repeat
                dic_save['all_data'] = all_data
                dic_save['all_labels'] = all_labels
                dic_save['train_data'] = train_data
                dic_save['train_labels'] = train_labels
                dic_save['val_data'] = val_data
                dic_save['val_labels'] = val_labels
                dic_save['test_data'] = test_data
                dic_save['test_labels'] = test_labels
                dic_save['train_dataset'] = train_dataset
                dic_save['val_dataset'] = val_dataset
                dic_save['test_dataset'] = test_dataset
                dic_save['all_dataset'] = all_dataset
                dic_save['train_loader'] = train_loader
                dic_save['val_loader'] = val_loader
                dic_save['test_loader'] = test_loader
                dic_save['all_loader'] = all_loader

                os.makedirs(os.path.dirname(name_save), exist_ok=True)
                pickle.dump(dic_save, open(name_save, 'wb'))

                print(name_save)
