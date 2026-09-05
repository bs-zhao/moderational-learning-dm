import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
from simple_mlps import simple_mlp1
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
import torch.nn.functional as F
import copy
from scipy.optimize import minimize
import pandas as pd

device = 'cpu'

seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

for data_name in ["RiskC"]:

    for fold in range(10):

        n_trial_test = 5

        batch_size = 1
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
            loss_func_beta = F.mse_loss
            loss_func_info = F.mse_loss
        else:
            is_price = 0
            loss_func_beta = F.binary_cross_entropy
            loss_func_info = F.binary_cross_entropy

        dir_save = f"save/hcog/getdata/{data_name}/f_{fold}/"

        os.makedirs(dir_save, exist_ok=True)

        save_name_tmp = f"{dir_save}/tmp.pkl"
        save_name = f"{dir_save}/all.pkl"

        dir_in = f"../Peter_v3/save/s1_data/"
        with open(f"{dir_in}/{data_name}_nonNorm.pkl", "rb") as f:
            data = pickle.load(f)

        all_subj = data["all_subj"]
        n_subj = data["n_subj"]
        all_len_trial = data["all_len_trial"]
        data_list_all = data["data_list_all"]

        n_train_subjs = data["n_train_subjs"]
        n_valid_subjs = data["n_valid_subjs"]
        n_test_subjs = data["n_test_subjs"]
        n_trial_inference_train = data["n_trial_inference_train"]
        n_trial_inference_valid = data["n_trial_inference_valid"]
        n_trial_inference_test = data["n_trial_inference_test"]
        pick_min_train = data["pick_min_train"]
        pick_max_train = data["pick_max_train"]
        pick_min_valid = data["pick_min_valid"]
        pick_max_valid = data["pick_max_valid"]
        max_len_trials = data["max_len_trials"]

        outer_dim = x_dim + y_dim
        max_len_trials_outer = max_len_trials
        data_list_all_outer = data_list_all

        n_trial_inference_test = n_trial_inference_test.repeat(5)
        n_trial_inference_valid = n_trial_inference_valid.repeat(5)
        n_trial_inference_train = n_trial_inference_train.repeat(5)

        n_test_subjs = int(0.1*n_subj)
        n_valid_subjs = int(0.1*n_subj)
        n_train_subjs = n_subj - n_test_subjs - n_valid_subjs

        data_list_test = data_list_all[fold*n_test_subjs:(fold+1)*n_test_subjs]
        data_list_rest = data_list_all[:fold*n_test_subjs] + data_list_all[(fold+1)*n_test_subjs:]

        data_list_train = data_list_rest[:n_train_subjs]
        data_list_val = data_list_rest[n_train_subjs:n_train_subjs+n_valid_subjs]

        train_dataset = ListDataset(data_list_train, data_list_train, data_list_train)
        val_dataset = ListDataset(data_list_val, data_list_val, data_list_val)
        test_dataset = ListDataset(data_list_test, data_list_test, data_list_test)
        all_data = ListDataset(data_list_all, data_list_all, data_list_all)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_to_lists)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_to_lists)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_to_lists)
        all_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, collate_fn=collate_to_lists)

        test_loss = 0
        test_correct = 0
        test_total = 0

        all_mu_test = []
        all_log_var_test = []
        i_in_test = 0

        all_Y_target_test = []
        all_Yh_test = []

        all_test_loss = []

        subj_now = 0
        df_train = pd.DataFrame()
        df_test_inference = pd.DataFrame()
        df_test_generation = pd.DataFrame()

        for X_Y_batch, X_Y_batch_outer, labels_batch in train_loader:
            B = len(X_Y_batch)

            loss_batch = 0
            for i in range(B):
                subj_now += 1

                X_Y = torch.tensor(X_Y_batch[i], device=device, dtype=torch.float32)
                n_trial = X_Y.shape[0]

                arr_subj_now = np.full(n_trial, subj_now)
                arr_trial_now = np.arange(n_trial)
                arr_choice = X_Y[:, -1].detach().cpu().numpy()
                arr_a_amount = X_Y[:, 0].detach().cpu().numpy()
                arr_a_prob = X_Y[:, 1].detach().cpu().numpy()
                arr_b_amount = X_Y[:, 2].detach().cpu().numpy()
                arr_b_prob = X_Y[:, 3].detach().cpu().numpy()

                df_tmp = pd.DataFrame({
                    "subj": arr_subj_now,
                    "trial": arr_trial_now,
                    "choice": arr_choice,
                    "a_amount": arr_a_amount,
                    "a_prob": arr_a_prob,
                    "b_amount": arr_b_amount,
                    "b_prob": arr_b_prob
                })

                df_train = pd.concat([df_train, df_tmp], ignore_index=True)

        for X_Y_batch, X_Y_batch_outer, labels_batch in test_loader:
            B = len(X_Y_batch)

            for i in range(B):

                subj_now += 1

                X_Y = torch.tensor(X_Y_batch[i], device=device, dtype=torch.float32)

                X_train_valid = X_Y[n_trial_test:, :x_dim]
                Y_train_valid = X_Y[n_trial_test:, -y_dim:]
                X_test = X_Y[:n_trial_test, :x_dim]
                Y_test = X_Y[:n_trial_test, -y_dim:]

                n_trial_valid = int(0.3*X_train_valid.shape[0])
                X_valid = X_train_valid[:n_trial_valid, :]
                Y_valid = Y_train_valid[:n_trial_valid, -y_dim:]
                X_train = X_train_valid[n_trial_valid:, :]
                Y_train = Y_train_valid[n_trial_valid:, -y_dim:]

                XY_train_valid = torch.cat([X_train_valid, Y_train_valid], dim=1)
                XY_train_valid_np = XY_train_valid.detach().cpu().numpy()

                XY_test = torch.cat([X_test, Y_test], dim=1)
                XY_test_np = XY_test.detach().cpu().numpy()

                n_trial_train = XY_train_valid_np.shape[0]
                arr_subj_now = np.full(n_trial_train, subj_now)
                arr_trial_now = np.arange(n_trial_train)
                arr_choice = XY_train_valid_np[:, -1]
                arr_a_amount = XY_train_valid_np[:, 0]
                arr_a_prob = XY_train_valid_np[:, 1]
                arr_b_amount = XY_train_valid_np[:, 2]
                arr_b_prob = XY_train_valid_np[:, 3]

                df_tmp = pd.DataFrame({
                    "subj": arr_subj_now,
                    "trial": arr_trial_now,
                    "choice": arr_choice,
                    "a_amount": arr_a_amount,
                    "a_prob": arr_a_prob,
                    "b_amount": arr_b_amount,
                    "b_prob": arr_b_prob
                })

                df_test_inference = pd.concat([df_test_inference, df_tmp], ignore_index=True)

                n_trial_test = XY_test_np.shape[0]
                arr_subj_now = np.full(n_trial_test, subj_now)
                arr_trial_now = np.arange(n_trial_test)
                arr_choice = XY_test_np[:, -1]
                arr_a_amount = XY_test_np[:, 0]
                arr_a_prob = XY_test_np[:, 1]
                arr_b_amount = XY_test_np[:, 2]
                arr_b_prob = XY_test_np[:, 3]

                df_tmp = pd.DataFrame({
                    "subj": arr_subj_now,
                    "trial": arr_trial_now,
                    "choice": arr_choice,
                    "a_amount": arr_a_amount,
                    "a_prob": arr_a_prob,
                    "b_amount": arr_b_amount,
                    "b_prob": arr_b_prob
                })

                df_test_generation = pd.concat([df_test_generation, df_tmp], ignore_index=True)

        df_train.to_csv(f"{dir_save}/df_train.csv", index=False)
        df_test_inference.to_csv(f"{dir_save}/df_test_inference.csv", index=False)
        df_test_generation.to_csv(f"{dir_save}/df_test_generation.csv", index=False)

        print(f"Saved df_train.csv and df_test.csv to {dir_save}")

        print(f"Saved df_test_inference.csv and df_test_generation.csv to {dir_save}")

        print(f"Saved df_test_generation.csv to {dir_save}")
