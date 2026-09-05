import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))
from neural_processes import *
from loss_func import *
from tools import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import CCA
import pickle
import pandas as pd
import random
import scipy.stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 31
random.seed(SEED)
np.random.seed(SEED)

data_name = "BB_outer"

pick_min_train  = 15
pick_max_train = 45
pick_min_valid = 20
pick_max_valid = 45

p_valid = 0.15
p_test = 0.1

path_data = f"../BB1/data/bb.csv"

df_data = pd.read_csv(path_data)

path_data = f"../BB1/data/bart.csv"
df_data_outer = pd.read_csv(path_data)
df_data_outer = df_data_outer.fillna(df_data_outer.mean())

df_data["response"] = (df_data["Decision_X"] == 1).astype(int)

all_subj = []
tmp = df_data["partid"].unique()
tmp_outer = df_data_outer["partid"].unique()
for subj in tmp:
    df_subj = df_data[df_data["partid"] == subj]

    if 1:
        if subj in tmp_outer:
            all_subj.append(subj)

all_subj = np.array(all_subj)
n_subj = len(all_subj)

all_subj = all_subj[np.random.permutation(n_subj)]

df_data = df_data[df_data["partid"].isin(all_subj)]
df_data_outer = df_data_outer[df_data_outer["partid"].isin(all_subj)]

col_norm = ['X1', 'X2', 'PX1', 'Z1', 'Z2', 'PZ1']
for col in col_norm:

    df_data[col] = (df_data[col] - df_data[col].mean()) / df_data[col].std()
    print(f"{col}: {np.max(df_data[col])}")

col_norm_outer = ['pumps', 'pumps_sh1', 'pumps_sh2', 'pumps_adj', 'pumps1', 'pumps2', 'pumps3']
for col in col_norm_outer:
    df_data_outer[col] = (df_data_outer[col] - df_data_outer[col].mean()) / df_data_outer[col].std()
    print(f"{col}: {np.max(df_data_outer[col])}")

all_len_trial = []
all_len_trial_outer = []
for subj in all_subj:
    df_subj = df_data[df_data["partid"] == subj]
    all_len_trial.append(len(df_subj))
    df_subj_outer = df_data_outer[df_data_outer["partid"] == subj]
    all_len_trial_outer.append(len(df_subj_outer))

all_len_trial = np.array(all_len_trial)
all_len_trial_outer = np.array(all_len_trial_outer)
print(np.min(all_len_trial), np.max(all_len_trial), np.mean(all_len_trial))
print(np.min(all_len_trial_outer), np.max(all_len_trial_outer), np.mean(all_len_trial_outer))
max_len_trials = np.max(all_len_trial)
max_len_trials_outer = np.max(all_len_trial_outer)

data_list_all = []
for idx_subj in range(n_subj):
    df_subj = df_data[df_data["partid"] == all_subj[idx_subj]]
    tmp = df_subj[['X1', 'X2', 'PX1', 'Z1', 'Z2', 'PZ1', "response"]].values

    tmp = tmp[np.random.permutation(len(tmp))]
    data_list_all.append(tmp)

data_list_all_outer = []
for idx_subj in range(n_subj):
    df_subj_outer = df_data_outer[df_data_outer["partid"] == all_subj[idx_subj]]
    tmp_outer = df_subj_outer.drop(columns=["partid"])
    tmp_outer = tmp_outer.values
    data_list_all_outer.append(tmp_outer)

n_valid_subjs = int(n_subj *  p_valid)
n_test_subjs = int(n_subj *  p_test)
n_train_subjs = n_subj - n_valid_subjs - n_test_subjs

n_trial_inference_train = torch.randint(pick_min_train, pick_max_train + 1, (n_train_subjs,))
n_trial_inference_valid = torch.randint(pick_min_valid, pick_max_valid + 1, (n_valid_subjs,))
n_trial_inference_test = torch.randint(pick_min_valid, pick_max_valid + 1, (n_test_subjs,))

n_trial_test= None

data_dict = {
    "all_subj": all_subj,
    "n_subj": n_subj,
    "all_len_trial": all_len_trial,
    "max_len_trials": max_len_trials,
    "max_len_trials_outer": max_len_trials_outer,
    "data_list_all": data_list_all,
    "data_list_all_outer": data_list_all_outer,
    "n_train_subjs": n_train_subjs,
    "n_valid_subjs": n_valid_subjs,
    "n_test_subjs": n_test_subjs,
    "n_trial_inference_train": n_trial_inference_train,
    "n_trial_inference_valid": n_trial_inference_valid,
    "n_trial_inference_test": n_trial_inference_test,
    "pick_min_train": pick_min_train,
    "pick_max_train": pick_max_train,
    "pick_min_valid": pick_min_valid,
    "pick_max_valid": pick_max_valid,
    "n_trial_test": n_trial_test,
    "outer_dim": 7,
}
dir_save = f"save/s1_data/"
os.makedirs(dir_save, exist_ok=True)
pickle.dump(data_dict, open(f"{dir_save}/{data_name}.pkl", "wb"))
