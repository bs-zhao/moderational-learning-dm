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

data_name = "BB"

pick_min_train  = 15
pick_max_train = 45
pick_min_valid = 20
pick_max_valid = 45

p_valid = 0.15
p_test = 0.1

path_data = f"../BB1/data/bb.csv"

df_data = pd.read_csv(path_data)

df_data["response"] = (df_data["Decision_X"] == 1).astype(int)

all_subj = []
tmp = df_data["partid"].unique()
for subj in tmp:
    df_subj = df_data[df_data["partid"] == subj]

    if 1:
        all_subj.append(subj)

all_subj = np.array(all_subj)
n_subj = len(all_subj)

all_subj = all_subj[np.random.permutation(n_subj)]

df_data = df_data[df_data["partid"].isin(all_subj)]

col_norm = ['X1', 'X2', 'PX1', 'Z1', 'Z2', 'PZ1']

all_len_trial = []
for subj in all_subj:
    df_subj = df_data[df_data["partid"] == subj]
    all_len_trial.append(len(df_subj))

all_len_trial = np.array(all_len_trial)
print(np.min(all_len_trial), np.max(all_len_trial), np.mean(all_len_trial))

max_len_trials = np.max(all_len_trial)

data_list_all = []
for idx_subj in range(n_subj):
    df_subj = df_data[df_data["partid"] == all_subj[idx_subj]]
    tmp = df_subj[['X1', 'X2', 'PX1', 'Z1', 'Z2', 'PZ1', "response"]].values

    tmp = tmp[np.random.permutation(len(tmp))]
    data_list_all.append(tmp)

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
    "data_list_all": data_list_all,
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
}
dir_save = f"save/s1_data/"
os.makedirs(dir_save, exist_ok=True)
pickle.dump(data_dict, open(f"{dir_save}/{data_name}_nonNorm.pkl", "wb"))

print(f"num subj: {len(all_subj)}")
