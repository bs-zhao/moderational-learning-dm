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

data_name = "RiskC"

pick_min_train  = 5
pick_max_train = 35
pick_min_valid = 20
pick_max_valid = 35

p_valid = 0.15
p_test = 0.1

path_data = f"data/s2_filter_trial/riskychoice_filter_trial.csv"
path_subinfo = f"data/s3_subjinfo/riskychoice_filter_trial_subjinfo.csv"

df_data = pd.read_csv(path_data)
df_subinfo = pd.read_csv(path_subinfo)

df_data = df_data[df_data["filter_trial"] == 1].reset_index(drop=True)

df_data["response"] = (df_data["response"] == 1).astype(int)
df_data["response2"] = ((df_data.a_money > df_data.b_money).astype(int) == df_data.response).astype(int)

all_subj = []
tmp = df_subinfo[(df_subinfo["filter_subj"] == 1) & (df_subinfo["n_trials"] >=15)]["subj_num"].unique()
for subj in tmp:
    df_subj = df_data[df_data["subj_num"] == subj]

    if (np.mean(df_subj["response"])>=0.1) and (np.mean(df_subj["response"])<=0.9) and (np.mean(df_subj["response2"])>=0.1) and (np.mean(df_subj["response"])<=0.9):
        all_subj.append(subj)

all_subj = np.array(all_subj)
n_subj = len(all_subj)

all_subj = all_subj[np.random.permutation(n_subj)]

df_data = df_data[df_data["subj_num"].isin(all_subj)]

col_norm = ['a_money', 'a_prob_win', 'b_money', 'b_prob_win']
for col in col_norm:

    df_data[col] = (df_data[col] - df_data[col].mean()) / df_data[col].std()
    print(f"{col}: {np.max(df_data[col])}")

all_len_trial = []
for subj in all_subj:
    df_subj = df_data[df_data["subj_num"] == subj]
    all_len_trial.append(len(df_subj))

all_len_trial = np.array(all_len_trial)
print(np.min(all_len_trial), np.max(all_len_trial), np.mean(all_len_trial))

max_len_trials = np.max(all_len_trial)

data_list_all = []
for idx_subj in range(n_subj):
    df_subj = df_data[df_data["subj_num"] == all_subj[idx_subj]]
    tmp = df_subj[['a_money', 'a_prob_win', 'b_money', 'b_prob_win', "response"]].values

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
    "p_valid": p_valid,
    "p_test": p_test,
}
dir_save = f"save/s1_data/"
os.makedirs(dir_save, exist_ok=True)
pickle.dump(data_dict, open(f"{dir_save}/{data_name}.pkl", "wb"))
