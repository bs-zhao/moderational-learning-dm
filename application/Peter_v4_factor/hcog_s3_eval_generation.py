"""hcog_s3_eval_generation.py

 df_test_generation hcog_s2 test posterior posterior_means.csv
 RiskC_context / RiskC_test choice accuracy
 fold s7_cross_test_mod_v3.py test_acc

 - hcog_s1_getdata.py save/hcog/getdata/<TASK>/f_<k>/df_test_generation.csv
 - hcog_s2_fit.r test fold save/hcog/fit_test/<TASK>/f_<k>/posterior_means.csv

 real_data/Peter_v4_factor
 python hcog_s3_eval_generation.py"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

TASK = "RiskC"
FOLD_MIN = 0
FOLD_MAX = 9

def task_is_price(name: str) -> bool:
    return name.endswith("P")

def weight_prob(p: np.ndarray, gamma: float, tau: float) -> np.ndarray:
    "Stan RiskC_context.stan weight_prob"
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    log_num = tau + gamma * np.log(p)
    log_denom = np.logaddexp(tau + gamma * np.log(p), gamma * np.log1p(-p))
    return np.exp(log_num - log_denom)

def prob_choose_b(
    a_amt: np.ndarray,
    a_prob: np.ndarray,
    b_amt: np.ndarray,
    b_prob: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    tau: float,
) -> np.ndarray:
    w_a = weight_prob(a_prob, gamma, tau)
    w_b = weight_prob(b_prob, gamma, tau)
    v_a = np.power(a_amt, alpha) * w_a
    v_b = np.power(b_amt, alpha) * w_b
    logit = beta * (v_b - v_a)
    return 1.0 / (1.0 + np.exp(-logit))

def eval_one_fold(task: str, fold: int, base_dir: str) -> Optional[Dict[str, Any]]:
    if task_is_price(task):
        raise NotImplementedError(
            "DelayP/RiskP Stan posterior RiskC choice accuracy"
        )

    path_gen = os.path.join(
        base_dir, "save", "hcog", "getdata", task, f"f_{fold}", "df_test_generation.csv"
    )
    path_pm = os.path.join(
        base_dir, "save", "hcog", "fit_test", task, f"f_{fold}", "posterior_means.csv"
    )
    if not os.path.isfile(path_gen):
        print(f"[ fold {fold}] : {path_gen}")
        return None
    if not os.path.isfile(path_pm):
        print(f"[ fold {fold}] : {path_pm} fold  hcog_s2_fit.r test")
        return None

    df_g = pd.read_csv(path_gen)
    df_p = pd.read_csv(path_pm)

    need_g = ["subj", "choice", "a_amount", "a_prob", "b_amount", "b_prob"]
    if not all(c in df_g.columns for c in need_g):
        raise ValueError(f"{path_gen} : {need_g}")

    need_p = ["subj", "alpha_mean", "beta_mean", "gamma_mean", "tau_mean"]
    if not all(c in df_p.columns for c in need_p):
        raise ValueError(f"{path_pm} : {need_p}")

    pm_map = df_p.set_index("subj")[["alpha_mean", "beta_mean", "gamma_mean", "tau_mean"]]

    preds = []
    targets = []
    for subj, g_sub in df_g.groupby("subj"):
        if subj not in pm_map.index:
            print(f"   fold {fold}:  {subj}  posterior ")
            continue
        row = pm_map.loc[subj]
        alpha = float(row["alpha_mean"])
        beta = float(row["beta_mean"])
        gamma = float(row["gamma_mean"])
        tau = float(row["tau_mean"])

        a_amt = g_sub["a_amount"].to_numpy(dtype=np.float64)
        a_pb = g_sub["a_prob"].to_numpy(dtype=np.float64)
        b_amt = g_sub["b_amount"].to_numpy(dtype=np.float64)
        b_pb = g_sub["b_prob"].to_numpy(dtype=np.float64)
        y = g_sub["choice"].to_numpy(dtype=np.float64)

        p_b = prob_choose_b(a_amt, a_pb, b_amt, b_pb, alpha, beta, gamma, tau)
        pred_c = (p_b > 0.5).astype(np.float64)
        preds.append(pred_c)
        targets.append(y)

    if not preds:
        print(f"[fold {fold}] ")
        return None

    pred_all = np.concatenate(preds)
    tgt_all = np.concatenate(targets)
    acc = float(np.mean(pred_all == tgt_all))
    return {"fold": fold, "n_trials": int(pred_all.size), "accuracy": acc, "mse": np.nan}

def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for fold in range(FOLD_MIN, FOLD_MAX + 1):
        r = eval_one_fold(TASK, fold, base_dir)
        if r is not None:
            rows.append(r)
            if task_is_price(TASK):
                print(
                    f"fold {fold}: MSE={r['mse']:.6f} | trials={r['n_trials']}"
                )
            else:
                print(
                    f"fold {fold}: accuracy={r['accuracy']:.6f} | trials={r['n_trials']}"
                )

    if not rows:
        print("fold")
        sys.exit(1)

    out_dir = os.path.join(base_dir, "save", "hcog", "eval", TASK)
    os.makedirs(out_dir, exist_ok=True)
    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "generation_metrics_by_fold.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\n fold : {out_csv}")

    if task_is_price(TASK):
        mean_mse = float(df_out["mse"].mean())
        print(f"\n=== {TASK}  fold  MSEdf_test_generation: {mean_mse:.6f} ===")
        print("s7_cross_test_mod_v3")
    else:
        mean_acc = float(df_out["accuracy"].mean())
        std_acc = float(df_out["accuracy"].std(ddof=0))
        print(
            f"\n=== {TASK}  fold  accuracydf_test_generation: "
            f"{mean_acc:.6f} (sd across folds={std_acc:.6f}) ==="
        )
        print("s7_cross_test_mod_v3.py TestAcc")

if __name__ == "__main__":
    main()
