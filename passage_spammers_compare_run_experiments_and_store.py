#!/usr/bin/env python3
"""
Run multiple ranking algorithms on a FaceAge pairwise dataset and store results.

This script loads a pre-computed pairwise-comparisons pickle (PC_faceage), runs
several ranking methods, computes metrics (accuracy, weighted-accuracy, Kendall's
Tau) and stores:
  - a CSV with summary metrics per method
  - a pickle with detailed score vectors per method
  - bar plots (png) summarizing the metrics

Usage example:
  python run_experiments_and_store.py --spammer_type random --spammer_percent 10

The script is intentionally defensive: if any method raises an exception it will
record NaNs for that method but continue running the others.
"""

import argparse
import os
import time
import pickle
import json
import math
from collections import OrderedDict

import numpy as np
import torch
from scipy import stats

# External libraries / local modules used in your environment
import choix
import opt_fair
from pgem import EMWrapper
from opt_fair import RankCentrality, FactorBT, CrowdBT, BARP
from metrics import compute_acc, compute_weighted_acc
from crowdkit.aggregation import NoisyBradleyTerry

def to_numpy(x):
    """Convert tensors or arrays to 1D numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().ravel()
    if isinstance(x, np.ndarray):
        return x.ravel()
    return np.asarray(x).ravel()


def safe_kendall(x, y):
    try:
        r = stats.kendalltau(x, y)
        return float(r.correlation) if r is not None else float('nan')
    except Exception:
        return float('nan')


def run_all_methods(pc_path, df_faceage_path, device, out_dir, df_path, max_iter_pgem=500):
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    with open(pc_path, 'rb') as f:
        PC_faceage = pickle.load(f)
    with open(df_faceage_path, 'rb') as f:
        df_faceage = pickle.load(f)
    import pandas
    df = pandas.read_csv(df_path)

    size = len(df_faceage)
    classes = df_faceage['gender'] if 'gender' in df_faceage else None

    results = []  # list of dicts for CSV
    detailed_scores = {}  # store raw score vectors per method

    # Prepare a version of all pairwise comparisons (no reviewer dimension) if needed
    try:
        all_pc_faceage = opt_fair._pc_without_reviewers(PC_faceage)
    except Exception:
        # fallback: if the function isn't available, attempt to aggregate manually
        all_pairs = []
        for reviewer, pairs in PC_faceage.items():
            all_pairs.extend(pairs)
        all_pc_faceage = all_pairs
    
    
    # 5) FactorBT
    method = 'FactorBT'
    t0 = time.perf_counter()
    try:
#         factorbt_test = FactorBT(data=PC_faceage, penalty=0, classes=classes, device=device)
#         factorbt_scores, y, z = factorbt_test.alternate_optim(iters=100)
#         %%time
        agg_noisybt = NoisyBradleyTerry(n_iter=10).fit_predict(df)
        gt_df = df_faceage
        
        def sort_df(df, column_name):
            # Sort by a specific column (replace 'column_name' with your column)
            df_sorted = df.sort_values(by=column_name, ascending=True)  # or ascending=False

            return df_sorted
        gt_df = sort_df(gt_df, 'label')
        agg_noisybt_df = pandas.DataFrame(list(agg_noisybt.items()), columns=['label', 'score'])
        agg_noisybt_df = sort_df(agg_noisybt_df, 'label')
        factorbt_scores = list(agg_noisybt_df['score'])
        factorbt_scores = to_numpy(factorbt_scores)
        kt = safe_kendall(factorbt_scores, df_faceage['score'])
        acc = compute_acc(df_faceage, factorbt_scores, device)
        wacc = compute_weighted_acc(df_faceage, factorbt_scores, device)
        runtime = time.perf_counter() - t0
        results.append(OrderedDict(method=method, acc=float(acc), wacc=float(wacc), kendall_tau=float(kt), runtime_s=float(runtime)))
        detailed_scores[method] = factorbt_scores
    except Exception as e:
        runtime = time.perf_counter() - t0
        results.append(OrderedDict(method=method, acc=math.nan, wacc=math.nan, kendall_tau=math.nan, runtime_s=float(runtime)))
        detailed_scores[method] = None
        print(f"{method} failed: {e}")
    
    # 1) PGEM (EMWrapper)
    method = 'PGEM'
    t0 = time.perf_counter()
    try:
        pg = EMWrapper(PC_faceage, max_iter_pgem, device)
        r_est, beta_est, ll = pg.run_algorithm()
        r_est = to_numpy(r_est)
        acc = compute_acc(df_faceage, r_est, device)
        if acc < 0.5:
            r_est = -1.0 * r_est
            acc = compute_acc(df_faceage, r_est, device)
        kt = safe_kendall(r_est, df_faceage['score'])
        wacc = compute_weighted_acc(df_faceage, r_est, device)
        runtime = time.perf_counter() - t0
        results.append(OrderedDict(method=method, acc=float(acc), wacc=float(wacc), kendall_tau=float(kt), runtime_s=float(runtime)))
        detailed_scores[method] = r_est
    except Exception as e:
        runtime = time.perf_counter() - t0
        results.append(OrderedDict(method=method, acc=math.nan, wacc=math.nan, kendall_tau=math.nan, runtime_s=float(runtime)))
        detailed_scores[method] = None
        print(f"{method} failed: {e}")

    # 2) Bradley-Terry (choix.opt_pairwise)
    method = 'BT'
    t0 = time.perf_counter()
    try:
        bt_scores = choix.opt_pairwise(size, all_pc_faceage, alpha=0, method='Newton-CG', initial_params=None, max_iter=None, tol=1e-05)
        bt_scores = to_numpy(bt_scores)
        kt = safe_kendall(bt_scores, df_faceage['score'])
        acc = compute_acc(df_faceage, bt_scores, device)
        wacc = compute_weighted_acc(df_faceage, bt_scores, device)
        runtime = time.perf_counter() - t0
        results.append(OrderedDict(method=method, acc=float(acc), wacc=float(wacc), kendall_tau=float(kt), runtime_s=float(runtime)))
        detailed_scores[method] = bt_scores
    except Exception as e:
        runtime = time.perf_counter() - t0
        results.append(OrderedDict(method=method, acc=math.nan, wacc=math.nan, kendall_tau=math.nan, runtime_s=float(runtime)))
        detailed_scores[method] = None
        print(f"{method} failed: {e}")

    # 3) BARP (opt_fair.BARP + alternate optim)
    method = 'BARP'
    t0 = time.perf_counter()
    try:
        size = len(df_faceage)
        classes = [0]*size
        face_obj = BARP(data=PC_faceage, penalty=0, classes=classes, device=device)
        num_reviewers = len(PC_faceage)
        annot_bt_temp, annot_bias = opt_fair._alternate_optim_torch(size, num_reviewers, face_obj, iters=100)
        annot_bt_temp = to_numpy(annot_bt_temp)
        kt = safe_kendall(annot_bt_temp, df_faceage['score'])
        acc = compute_acc(df_faceage, annot_bt_temp, device)
        wacc = compute_weighted_acc(df_faceage, annot_bt_temp, device)
        runtime = time.perf_counter() - t0
        results.append(OrderedDict(method=method, acc=float(acc), wacc=float(wacc), kendall_tau=float(kt), runtime_s=float(runtime)))
        detailed_scores[method] = annot_bt_temp
    except Exception as e:
        runtime = time.perf_counter() - t0
        results.append(OrderedDict(method=method, acc=math.nan, wacc=math.nan, kendall_tau=math.nan, runtime_s=float(runtime)))
        detailed_scores[method] = None
        print(f"{method} failed: {e}")

    # 4) Rank Centrality (opt_fair.RankCentrality)
    method = 'RC'
    t0 = time.perf_counter()
    try:
        rc_obj = RankCentrality(device)
        A = rc_obj.matrix_of_comparisons(size, all_pc_faceage)
        P = rc_obj.trans_prob(A)
        pi = rc_obj.stationary_dist(P)
        rank_centrality_scores = np.log(to_numpy(pi))
        kt = safe_kendall(rank_centrality_scores, df_faceage['score'])
        acc = compute_acc(df_faceage, rank_centrality_scores, device)
        wacc = compute_weighted_acc(df_faceage, rank_centrality_scores, device)
        runtime = time.perf_counter() - t0
        results.append(OrderedDict(method=method, acc=float(acc), wacc=float(wacc), kendall_tau=float(kt), runtime_s=float(runtime)))
        detailed_scores[method] = rank_centrality_scores
    except Exception as e:
        runtime = time.perf_counter() - t0
        results.append(OrderedDict(method=method, acc=math.nan, wacc=math.nan, kendall_tau=math.nan, runtime_s=float(runtime)))
        detailed_scores[method] = None
        print(f"{method} failed: {e}")

    

    # 6) CrowdBT
    method = 'CrowdBT'
    t0 = time.perf_counter()
    try:
        crowdbt_test = CrowdBT(data=PC_faceage, penalty=0, device=device)
        crowdbt_scores, y = crowdbt_test.alternate_optim(size, len(PC_faceage))
        crowdbt_scores = to_numpy(crowdbt_scores)
        kt = safe_kendall(crowdbt_scores, df_faceage['score'])
        acc = compute_acc(df_faceage, crowdbt_scores, device)
        wacc = compute_weighted_acc(df_faceage, crowdbt_scores, device)
        runtime = time.perf_counter() - t0
        results.append(OrderedDict(method=method, acc=float(acc), wacc=float(wacc), kendall_tau=float(kt), runtime_s=float(runtime)))
        detailed_scores[method] = crowdbt_scores
    except Exception as e:
        runtime = time.perf_counter() - t0
        results.append(OrderedDict(method=method, acc=math.nan, wacc=math.nan, kendall_tau=math.nan, runtime_s=float(runtime)))
        detailed_scores[method] = None
        print(f"{method} failed: {e}")

    # Save summary CSV
    import pandas as pd
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, 'metrics_summary.csv')
    df_results.to_csv(csv_path, index=False)

    # Save detailed scores
    scores_path = os.path.join(out_dir, 'detailed_scores.pickle')
    with open(scores_path, 'wb') as f:
        pickle.dump(detailed_scores, f)

    # Save metadata
    meta = dict(size=size, num_reviewers=len(PC_faceage), methods=list(detailed_scores.keys()))
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

#     # Make a couple of bar plots: acc, wacc, kendall
#     try:
#         import matplotlib.pyplot as plt
#         metrics_to_plot = ['acc', 'wacc', 'kendall_tau']
#         for m in metrics_to_plot:
#             vals = df_results[m].values
#             methods = df_results['method'].values
#             plt.figure(figsize=(8, 4))
#             plt.bar(methods, vals)
#             plt.title(f"{m} by method")
#             plt.ylabel(m)
#             plt.xticks(rotation=45, ha='right')
#             plt.tight_layout()
#             plt.savefig(os.path.join(out_dir, f"{m}_by_method.png"))
#             plt.close()
#     except Exception as e:
#         print(f"Plotting failed: {e}")

    print(f"Saved summary CSV to {csv_path}")
    print(f"Saved detailed scores to {scores_path}")
#     print(f"Saved plots (if plotting succeeded) to {out_dir}")

    return df_results, detailed_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ranking methods and store metrics.')
    parser.add_argument('--spammer_type', type=str, default='random', choices=['random', 'anti', 'left', 'right', 'combine', 'compe'])
    parser.add_argument('--spammer_percent', type=int, default=10)
    parser.add_argument('--device', type=str, default=None, help='torch device string, e.g. cuda:0 or cpu')
    parser.add_argument('--data_dir', type=str, default='spammer_analysis/spammer_data', help='where the PC pickle files live')
    parser.add_argument('--faceage_pickle', type=str, default='data/PassageDF1.pickle')
    parser.add_argument('--out_root', type=str, default='results', help='root dir to write results')
    args = parser.parse_args()

    # choose device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    pc_name = f"Passage_{args.spammer_type}_{args.spammer_percent}.pickle"
    df_path = f"df_Passage_{args.spammer_type}_{args.spammer_percent}.csv"
    pc_path = os.path.join(args.data_dir, pc_name)
    df_path = os.path.join(args.data_dir, df_path)


    out_dir = os.path.join(args.out_root, f"passage_{args.spammer_type}_{args.spammer_percent}")

    run_all_methods(pc_path, args.faceage_pickle, device, out_dir, df_path)
