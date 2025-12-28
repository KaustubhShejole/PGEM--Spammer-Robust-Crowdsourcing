#!/usr/bin/env python3
# cleaned_experiment.py

import os
import csv
import gc
import random
from collections import defaultdict
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the parent directory (one directory up)
# e.g., '/path/to/my_project_root'
parent_dir = os.path.join(current_dir, '..')

# Insert the parent directory path at the beginning of sys.path
sys.path.insert(0, parent_dir)

# Project-specific imports (assumed available in your env)
from pgem import EMWrapper
from metrics import compute_acc, compute_weighted_acc
import opt_fair
from crowdkit.aggregation import NoisyBradleyTerry
import choix
from distribution_utils import crowd_bt_dist, logistic_preference_dist, comparisons_to_df, safe_kendalltau, to_numpy

# -------------------------
# Environment / seeds
# -------------------------
# If you want to force a specific GPU, set CUDA_VISIBLE_DEVICES before torch touches CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# -------------------------
# Parameters
# -------------------------
# Make sure N_array and K_array have the same length
N_array = [10, 50, 50, 100, 200, 100, 200, 200, 500]
K_array = [2, 2, 6, 8, 8, 10, 10, 50, 50]
# N_array = [50]
# K_array = [2]

# number of comparisons per experiment
m = 5000
max_iter = 100
# CSV output
csv_file = "results/final_-1_1_comparison_metrics_10.csv"
os.makedirs("results", exist_ok=True)

# -------------------------
# Helper functions
# -------------------------
def get_ground_truth_df(true_r):
    return pd.DataFrame({
        'label': list(range(len(true_r))),
        'score': true_r
    })

# -------------------------
# Write CSV header
# -------------------------
header = [
    "N", "K",
    # PGEM (mean, std) for acc, wacc, tau
    "PGEM_acc_mean", "PGEM_acc_std",
    "PGEM_wacc_mean", "PGEM_wacc_std",
    "PGEM_tau_mean", "PGEM_tau_std",
    # BT single-run
    "BT_acc_mean", "BT_acc_std",
    "BT_wacc_mean", "BT_wacc_std",
    "BT_tau_mean", "BT_tau_std",
    # BARP single-run
    "BARP_acc_mean", "BARP_acc_std",
    "BARP_wacc_mean", "BARP_wacc_std",
    "BARP_tau_mean", "BARP_tau_std",
    # RankCentrality single-run
    "RC_acc_mean", "RC_acc_std",
    "RC_wacc_mean", "RC_wacc_std",
    "RC_tau_mean", "RC_tau_std",
    # FactorBT single-run
    "FactorBT_acc_mean", "FactorBT_acc_std",
    "FactorBT_wacc_mean", "FactorBT_wacc_std",
    "FactorBT_tau_mean", "FactorBT_tau_std",
    # CrowdBT (mean,std) for acc, wacc, tau
    "CrowdBT_acc_mean", "CrowdBT_acc_std",
    "CrowdBT_wacc_mean", "CrowdBT_wacc_std",
    "CrowdBT_tau_mean", "CrowdBT_tau_std"
]

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)



# -------------------------
# Main experiment loop
# -------------------------
for N, K in zip(N_array, K_array):
    # create GT rewards and worker betas
    PGEM_accs, PGEM_waccs, PGEM_taus = [], [], []
    FactorBT_accs, FactorBT_waccs, FactorBT_taus = [], [], []
    BT_accs, BT_waccs, BT_taus = [], [], []
    BARP_accs, BARP_waccs, BARP_taus = [], [], []
    CrowdBT_accs, CrowdBT_waccs, CrowdBT_taus = [], [], []
    RC_accs, RC_waccs, RC_taus = [], [], []
    
    SEEDS = range(20, 30)
    for sd in SEEDS:
        torch.manual_seed(sd)
        np.random.seed(sd)
        random.seed(sd)

        true_r = (6 * torch.rand(N, device=device) - 3).clone()
        true_r = true_r - true_r.mean()
        gt_df = get_ground_truth_df(true_r.detach().cpu().numpy())
        true_beta = (2 * torch.rand(K, device=device) - 1).clone()

        sample_fn, flat_probs = logistic_preference_dist(true_r, true_beta)
        samples = sample_fn(m)

        df = comparisons_to_df(samples.cpu().tolist())

        # Convert to PC_faceage format: dictionary worker_id -> list of (winner,loser) pairs
        PC_passage = {}
        for performer, group in df.groupby('worker'):
            key = int(performer)
            if key not in PC_passage:
                PC_passage[key] = []
            for _, row in group.iterrows():
                winner = int(row["label"])
                l, r = int(row["left"]), int(row["right"])
                # store as (winner, loser)
                PC_passage[key].append((l, r) if winner == l else (r, l))

        PC_faceage = PC_passage
        all_pc_faceage = opt_fair._pc_without_reviewers(PC_faceage)
        size = len(gt_df)
        classes = [0] * size

        # === PGEM (averaged over seeds) ===
        
        for seed in range(10, 20):
            # --- 1. Run the EM algorithm (Keep this section) ---
            pg = EMWrapper(PC_faceage, max_iter=max_iter, device=device, random_seed=seed)
            r_est_tensor, beta_est_tensor, ll = pg.run_algorithm()
            
            # Skip if ANY element is NaN
            if np.isnan(r_est_tensor).any() or np.isnan(beta_est_tensor).any() or np.isnan(ll):
                print("Skipping nan")
                continue
            
            # --- 2. Convert to NumPy once for unified processing (Improvement) ---
            # Convert the estimated item scores (r_est) to a NumPy array immediately.
            r_est_np = to_numpy(r_est_tensor)
            gt_scores = gt_df['score'].to_numpy() # Cache ground truth scores

            # --- 3. Robust Sign-Flipping based on Kendall's Tau (Major Improvement) ---
            # Check rank correlation between estimated scores and ground truth scores.
            # If the correlation (tau) is negative, the scores are flipped (non-identifiability).
            # This is more stable than checking if accuracy is below 0.5.
            current_tau = safe_kendalltau(r_est_np, gt_scores)

            if current_tau < 0:
                # Flip the sign of the scores to match the convention (higher score = better item)
                r_est_np = -r_est_np

            # --- 4. Calculate and Append Metrics (Simplified and Cleaned) ---
            # Since r_est_np is the final, correctly-signed array, use it directly for all metrics.
            PGEM_accs.append(compute_acc(gt_df, r_est_np, device))
            PGEM_waccs.append(compute_weighted_acc(gt_df, r_est_np, device))
            # Recalculate Tau for the correctly-signed scores (it should now be positive)
            PGEM_taus.append(safe_kendalltau(r_est_np, gt_scores))


        # === BT (choix) ===
        try:
            bt_scores = choix.opt_pairwise(size, all_pc_faceage, alpha=0, method='Newton-CG')
        except Exception as e:
            # fallback: zeros
            print(f"choix.opt_pairwise failed for N={N},K={K} with error {e}; using zeros")
            bt_scores = np.zeros(size)
        
        BT_tau = safe_kendalltau(bt_scores, gt_df['score'].to_numpy())
        if BT_tau < 0:
            bt_scores = -bt_scores
        BT_acc = compute_acc(gt_df, bt_scores, device)
        BT_wacc = compute_weighted_acc(gt_df, bt_scores, device)
        BT_tau = safe_kendalltau(bt_scores, gt_df['score'].to_numpy())
        BT_accs.append(BT_acc)
        BT_waccs.append(BT_wacc)
        BT_taus.append(BT_tau)

        # === BARP (opt_fair) ===
        try:
            FaceAge = opt_fair.BARP(data=PC_faceage, penalty=0, classes=classes, device=device)
            annot_bt_temp, annot_bias = opt_fair._alternate_optim_torch(size, K, FaceAge, iters=100)
            annot_bt_np = to_numpy(annot_bt_temp)
            BARP_tau = safe_kendalltau(annot_bt_np, gt_df['score'].to_numpy())
            if BARP_tau < 0:
                annot_bt_np = -annot_bt_np
            BARP_acc = compute_acc(gt_df, annot_bt_np, device)
            BARP_wacc = compute_weighted_acc(gt_df, annot_bt_np, device)
            BARP_tau = safe_kendalltau(annot_bt_np, gt_df['score'].to_numpy())
        except Exception as e:
            print(f"BARP failed for N={N},K={K} with error {e}; using zeros")
            BARP_acc = BARP_wacc = BARP_tau = 0.0
        BARP_accs.append(BARP_acc)
        BARP_waccs.append(BARP_wacc)
        BARP_taus.append(BARP_tau)

        # === RankCentrality (opt_fair) ===
        try:
            rc_obj = opt_fair.RankCentrality(device)
            A = rc_obj.matrix_of_comparisons(size, all_pc_faceage)
            P = rc_obj.trans_prob(A)
            pi = rc_obj.stationary_dist(P)
            rc_scores = np.log(to_numpy(pi))
            RC_tau = safe_kendalltau(rc_scores, gt_df['score'].to_numpy())
            if RC_tau < 0:
                rc_scores = -rc_scores
            RC_acc = compute_acc(gt_df, rc_scores, device)
            RC_tau = safe_kendalltau(rc_scores, gt_df['score'].to_numpy())
            RC_wacc = compute_weighted_acc(gt_df, rc_scores, device)
            
        except Exception as e:
            print(f"RankCentrality failed for N={N},K={K} with error {e}; using zeros")
            RC_acc = RC_wacc = RC_tau = 0.0
        
        RC_accs.append(RC_acc)
        RC_waccs.append(RC_wacc)
        RC_taus.append(RC_tau)
        


        # === FactorBT (NoisyBradleyTerry from crowdkit) ===
        try:
            agg_noisybt = NoisyBradleyTerry(n_iter=10).fit_predict(df)
            agg_noisybt_df = pd.DataFrame(list(agg_noisybt.items()), columns=['label', 'score']).sort_values(by='label')
            factorbt_scores = agg_noisybt_df['score'].to_numpy()
            FactorBT_tau = safe_kendalltau(factorbt_scores, gt_df['score'].to_numpy())
            if FactorBT_tau < 0:
                factorbt_scores = -factorbt_scores
            FactorBT_acc = compute_acc(gt_df, factorbt_scores, device)
            FactorBT_wacc = compute_weighted_acc(gt_df, factorbt_scores, device)
            FactorBT_tau = safe_kendalltau(factorbt_scores, gt_df['score'].to_numpy())
        except Exception as e:
            print(f"FactorBT (NoisyBradleyTerry) failed for N={N},K={K} with error {e}; using zeros")
            FactorBT_acc = FactorBT_wacc = FactorBT_tau = 0.0
        
        FactorBT_accs.append(FactorBT_acc)
        FactorBT_waccs.append(FactorBT_wacc)
        FactorBT_taus.append(FactorBT_tau)

        
        for seed in range(10):
            try:
                crowdbt_test = opt_fair.CrowdBT_3_0(data=PC_faceage, penalty=0, device=device, random_seed=seed)
                crowdbt_scores, _ = crowdbt_test.alternate_optim(size, K)
                crowdbt_scores_np = to_numpy(crowdbt_scores)
                tau = safe_kendalltau(crowdbt_scores_np, gt_df['score'].to_numpy())
                if tau < 0:
                    crowdbt_scores_np = -crowdbt_scores_np
                CrowdBT_accs.append(compute_acc(gt_df, crowdbt_scores_np, device))
                CrowdBT_waccs.append(compute_weighted_acc(gt_df, crowdbt_scores_np, device))
                CrowdBT_taus.append(safe_kendalltau(crowdbt_scores_np, gt_df['score'].to_numpy()))
            except Exception as e:
                print(f"CrowdBT seed {seed} failed for N={N},K={K} with error {e}; appending zeros")
                continue


        
        del true_r, true_beta, samples, df, gt_df  # delete variables
        torch.cuda.empty_cache()  # free memory on GPU (if applicable)
    
    # --- Save row ---
    row = [
        N, K,
        np.mean(PGEM_accs), np.std(PGEM_accs),
        np.mean(PGEM_waccs), np.std(PGEM_waccs),
        np.mean(PGEM_taus), np.std(PGEM_taus),
        np.mean(BT_accs), np.std(BT_accs),
        np.mean(BT_waccs), np.std(BT_waccs),
        np.mean(BT_taus), np.std(BT_taus),
        np.mean(BARP_accs), np.std(BARP_accs),
        np.mean(BARP_waccs), np.std(BARP_waccs),
        np.mean(BARP_taus), np.std(BARP_taus),
        np.mean(RC_accs), np.std(RC_accs),
        np.mean(RC_waccs), np.std(RC_waccs),
        np.mean(RC_taus), np.std(RC_taus),
        np.mean(FactorBT_accs), np.std(FactorBT_accs),
        np.mean(FactorBT_waccs), np.std(FactorBT_waccs),
        np.mean(FactorBT_taus), np.std(FactorBT_taus),
        np.mean(CrowdBT_accs), np.std(CrowdBT_accs),
        np.mean(CrowdBT_waccs), np.std(CrowdBT_waccs),
        np.mean(CrowdBT_taus), np.std(CrowdBT_taus)
    ]

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    # Print summary line
    print(f"N={N}, K={K}, PGEM_acc={np.mean(PGEM_accs):.4f}±{np.std(PGEM_accs):.4f}, "
          f"BT_acc={BT_acc:.4f}, BARP_acc={BARP_acc:.4f}, RC_acc={RC_acc:.4f}, "
          f"FactorBT_acc={np.mean(FactorBT_accs):.4f}±{np.std(FactorBT_accs):.4f}, CrowdBT_acc={np.mean(CrowdBT_accs):.4f}±{np.std(CrowdBT_accs):.4f}")

print(f"Metrics saved to {csv_file}")