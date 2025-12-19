import os
import csv
import gc
import random
from collections import defaultdict

import sys
sys.path.insert(0, "../")


import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau


from pgem import EMWrapper
from metrics import compute_acc, compute_weighted_acc
import opt_fair
from crowdkit.aggregation import NoisyBradleyTerry
import choix


from distribution_utils import crowd_bt_dist, logistic_preference_dist, comparisons_to_df, safe_kendalltau, to_numpy
from grad_em import *

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
csv_file = "results/gradient_0_1_comparison_metrics.csv"
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
    "Grad_acc_mean", "Grad_acc_std",
    "Grad_wacc_mean", "Grad_wacc_std",
    "Grad_tau_mean", "Grad_tau_std",
]

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)



# -------------------------
# Main experiment loop
# -------------------------
for N, K in zip(N_array, K_array):
    # create GT rewards and worker betas
    Grad_accs, Grad_waccs, Grad_taus = [], [], []
    
    SEEDS = range(20, 30)
    for sd in SEEDS:
        torch.manual_seed(sd)
        np.random.seed(sd)
        random.seed(sd)
        
        device = "cpu"
        true_r = (6 * torch.rand(N, device=device) - 3).clone()
        true_r = true_r - true_r.mean()
        gt_df = get_ground_truth_df(true_r.detach().cpu().numpy())
        true_beta = torch.rand(K, device=device)

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
            raw_data = samples

            # Convert to tensors
            winners = torch.tensor([d[0] for d in raw_data])
            losers = torch.tensor([d[1] for d in raw_data])
            annotators = torch.tensor([d[2] for d in raw_data])
            data_tensors = (winners, losers, annotators)

            model = GradientEM(N, K, random_seed=seed)

            # Separate optimizers for alternating updates
            opt_r = torch.optim.Adam(model.item_rewards.parameters(), lr=0.01)
            opt_beta = torch.optim.Adam(model.worker_betas.parameters(), lr=0.01)

            r, beta = train_with_convergence(model, data_tensors, opt_r, opt_beta)
            
            # Skip if ANY element is NaN
            if np.isnan(r).any() or np.isnan(beta).any():
                print("Skipping nan")
                continue
            
            # --- 2. Convert to NumPy once for unified processing (Improvement) ---
            # Convert the estimated item scores (r_est) to a NumPy array immediately.
            r_est_np = to_numpy(r)
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
            Grad_accs.append(compute_acc(gt_df, r_est_np, device))
            Grad_waccs.append(compute_weighted_acc(gt_df, r_est_np, device))
            # Recalculate Tau for the correctly-signed scores (it should now be positive)
            Grad_taus.append(safe_kendalltau(r_est_np, gt_scores))
    # --- Save row ---
    row = [
        N, K,
        np.mean(Grad_accs), np.std(Grad_accs),
        np.mean(Grad_waccs), np.std(Grad_waccs),
        np.mean(Grad_taus), np.std(Grad_taus)
    ]

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    # Print summary line
    print(f"N={N}, K={K}, GradEM_acc={np.mean(Grad_accs):.4f}±{np.std(Grad_accs):.4f}")