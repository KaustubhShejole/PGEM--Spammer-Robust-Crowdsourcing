import numpy as np
import pandas as pd
import os
import torch
import scipy.sparse.linalg
from tqdm import tqdm
import random

# Helper functions
def estimate_labels(true_r, r_est, comparisons):
    labels = []
    for a,b,_ in comparisons:
        if true_r[a] > true_r[b]:
            labels.append(1 if r_est[a] > r_est[b] else 0)
        else:
            labels.append(1 if r_est[a] < r_est[b] else 0)
    return labels

def get_original_label(true_r, comparisons):
    labels = []
    for a,b,_ in comparisons:
        labels.append(1 if true_r[a] > true_r[b] else 0)
    return labels


class EMWrapper:
    def __init__(self, df_by_worker, max_iter, device, random_seed=45):
#         if device not in ("cuda", "cpu"):
#             raise ValueError(f"Unsupported device: {device}. Use 'cuda' or 'cpu'.")
        self.device = device
        print(self.device)
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.comparisons = self._get_compatible_data(df_by_worker)
#         print(self.comparisons[:5])
        self.max_iter = max_iter
        

        # Number of workers
        self.num_workers = len(df_by_worker)

        # Number of items = 1 + max index seen
        max_index = max(
            max(winner, loser) for winner, loser, _ in self.comparisons
        )
        self.num_items = max_index + 1
        
#         print("init done", self.comparisons)

    def _get_compatible_data(self, data):
        """
        PC_data: numpy array (dtype=object) or list of lists
            PC_data[i] = list of [winner_index, loser_index] for worker i

        Returns:
            A shuffled list of comparisons in the form [winner_index, loser_index, worker_id]
        """
        comparisons = []
        worker_id = 0
        while worker_id in data:
            cc = data[worker_id]
            for comp in cc:
                winner, loser = comp
                comparisons.append([winner, loser, worker_id])
            worker_id += 1

        random.shuffle(comparisons)
        return np.array(comparisons, dtype=np.int64)
    
    def run_algorithm(self):
        pgem = PolyaGamma_EM(self.num_items, self.num_workers, max_iter=self.max_iter, device=self.device, random_seed=self.random_seed)
        r_est, b_est, ll = pgem.fit(self.comparisons)
        return r_est, b_est, ll



class PolyaGamma_EM:
    def __init__(self, num_items, num_workers, max_iter=500, epsilon=1e-6, device='cuda', random_seed=45):
        self.device = device
        print(self.device)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)  # For multi-GPU

        np.random.seed(self.random_seed)

        self.num_items = num_items
        self.num_workers = num_workers
        self.max_iter = max_iter
        self.epsilon = epsilon

        # Initialize parameters with identifiability constraints
        self.r = torch.randn(num_items, device=device)
#         self.r = torch.zeros(num_items, device=device)
        # self.r = 1000.0 * torch.randn(num_items, device=device)
        self.r -= self.r.mean()  # Zero-mean initialization

        self.beta = (2 * torch.rand(num_workers, device=device) - 1.0).clone()
        # self.beta = 0 + 0.01 * torch.randn(num_workers, device=device)

        # Pre-allocated tensors for comparison processing
        self.winner_idx = None
        self.loser_idx = None
        self.worker_idx = None

    def _log_likelihood(self):
#         print("BETA:", self.beta)
#         print("rewards:", self.r)
        logits = self.beta[self.worker_idx] * (self.r[self.winner_idx] - self.r[self.loser_idx])
#         print("LOGITS:", logits)
#         are_all_close_to_zero = torch.allclose(logits, torch.zeros_like(logits), rtol=1e-6, atol=1e-6)
#         print("All elements are close to zero:", are_all_close_to_zero)
#         print("sigmoid logits:", torch.sigmoid(logits))
#         print("log_likelihood:", torch.log(torch.sigmoid(logits) + 1e-8))
        return torch.log(torch.sigmoid(logits) + 1e-8).mean()

    def _compute_pg_expectations(self):
        deltas = self.r[self.winner_idx] - self.r[self.loser_idx]
        x = self.beta[self.worker_idx] * deltas
        
        abs_x = x.abs()
        
        kappas = torch.where(
            abs_x < 1e-8,
            0.25 - (x**2)/48.0,                # Taylor approx for small x
            torch.tanh(x/2) / (2*x)            # exact for other x
        )

#         mask = x.abs() < 1e-8

#         # Vectorized computation using Taylor approximation for stability
#         kappas = torch.zeros_like(x)
#         kappas[mask] = 0.25 - (x[mask]**2)/48.0
#         kappas[~mask] = torch.tanh(x[~mask]/2) / (2*x[~mask])

        return kappas
    
    def _update_competencies(self, kappas):
        deltas = self.r[self.winner_idx] - self.r[self.loser_idx]

        # Vectorized contributions
        num_contrib = 0.5 * deltas
        denom_contrib = kappas * deltas**2

        # Aggregate by worker
        numerator = torch.zeros(self.num_workers, device=self.device)
        denominator = torch.zeros(self.num_workers, device=self.device)
        numerator.index_add_(0, self.worker_idx, num_contrib)
        denominator.index_add_(0, self.worker_idx, denom_contrib)

        # Update competencies with stability checks
        valid = denominator > 1e-6
        self.beta[valid] = numerator[valid] / denominator[valid]

#         self.beta[~valid] = 0  # Default for workers with no valid comparisons
#         if len(self.beta[~valid]) > 0:
#             print(f"Invalid betas: {self.beta[~valid]}")
    
    
    import torch

    def _update_rewards(self, kappas):
        num_items = self.num_items
        device = self.device
        beta = self.beta

        # 1. Faster Sparse Construction
        beta_sq = beta[self.worker_idx] ** 2
        summands = beta_sq * kappas

        rows = torch.cat([self.winner_idx, self.loser_idx, self.winner_idx, self.loser_idx])
        cols = torch.cat([self.winner_idx, self.loser_idx, self.loser_idx, self.winner_idx])
        vals = torch.cat([summands, summands, -summands, -summands])

        # Keep H as a sparse tensor to save memory and avoid dense conversion overhead
        indices = torch.stack([rows, cols], dim=0)
        H_sparse = torch.sparse_coo_tensor(indices, vals, (num_items, num_items), device=device).coalesce()

        # 2. Build RHS vector b (Vectorized)
        b = torch.zeros(num_items, device=device)
        b_i_vals = 0.5 * beta[self.worker_idx]
        b.index_add_(0, self.winner_idx, b_i_vals)
        b.index_add_(0, self.loser_idx, -b_i_vals)

        # 3. GPU-Accelerated Conjugate Gradient
        # We replace scipy.sparse.linalg.cg with a native PyTorch CG solver
        # This avoids the costly .cpu().numpy() transfer.
        r = self._torch_cg(H_sparse, b, r0=self.r, max_iter=500, tol=1e-5, reg=1e-8)

        # Apply zero-mean constraint
        self.r = r - r.mean()

    def _torch_cg_copy(self, A_sparse, b, r0=None, max_iter=500, tol=1e-5, reg=1e-8):
        """
        Pure PyTorch implementation of Conjugate Gradient.
        Solves (A + reg*I)x = b
        """
        x = r0 if r0 is not None else torch.zeros_like(b)

        # Helper for matrix-vector product (A + reg*I) @ x
        def mvp(v):
            return torch.matmul(A_sparse, v.unsqueeze(1)).squeeze(1) + reg * v

        r = b - mvp(x)
        if torch.norm(r) < tol:
            return x

        p = r.clone()
        rdotr = torch.dot(r, r)

        for i in range(max_iter):
            Ap = mvp(p)
            alpha = rdotr / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap

            new_rdotr = torch.dot(r, r)
            if torch.sqrt(new_rdotr) < tol:
                break

            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def _torch_cg(self, A_sparse, b, r0=None, max_iter=500, tol=1e-5, reg=1e-8):
        # 1. High-precision promotion
        orig_dtype = b.dtype
        b_64 = b.detach().to(torch.float64)
        A_64 = A_sparse.detach().to(torch.float64)
        x = r0.to(torch.float64) if r0 is not None else torch.zeros_like(b_64)

        def mvp(v):
            if A_64.is_sparse:
                return torch.sparse.mm(A_64, v.unsqueeze(1)).squeeze(1) + reg * v
            return torch.matmul(A_64, v.unsqueeze(1)).squeeze(1) + reg * v

        # 2. Construct Diagonal Preconditioner (Jacobi)
        # Extract diagonal: A_ii + reg
        if A_64.is_sparse:
            # For sparse COO, find indices where row == col
            indices = A_64._indices()
            values = A_64._values()
            mask = (indices[0] == indices[1])
            diag = torch.zeros(b_64.size(0), device=b_64.device, dtype=torch.float64)
            diag[indices[0][mask]] = values[mask]
            diag += reg
        else:
            diag = torch.diag(A_64) + reg

        # M_inv scales the residual to be near 1.0 based on the matrix scale
        M_inv = 1.0 / (diag + 1e-12) 

        r = b_64 - mvp(x)
        z = M_inv * r  # This is your "scaled" residual
        p = z.clone()
        rdotz = torch.dot(r, z)

        if torch.sqrt(rdotz) < tol:
            return x.to(orig_dtype)

        for i in range(max_iter):
            Ap = mvp(p)

            denom = torch.dot(p, Ap)
            if denom <= 1e-16: # Stability break
                break

            alpha = rdotz / denom
            x = x + alpha * p

            # Periodic refresh for stability
            if i % 50 == 0:
                r = b_64 - mvp(x)
            else:
                r = r - alpha * Ap

            z = M_inv * r  # Apply scaling/preconditioning
            new_rdotz = torch.dot(r, z)

            if torch.norm(r) < tol:
                break

            beta = new_rdotz / (rdotz + 1e-16)
            p = z + beta * p
            rdotz = new_rdotz

            if torch.isnan(x).any():
                break

        return x.to(orig_dtype)
    
    def _check_convergence(self, prev_r, prev_beta):
        r_diff = torch.norm(self.r - prev_r)
        beta_diff = torch.norm(self.beta - prev_beta)
        return r_diff < self.epsilon and beta_diff < self.epsilon
    
    def fit(self, comparisons):
        # Convert comparisons to tensor indices
        comparisons_tensor = torch.tensor(comparisons, dtype=torch.long, device=self.device)
        self.winner_idx = comparisons_tensor[:, 0]
        self.loser_idx = comparisons_tensor[:, 1]
        self.worker_idx = comparisons_tensor[:, 2]

        prev_r = self.r.clone()
        prev_beta = self.beta.clone()
        prev_ll = -float("inf")
        
#         print(prev_r, prev_beta, "max iter: ", self.max_iter)

        for iter in tqdm(range(self.max_iter)):
            # E-step: Compute Polya-Gamma expectations
            kappas = self._compute_pg_expectations()
#             print("printing kappas for first step", kappas)

            # M-step: Update parameters
            for _ in range(10):
#                 print("M step iter number: ", _)
                self._update_competencies(kappas) # beta vector
#                 print("beta done")
                self._update_rewards(kappas)      # r vector
#                 print("r done")

            # Enforce identifiability constraints
            self.r -= self.r.mean()
            self.beta = torch.clamp(self.beta, min=-1, max=1)

            # Compute log-likelihood
            ll = self._log_likelihood().item()
            if iter % 100 == 0:
                print(f"Iter {iter:03d}: Log-likelihood = {ll:.6f}")

            # Convergence based on log-likelihood change
            if abs(ll - prev_ll) < self.epsilon:
                print(f"Converged at iter {iter}, Log-likelihood change = {ll - prev_ll:.6e}")
                break

            prev_ll = ll

            prev_r = self.r.clone()
            prev_beta = self.beta.clone()

        return self.r.cpu().numpy(), self.beta.cpu().numpy(), self._log_likelihood().item()

def to_tensor(x, dtype=None, device=None):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)