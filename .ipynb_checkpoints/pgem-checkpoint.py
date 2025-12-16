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
        self.comparisons = self._get_compatible_data(df_by_worker)
        print(self.comparisons[:5])
        self.max_iter = max_iter
        random.seed(self.random_seed)

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
            with a fixed random seed (42).
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


import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm 

# Define the target data type
DTYPE = torch.float64 

# Utility function for converting arrays to tensors
def to_tensor(x, dtype=DTYPE, device=None):
    """Convert numpy array or list to torch tensor."""
    if isinstance(x, np.ndarray):
        # Ensure NumPy array is float64 before conversion
        if x.dtype != np.float64:
             x = x.astype(np.float64)
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)

class PolyaGamma_EM:
    def __init__(self, num_items, num_workers, max_iter=500, epsilon=1e-6, device='cuda', random_seed=45):
        self.device = device
        print(f"Device: {self.device}")
        self.random_seed = random_seed
        
        # Set seeds for reproducibility
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)

        self.num_items = num_items
        self.num_workers = num_workers
        self.max_iter = max_iter
        self.epsilon = epsilon
        
        # --- FIX: Initialize all main tensors with DTYPE=torch.float64 ---
        
        # Initialize parameters with zero-mean constraint for r
        self.r = torch.randn(num_items, device=device, dtype=DTYPE)
        self.r -= self.r.mean() 
        
        # Initialize beta in [-1, 1]
        self.beta = (2 * torch.rand(num_workers, device=device, dtype=DTYPE) - 1.0)
        
        # Pre-allocated tensors for comparison processing (indices)
        self.winner_idx = None
        self.loser_idx = None
        self.worker_idx = None

    def _log_likelihood(self):
        """
        Calculates the mean log-likelihood using F.logsigmoid for 
        numerical stability, preventing premature convergence due to NaN/inf.
        """
        logits = self.beta[self.worker_idx] * (self.r[self.winner_idx] - self.r[self.loser_idx])
        return F.logsigmoid(logits).mean()

    def _compute_pg_expectations(self):
        """E-step: Compute E[omega] (kappas) with stable computation."""
        deltas = self.r[self.winner_idx] - self.r[self.loser_idx]
        x = self.beta[self.worker_idx] * deltas
        
        abs_x = x.abs()
        
        # Numerically stable computation of E[omega] = kappa
        kappas = torch.where(
            abs_x < 1e-6, 
            0.25 - (x**2) / 48.0, 
            0.5 * torch.tanh(x / 2.0) / (x / 2.0)
        )
        return kappas
    
    def _update_competencies(self, kappas):
        """
        M-step (Beta): Update worker competencies (beta).
        FIX: Ensures numerator/denominator are created with DTYPE.
        """
        deltas = self.r[self.winner_idx] - self.r[self.loser_idx]

        num_contrib = 0.5 * deltas
        denom_contrib = kappas * deltas**2

        # FIX: Explicitly set DTYPE for aggregation tensors
        numerator = torch.zeros(self.num_workers, device=self.device, dtype=DTYPE)
        denominator = torch.zeros(self.num_workers, device=self.device, dtype=DTYPE)
        
        # This is where the original RuntimeError occurred. Now types match (Double).
        numerator.index_add_(0, self.worker_idx, num_contrib)
        denominator.index_add_(0, self.worker_idx, denom_contrib)

        valid = denominator > 1e-8 
        self.beta[valid] = numerator[valid] / denominator[valid]
            
        # Re-clamp beta after update
        self.beta = torch.clamp(self.beta, min=-1.0, max=1.0)
    
    def _update_rewards(self, kappas):
        """
        M-step (R): Update item rewards (r) by solving the linear system Hr = b.
        FIX: Ensures b is created with DTYPE and r_np is converted back with DTYPE.
        """
        num_items = self.num_items
        device = self.device

        beta_sq = self.beta[self.worker_idx] ** 2
        summands = beta_sq * kappas

        # --- Constructing the Hessian (H) Matrix (Sparse) ---
        rows = torch.cat([self.winner_idx, self.loser_idx, self.winner_idx, self.loser_idx])
        cols = torch.cat([self.winner_idx, self.loser_idx, self.loser_idx, self.winner_idx])
        vals = torch.cat([summands, summands, -summands, -summands])
        
        # --- Constructing the RHS Vector (b) ---
        # FIX: Explicitly set DTYPE
        b = torch.zeros(num_items, device=device, dtype=DTYPE)
        b_i_vals = 0.5 * self.beta[self.worker_idx]
        b.index_add_(0, self.winner_idx, b_i_vals)
        b.index_add_(0, self.loser_idx, -b_i_vals)

        # --- Solving Hr = b using SciPy's Conjugate Gradient (CG) Solver ---
        
        # 1. Convert PyTorch COO components to SciPy CSR format
        rows_np = rows.cpu().numpy()
        cols_np = cols.cpu().numpy()
        # SciPy/NumPy defaults to float64, which is DTYPE
        vals_np = vals.cpu().numpy() 
        
        H_sparse = sp.coo_matrix(
            (vals_np, (rows_np, cols_np)), 
            shape=(num_items, num_items)
        ).tocsr()
        
        # 2. Add regularization/conditioning term
        reg_factor = 1e-8 
        H_reg = H_sparse + reg_factor * sp.identity(num_items, format='csr')

        b_np = b.cpu().numpy()

        # 3. Solve the linear system
        r_np, info = spla.cg(H_reg, b_np, atol=1e-6, maxiter=500)
        
        # Check for CG solver failure
        if info != 0:
            tqdm.write(f"Warning: CG solver failed to converge (status: {info}). This can indicate an issue with regularization or data.")

        # 4. Convert back to torch and enforce zero-mean constraint
        # FIX: Explicitly use DTYPE for conversion
        r = torch.from_numpy(r_np).to(device=device, dtype=DTYPE)
        r -= r.mean()
        
        self.r = r
    
    def fit(self, comparisons):
        """
        Fits the model to a list of comparisons: [(win_idx, lose_idx, worker_idx), ...].
        """
        # Convert comparisons to tensor indices (indices should be long/int)
        comparisons_tensor = torch.tensor(comparisons, dtype=torch.long, device=self.device)
        self.winner_idx = comparisons_tensor[:, 0]
        self.loser_idx = comparisons_tensor[:, 1]
        self.worker_idx = comparisons_tensor[:, 2]

        # FIX: Calculate and print initial LL
        prev_ll = self._log_likelihood().item() 
        tqdm.write(f"Initial Log-likelihood: {prev_ll:.6f}")
        
        M_STEP_INNER_ITERS = 5 

        # --- Optimization Loop ---
        for iter in tqdm(range(self.max_iter), desc="PG-EM Optimization"):
            # E-step
            kappas = self._compute_pg_expectations()

            # M-step
            for _ in range(M_STEP_INNER_ITERS):
                self._update_competencies(kappas) 
                self._update_rewards(kappas)      
            
            # Enforce identifiability constraints
            self.r -= self.r.mean() 
            self.beta = torch.clamp(self.beta, min=-1.0, max=1.0)
            
            # --- Convergence Check ---
            ll = self._log_likelihood().item()
            
            if iter % 100 == 0:
                tqdm.write(f"Iter {iter:03d}: Log-likelihood = {ll:.6f}")

            # Convergence based on log-likelihood change
            if abs(ll - prev_ll) < self.epsilon and iter > 0:
                tqdm.write(f"Converged at iter {iter}, Log-likelihood change = {ll - prev_ll:.6e}")
                break

            prev_ll = ll

        # Return CPU tensors
        return self.r.cpu(), self.beta.cpu(), self._log_likelihood().item()
    
    

def to_tensor(x, dtype=None, device=None):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)

class PolyaGamma_EM_2_0:
    def __init__(self, num_items, num_workers, r=None, beta=None, max_iter=500, epsilon=1e-6, device='cuda', random_seed=45):
        self.device = device
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
        
        if r is None:
            # Initialize parameters with identifiability constraints
            self.r = torch.randn(num_items, device=device)
            # self.r = 1000.0 * torch.randn(num_items, device=device)
            self.r -= self.r.mean()  # Zero-mean initialization
        else:
            self.r = to_tensor(r, device=device)
        if beta is None:
            self.beta = (2 * torch.rand(num_workers, device=device) - 1.0).clone()
        # self.beta = 0 + 0.01 * torch.randn(num_workers, device=device)
        else:
            self.beta = to_tensor(beta, device=device)

        # Pre-allocated tensors for comparison processing
        self.winner_idx = None
        self.loser_idx = None
        self.worker_idx = None

    def _log_likelihood(self):
        logits = self.beta[self.worker_idx] * (self.r[self.winner_idx] - self.r[self.loser_idx])
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
        if len(self.beta[~valid]) > 0:
            print(f"Invalid betas: {self.beta[~valid]}")
    
    def _update_rewards(self, kappas):
        num_items = self.num_items
        device = self.device

        r = self.r
        beta = self.beta

        beta_sq = beta[self.worker_idx] ** 2
        summands = beta_sq * kappas

#         print(len(self.winner_idx))        
        rows = torch.cat([self.winner_idx, self.loser_idx, self.winner_idx, self.loser_idx])
        cols = torch.cat([self.winner_idx, self.loser_idx, self.loser_idx, self.winner_idx])
        vals = torch.cat([summands, summands, -summands, -summands])

        # Now create a sparse COO matrix
        H_sparse = torch.sparse_coo_tensor(
            torch.stack([rows, cols], dim=0),
            vals,
            (num_items, num_items),
            device=device
        )
        # Optionally convert to dense if you must (careful with memory!)
        H = H_sparse.to_dense()
#         print(H)

        b = torch.zeros(num_items, device=device)
        b_i_vals = 0.5 * beta[self.worker_idx]
        b.index_add_(0, self.winner_idx, b_i_vals)
        b.index_add_(0, self.loser_idx, -b_i_vals)

#         b = torch.zeros(num_items, device=device)
#         b_i_vals = 0.5 * beta[self.worker_idx]
#         b.index_add_(0, self.winner_idx, b_i_vals)
#         b.index_add_(0, self.loser_idx, -b_i_vals)



#         H = torch.zeros((num_items, num_items), device=device)        
#         for i in range(len(self.winner_idx)):
#             print(i)
#             wi = self.winner_idx[i]
#             li = self.loser_idx[i]
#             ki = self.worker_idx[i]
#             summand_i = summands[i]
#             beta_i = beta[ki]

#             # diagonal terms
#             H[wi, wi] += summand_i
#             H[li, li] += summand_i
#             # non diagonal terms
#             H[wi, li] -= summand_i
#             H[li, wi] -= summand_i

#             # RHS vector b
#             b[wi] += 0.5 * beta_i
#             b[li] -= 0.5 * beta_i

#         print("H matrix made")
        # Gradient descent setup
        
        H_reg = H + 1e-8 * torch.eye(num_items, device=device)
        H_reg_np = H_reg.cpu().numpy()
        b_np = b.cpu().numpy()

        r_np, info = scipy.sparse.linalg.cg(H_reg_np, b_np, atol=1e-5, maxiter=500)
        r = torch.from_numpy(r_np).to(device)
        r -= r.mean()

#         lr = 1e-2  # learning rate
#         max_iter = 500
#         tol = 1e-5

#         for _ in range(max_iter):
#             grad = H @ r - b
#             r = r - lr * grad
#             r = r - r.mean()  # zero-mean constraint

#             if torch.norm(grad) < tol:
#                 break

        self.r = r
    
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
#         print(self.winner_idx[:5])
#         print(self.loser_idx[:5])
#         print(self.worker_idx[:5])

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
            if iter % 100 == 1:
                print(f"Iter {iter:03d}: Log-likelihood = {ll:.6f}")

            # Convergence based on log-likelihood change
            if abs(ll - prev_ll) < self.epsilon:
                print(f"Converged at iter {iter}, Log-likelihood change = {ll - prev_ll:.6e}")
                break

            prev_ll = ll

            prev_r = self.r.clone()
            prev_beta = self.beta.clone()

        return self.r.cpu().numpy(), self.beta.cpu().numpy(), self._log_likelihood().item()
