import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.stats import kendalltau

def crowd_bt_dist(r, beta):
    """
    Returns a sample function and the flattened probability tensor for CrowdBT.
    
    Parameters
    ----------
    r : torch.Tensor, shape (N,)
        True item scores (latent utilities).
    beta : torch.Tensor, shape (K,)
        Worker reliabilities (0 <= beta <= 1).

    Returns
    -------
    sample_fn : callable
        Function sample_fn(n) returns n samples of shape (n, 3) with
        columns (winner_index, loser_index, worker_index)
    flat_probs : torch.Tensor
        Flattened probabilities for all (winner, loser, worker) triplets.
    """
    r = r.view(-1)       # (N,)
    beta = beta.view(-1) # (K,)
    N = r.shape[0]
    K = beta.shape[0]

    # Compute pairwise probabilities for all item pairs (Bradley-Terry component)
    r_i = r.view(N, 1)      # (N,1)
    r_j = r.view(1, N)      # (1,N)
    p_win = torch.sigmoid(r_i - r_j)  # (N,N)

    # Broadcast with beta for worker reliability (CrowdBT probability)
    # P(i > j | worker k) = beta_k * P(i > j) + (1 - beta_k) * P(j > i)
    p_win_workers = beta.view(1, 1, K) * p_win.unsqueeze(-1) \
                  + (1 - beta).view(1, 1, K) * (1 - p_win).unsqueeze(-1)  # (N,N,K)

    # Mask self-comparisons
    mask = torch.eye(N, device=r.device).unsqueeze(-1)  # (N,N,1)
    probs = p_win_workers * (1 - mask)

    # Flatten probabilities for categorical sampling
    flat_probs = probs.flatten()
    total = flat_probs.sum()
    if total.item() == 0:
        raise RuntimeError("All sampling probabilities are zero (check r/beta).")
    flat_probs = flat_probs / total

    cat = torch.distributions.Categorical(flat_probs)

    def sample_fn(n):
        samples = cat.sample((n,))  # indices in [0, N*N*K)
        # Convert flat index back to (winner, loser, worker) indices
        winner = samples // (N * K)
        loser = (samples % (N * K)) // K
        worker = samples % K
        return torch.stack([winner, loser, worker], dim=-1)

    return sample_fn, flat_probs


def comparisons_to_df(comparisons):
    """
    comparisons: iterable of [win, lose, worker] (indices)
    returns DataFrame with columns left (winner), right (loser), worker, label (=left)
    """
    df = pd.DataFrame(comparisons, columns=['left', 'right', 'worker']).assign(label=lambda d: d['left'])
    return df

def safe_kendalltau(x, y):
    """Wrap kendalltau and replace nan with 0.0"""
    try:
        tau, p = kendalltau(x, y)
        if np.isnan(tau):
            return 0.0
        return tau
    except Exception:
        # Catch other errors, e.g., if input arrays are too short/constant
        return 0.0

def to_numpy(x):
    """Convert tensor/scalar/list to numpy array"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return np.array(x)


def logistic_preference_dist(r, beta):
    """
    Constructs a sampling function from the logistic preference distribution (CrowdBT probability),
    excluding self-comparisons (i.e., where a == b).

    Args:
        r (torch.Tensor): A 1D tensor of item rewards (latent utilities) of shape (N,).
        beta (torch.Tensor): A 1D tensor of worker competencies of shape (K,).

    Returns:
        sample_fn (function): Function that samples (a, b, c) tuples where a ≠ b.
        flat_probs (torch.Tensor): Flattened sampling probabilities of shape (N*N*K,), with 0 for a == b entries.
    """
    r = r.view(-1)
    beta = beta.view(-1)
    N = r.shape[0]
    K = beta.shape[0]

    # Compute logits: beta_c * (r_a - r_b)
    r_a = r.view(N, 1, 1)
    r_b = r.view(1, N, 1)
    beta_c = beta.view(1, 1, K)

    logits = beta_c * (r_a - r_b)  # (N, N, K)
    probs = torch.sigmoid(logits)  # P(a > b | worker c)

    # Set probs[a == b] = 0 to exclude self-comparisons
    mask = torch.eye(N, device=r.device).unsqueeze(-1)  # (N, N, 1)
    probs = probs * (1 - mask)  # zero out diagonal

    # Normalize
    probs = probs / probs.sum()
    flat_probs = probs.flatten()
    cat = torch.distributions.Categorical(flat_probs)

    def sample_fn(n):
        """
        Samples n (a, b, c) comparisons where a ≠ b.

        Returns:
            torch.Tensor: (n, 3) with rows (winner_index, loser_index, worker_index)
        """
        samples = cat.sample((n,))
        # Convert flat index back to (winner, loser, worker) indices
        a = samples // (N * K)
        b = (samples % (N * K)) // K
        c = samples % K
        return torch.stack([a, b, c], dim=-1)

    return sample_fn, flat_probs