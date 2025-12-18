import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy import linalg
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Helper functions
# def _safe_exp(x):
#     x_t = torch.as_tensor(x, dtype=torch.float32)  # ensures torch tensor
#     return torch.exp(torch.clamp(x_t, max=50)).detach().cpu().numpy()

def _safe_exp(x):
    """Safe exponential that preserves gradients"""
    return torch.exp(torch.clamp(x, max=50))

class CrowdBT:
    def __init__(self, data, device, penalty=0.0, dtype=torch.float32):
        """
        data: dict mapping reviewer i -> list of (winner, loser) pairs
        penalty: L2 regularization for item scores
        """
        self._data = data
        self._penalty = penalty
        self.device = device
        self.dtype = dtype

    def crowdbt_objective(self, params, rev_params):
        """
        Negative penalized log-likelihood
        params: torch tensor of item scores, shape (n_items,)
        rev_params: torch tensor of reviewer reliability, shape (n_reviewers,)
        """
        val = self._penalty * torch.sum(params ** 2)

        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(list(pairs), device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            pw = _safe_exp(params[win_idx])
            pl = _safe_exp(params[los_idx])
            denom = pw + pl

            prob = rev_params[i] * pw / denom + (1 - rev_params[i]) * pl / denom
            val += -torch.sum(torch.log(prob))

        return val

    def crowdbt_gradient_scores(self, params, rev_params):
        grad = 2 * self._penalty * params.clone()
        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(pairs, device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            pw = _safe_exp(params[win_idx])
            pl = _safe_exp(params[los_idx])
            denom = pw + pl

            prob = rev_params[i] * pw / denom + (1 - rev_params[i]) * pl / denom

            # z factor vectorized
            z = (pw * pl / denom**2) * (2 * rev_params[i] - 1) / prob

            # accumulate gradient
            grad.index_add_(0, win_idx, -z)
            grad.index_add_(0, los_idx, z)

        return grad

    def crowdbt_gradient_revs(self, params, rev_params):
        grad = torch.zeros_like(rev_params, device=self.device, dtype=self.dtype)
        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(pairs, device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            pw = _safe_exp(params[win_idx])
            pl = _safe_exp(params[los_idx])
            denom = pw + pl

            prob = rev_params[i] * pw / denom + (1 - rev_params[i]) * pl / denom
            z = -(pw - pl) / denom / prob

            grad[i] += z.sum()

        return grad
    
    def alternate_optim(self, size, num_reviewers, iters=100, tol=1e-5, gtol=1e-5, lr_x=0.01, lr_y=0.01):
        """
        Alternating optimization for CrowdBT
        Returns:
            x0: item scores (torch tensor)
            y0: reviewer reliabilities (torch tensor)
        """
        device = self.device
        dtype = self.dtype

        # Initialize parameters
        x0 = torch.zeros(size, device=device, dtype=dtype, requires_grad=True)
        y0 = torch.full((num_reviewers,), 0.7, device=device, dtype=dtype, requires_grad=True)
        
        print("init done")

        for _ in tqdm(range(iters)):
            # --- Update x (item scores) with y fixed ---
            x0.requires_grad_(True)
            optimizer_x = torch.optim.SGD([x0], lr=lr_x)
            optimizer_x.zero_grad()
            loss_x = self.crowdbt_objective(x0, y0.detach())
            loss_x.backward()
            optimizer_x.step()

            # --- Update y (reviewer reliabilities) with x fixed ---
            y0.requires_grad_(True)
            optimizer_y = torch.optim.SGD([y0], lr=lr_y)
            optimizer_y.zero_grad()
            loss_y = self.crowdbt_objective(x0.detach(), y0)
            loss_y.backward()
            optimizer_y.step()
            y0.data.clamp_(0.0, 1.0)  # enforce bounds [0,1]

#             print(x0, y0)

            # Check convergence
            if loss_x.item() < tol and loss_y.item() < gtol:
                break

        return x0.detach(), y0.detach()


import torch
import random
import numpy as np
from tqdm import tqdm # Assuming tqdm is available for the loop

class FactorBT:
    def __init__(self, data, classes, device=None, penalty=0.0, dtype=torch.float32, seed=42):
        """
        data: dict mapping reviewer i -> list of (winner, loser) pairs
        classes: list or tensor of item classes
        device: 'cpu' or 'cuda' device
        penalty: L2 regularization for item scores
        dtype: torch data type
        seed: Random seed for initialization
        """
        self._data = data
        self._classes = torch.tensor(classes, device=device if device else 'cpu')
        self._penalty = penalty
        self.device = device if device else 'cpu'
        self.dtype = dtype
        self.seed = seed
        
        # Set the random seed upon initialization
        self.set_seed(self.seed)

    def set_seed(self, seed):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            # For deterministic behavior on CUDA
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def FactorBT_objective(self, params, rev_params_g, rev_params_r):
        # ... (objective function remains the same)
        val = self._penalty * torch.sum(params ** 2)

        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(list(pairs), device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            temp = torch.sign(self._classes[win_idx] - self._classes[los_idx]).to(self.dtype)

            pw = torch.sigmoid(params[win_idx] - params[los_idx])
            pr = torch.sigmoid(rev_params_r[i] * temp)

            prob = rev_params_g[i] * pw + (1 - rev_params_g[i]) * pr
            val += -torch.sum(torch.log(prob + 1e-12))  # small epsilon for numerical stability

        return val

    def FactorBT_gradient_scores(self, params, rev_params_g, rev_params_r):
        # ... (gradient calculation remains the same)
        grad = 2 * self._penalty * params.clone()
        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(pairs, device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            temp = torch.sign(self._classes[win_idx] - self._classes[los_idx]).to(self.dtype)

            pw = torch.sigmoid(params[win_idx] - params[los_idx])
            pr = torch.sigmoid(rev_params_r[i] * temp)
            prob = rev_params_g[i] * pw + (1 - rev_params_g[i]) * pr

            z = rev_params_g[i] * pw * (1 - pw) / (prob + 1e-12)
            grad.index_add_(0, win_idx, -z)
            grad.index_add_(0, los_idx, z)

        return grad

    def FactorBT_gradient_g(self, params, rev_params_g, rev_params_r):
        # ... (gradient calculation remains the same)
        grad = torch.zeros_like(rev_params_g, device=self.device, dtype=self.dtype)
        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(pairs, device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            temp = torch.sign(self._classes[win_idx] - self._classes[los_idx]).to(self.dtype)

            pw = torch.sigmoid(params[win_idx] - params[los_idx])
            pr = torch.sigmoid(rev_params_r[i] * temp)
            prob = rev_params_g[i] * pw + (1 - rev_params_g[i]) * pr

            # Note: The original gradient for g was incorrect as sigmoid(-g) is not the derivative
            # of g, but we rely on PyTorch's autograd for the actual optimization steps.
            # We keep the original (though potentially analytically flawed) FactorBT_gradient_g for completeness,
            # but it is NOT used in alternate_optim.
            z = -(pw - pr) * torch.sigmoid(-rev_params_g[i]) / (prob + 1e-12)
            grad[i] += z.sum()

        return grad

    def FactorBT_gradient_r(self, params, rev_params_g, rev_params_r):
        # ... (gradient calculation remains the same)
        grad = torch.zeros_like(rev_params_r, device=self.device, dtype=self.dtype)
        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(pairs, device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            temp = torch.sign(self._classes[win_idx] - self._classes[los_idx]).to(self.dtype)

            pr = torch.sigmoid(rev_params_r[i] * temp)
            pw = torch.sigmoid(params[win_idx] - params[los_idx])
            prob = rev_params_g[i] * pw + (1 - rev_params_g[i]) * pr

            z = (1 - rev_params_g[i]) * pr * (1 - pr) * temp / (prob + 1e-12)
            grad[i] += -z.sum()

        return grad
       
    def alternate_optim(self, iters=100, tol=1e-5, gtol=1e-5, lr_x=0.01, lr_g=0.01, lr_r=0.01):
        """
        Alternating optimization for FactorBT using PyTorch with random seed initialization.
        """
        size = len(self._classes)
        num_reviewers = len(self._data)
        device = self.device
        dtype = self.dtype

        # --- Initialize parameters using random seeds ---
        # x0 (item scores): Initialized using a small normal distribution (e.g., Xavier/Kaiming, but small normal is common for scores)
        x0 = torch.randn(size, device=device, dtype=dtype) * 0.01
        x0.requires_grad_(True)
        
        # y0 (reviewer g biases): Initialized around the starting point of 0.7, but with noise
        y0 = torch.full((num_reviewers,), 0.7, device=device, dtype=dtype) + torch.randn(num_reviewers, device=device, dtype=dtype) * 0.01
        y0.requires_grad_(True)
        
        # z0 (reviewer r biases): Initialized using a small normal distribution
        z0 = torch.randn(num_reviewers, device=device, dtype=dtype) * 0.01
        z0.requires_grad_(True)

        for it in tqdm(range(iters)):
            # --- Update y (g biases) with x0, z0 fixed ---
            # y0.requires_grad_(True) is already set, but we make sure the others are detached
            optimizer_y = torch.optim.SGD([y0], lr=lr_g)
            optimizer_y.zero_grad()
            # Use autograd for optimization
            loss_y = self.FactorBT_objective(x0.detach(), y0, z0.detach())
            loss_y.backward()
            optimizer_y.step()
            y0.data.clamp_(0.0, 1.0)  # bounds [0,1]

            # --- Update z (r biases) with x0, y0 fixed ---
            # z0.requires_grad_(True) is already set
            optimizer_z = torch.optim.SGD([z0], lr=lr_r)
            optimizer_z.zero_grad()
            loss_z = self.FactorBT_objective(x0.detach(), y0.detach(), z0)
            loss_z.backward()
            optimizer_z.step()

            # --- Update x (item scores) with y0, z0 fixed ---
            # x0.requires_grad_(True) is already set
            optimizer_x = torch.optim.SGD([x0], lr=lr_x)
            optimizer_x.zero_grad()
            loss_x = self.FactorBT_objective(x0, y0.detach(), z0.detach())
            loss_x.backward()
            optimizer_x.step()

            # Check convergence (using detached loss values)
            if loss_x.item() < tol and loss_y.item() < gtol:
                break

        return x0.detach(), y0.detach(), z0.detach()

class BARP:
    def __init__(self, data, penalty, classes, device, dtype=torch.float32):
        self._data = data
        self.device = device
        self.dtype = dtype
#         print(self.device)
        self._classes = torch.tensor(classes, device=self.device, dtype=self.dtype)
        self._penalty = penalty
        
    def objective(self, params, rev_params):
        val = self._penalty * torch.sum(params ** 2)

#         print(f"=== DEBUG OBJECTIVE ===")  # Debug line
#         print(f"params shape: {params.shape}, rev_params shape: {rev_params.shape}")
#         print(f"Number of reviewers (i keys): {len(self._data)}")
#         print(f"rev_params length: {len(rev_params)}")

        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(list(pairs), device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            # compute delta = (params[win] + rev_params[i]*class[win]) - (params[los] + rev_params[i]*class[los])
            delta = (params[win_idx] + rev_params[i] * self._classes[win_idx] -
                     params[los_idx] - rev_params[i] * self._classes[los_idx])

            val += torch.sum(F.softplus(-delta))  # softplus(x) = log(1 + exp(x)) = logaddexp(0, x)
        return val

    def gradient_scores(self, params, rev_params):
        grad = 2 * self._penalty * params.clone()
        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(pairs, device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            delta = (params[win_idx] + rev_params[i] * self._classes[win_idx] -
                     params[los_idx] - rev_params[i] * self._classes[los_idx])
            z = 1 / (1 + _safe_exp(delta))

            grad.index_add_(0, win_idx, -z)
            grad.index_add_(0, los_idx, z)
        return grad

    def gradient_revs(self, params, rev_params):
        grad = torch.zeros_like(rev_params, device=self.device, dtype=self.dtype)
        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(pairs, device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            delta = (params[win_idx] + rev_params[i] * self._classes[win_idx] -
                     params[los_idx] - rev_params[i] * self._classes[los_idx])
            z = (self._classes[los_idx] - self._classes[win_idx]) / (1 + _safe_exp(delta))
            grad[i] += z.sum()
        return grad



class RankCentrality:
    def __init__(self, device):
        self.device = device
    
    def matrix_of_comparisons(self, size, comparisons, reg=1.0, dtype=torch.float32):
        """
        PyTorch version of _matrix_of_comparisons.
        Inputs:
          - size: int, number of items
          - comparisons: iterable of (i, j) pairs where the FIRST element of the pair
                         is the one preferred (same semantic as your original code).
                         NOTE: original code did `A[j,i] += 1` for pair (i,j). We preserve that.
          - reg: regularizer scalar (float)
          - device: 'cuda' or 'cpu' or torch.device
          - dtype: torch dtype
        Output:
          - B: size x size torch tensor, where B_ij = fraction of times object j was preferred to object i
               plus reg*(1 - I).
        """
        A = torch.zeros((size, size), dtype=dtype, device=self.device)

        # accumulate counts. preserve original semantics: A[j,i] += 1 for pair (i,j)
        # comparisons might be list of tuples or a 2D tensor
        if isinstance(comparisons, torch.Tensor):
            if comparisons.numel() > 0:
                idx_i = comparisons[:, 0].long().to(self.device)
                idx_j = comparisons[:, 1].long().to(self.device)
                # use scatter_add
                A.index_put_((idx_j, idx_i), torch.ones_like(idx_i, dtype=dtype), accumulate=True)
        else:
            # for python list of tuples we vectorize by converting to tensor (if not huge),
            # else simple loop (loop is fine for small lists)
            try:
                comp_t = torch.tensor(comparisons, dtype=torch.long, device=self.device)
                if comp_t.numel() > 0:
                    idx_i = comp_t[:, 0]
                    idx_j = comp_t[:, 1]
                    A.index_put_((idx_j, idx_i), torch.ones_like(idx_i, dtype=dtype), accumulate=True)
            except Exception:
                # fallback: incremental
                for (i, j) in comparisons:
                    A[j, i] += 1.0

        # compute pairwise ratio: B[i,j] = A[i,j] / (A[i,j] + A[j,i]) when denom>0
        denom = A + A.t()
        B = torch.zeros_like(A)
        mask = denom > 0
        B[mask] = A[mask] / denom[mask]

        # add reg on off-diagonal as in original
        I = torch.eye(size, dtype=dtype, device=self.device)
        B = B + reg * (torch.ones_like(B) - I)

        return B


    def trans_prob(self, A):
        """
        PyTorch version of _trans_prob.
        Input A should be the matrix returned by matrix_of_comparisons (float tensor).
        Produces row-stochastic matrix P: rows sum to 1 (adds self-loop to account for remaining mass).
        """
        # assume A is torch tensor
        device = A.device
        dtype = A.dtype
        n = A.shape[0]

        # maximum out-degree: max number of nonzero entries per row (count_nonzero)
        counts = torch.count_nonzero(A, dim=1).to(dtype)
        d_max = counts.max().item()
        if d_max == 0:
            # no edges at all: return identity (stationary = uniform)
            return torch.eye(n, dtype=dtype, device=self.device)

        P = A / float(d_max)  # scale
        sum_by_row = P.sum(dim=1)

        # set diagonal to remaining probability
        # P[i,i] = 1 - sum_by_row[i]  -> use diag assignment
        diag_values = (1.0 - sum_by_row).to(dtype)
        P = P.clone()
        P.view(-1)[::n+1] = diag_values  # faster diagonal set

        # numerical guard: ensure row sums are exactly 1 (small numerical issues)
        row_sums = P.sum(dim=1, keepdim=True)
        P = P / row_sums

        return P


    def stationary_dist(self, P, tol=1e-9, max_iter=10000, dtype=None):
        """
        Compute stationary distribution of row-stochastic matrix P using power iteration.
        Returns a 1D tensor of length n (row vector) representing stationary distribution pi
        that satisfies pi = pi @ P.
        Uses GPU if P is on cuda.
        """
        if dtype is None:
            dtype = P.dtype

        n = P.shape[0]
        # start with uniform row vector
        pi = torch.full((1, n), 1.0 / n, dtype=dtype, device=self.device)  # shape (1,n)

        # Power iteration: iterate pi <- pi @ P
        for k in range(max_iter):
            pi_next = pi @ P
            diff = torch.sum(torch.abs(pi_next - pi))
            pi = pi_next
            if diff.item() < tol:
                break

        # return as 1D tensor
        pi = pi.view(-1)
        # ensure non-negative and normalized (numerical guard)
        pi = torch.clamp(pi, min=0.0)
        s = pi.sum()
        if s <= 0:
            # fallback to uniform
            pi = torch.full((n,), 1.0 / n, dtype=dtype, device=self.device)
        else:
            pi = pi / s
        return pi



################# OLD #################
def _sample_pairs(scores, n_pairs ):
    pairs = []
    numbers = np.arange(len(scores))
    for i in range(n_pairs):
        
        a, b = np.random.choice(numbers, size=2, replace=False) #replace = False to ensure a != b
        #while (a, b) in pairs or (b, a) in pairs: #sample the pair, in this version the reviewer can evaluate the same pair only once
            #a, b = np.random.choice(numbers, size=2, replace=False)

        #make them play
        if np.random.rand() < (np.exp(scores[a])/(np.exp(scores[a])+np.exp(scores[b]))):
        #if scores[a]>scores[b]: #deterministic version
        # this block of code will be executed with probability p
            pairs.append((a, b)) #who win is the first of the pair!!! i.e. a won
        else:
        # this block of code will be executed with probability 1-p   
            pairs.append((b, a)) #b won
    return(pairs)

def _create_matrix_biased_scores(original,rev_bias,classes):
    '''this matrix represents how much bias each reviewer has
    original: the original scores for the items
    rev_bias: the vector with the reviewers' biases
    classes: the items' classes 
    return:
    biases_scores: the matrix with the scores as 'seen' by each reviewer'''
    #matrix of biased scores, each reviewer correspond to a column
    biased_scores = np.zeros((len(original),len(rev_bias)))
    for col,bias in enumerate(rev_bias):
        for row,value in enumerate(classes):
            if value == 1:
                biased_scores[row,col] = original[row] + bias #add bias to reviewers ranking
                                                                                    
            elif value == 0:
                biased_scores[row,col] = original[row] 
    #biased_scores[biased_scores <= 0] = 0.00001
    return biased_scores

def _create_pc_set_for_reviewers(biased_scores,pair_per_reviewer):
    revs_set = {}
    for i in range(np.shape(biased_scores)[1]):
        revs_set.update({i:_sample_pairs(biased_scores[:,i], n_pairs = pair_per_reviewer )})
        
    return revs_set

def create_pc_set_for_reviewers_custom(biased_scores,pair_per_reviewer):
    revs_set = {}
    for i in range(np.shape(biased_scores)[1]):
        revs_set.update({i:_sample_pairs(biased_scores[:,i], n_pairs = pair_per_reviewer[i] )})
        
    return revs_set

def _pc_without_reviewers(revs_set):
    ''' input: the set of pc for each reviewer
        output: pc without the reviewer info'''
    return [[val1, val2] for sublist in revs_set.values() for val1, val2 in sublist]


def _alternate_optim(size, num_reviewers, pc_with_revs, iters = 101, tol = 1e-5, gtol = 1e-5):
    '''x0 is the estimated scores
       y0 is the estimated bias for each reviewer'''
    x0 = np.zeros(size)
    y0 = np.zeros(num_reviewers)
    for i in range(iters):
        # minimize with x fixed and update y
        res_y = minimize(lambda y: pc_with_revs.objective(x0, y), y0,tol = tol, jac=lambda y: pc_with_revs.gradient_revs(x0, y), options={"gtol": gtol,'maxiter': 1})
        y0 = res_y.x

        # minimize with y fixed and update x
        res_x = minimize(lambda x: pc_with_revs.objective(x, y0), x0,tol = tol, jac=lambda x: pc_with_revs.gradient_scores(x, y0), options={"gtol": gtol,'maxiter': 1})
        x0 = res_x.x

        if res_x.success and res_y.success:
            break
    return x0,y0

def _alternate_optim_torch(size, num_reviewers, pc_with_revs, iters=100, lr_x=1e-2, lr_y=1e-2, device="cuda"):
    """
    Torch GPU-based alternate optimization
    x0: estimated scores
    y0: estimated bias for each reviewer
    """
    # Initialize on device
    x0 = torch.zeros(size, device=device, requires_grad=True)
    y0 = torch.zeros(num_reviewers, device=device, requires_grad=True)

    # Separate optimizers for x and y
    optimizer_x = torch.optim.SGD([x0], lr=lr_x)
    optimizer_y = torch.optim.SGD([y0], lr=lr_y)

    for i in tqdm(range(iters)):
        # --- Optimize y with x fixed ---
        optimizer_y.zero_grad()
        loss_y = pc_with_revs.objective(x0.detach(), y0)  # detach x0 so y updates only
        loss_y.backward()
        optimizer_y.step()

        # --- Optimize x with y fixed ---
        optimizer_x.zero_grad()
        loss_x = pc_with_revs.objective(x0, y0.detach())  # detach y0 so x updates only
        loss_x.backward()
        optimizer_x.step()

        # Optional: stopping criterion (tiny gradient norm)
        if torch.norm(x0.grad if x0.grad is not None else torch.zeros_like(x0)) < 1e-5 and \
           torch.norm(y0.grad if y0.grad is not None else torch.zeros_like(y0)) < 1e-5:
            break

    return x0.detach(), y0.detach()



class CrowdBT_2_0:
    def __init__(self, data, device, random_seed = 42, penalty=0.0, dtype=torch.float32):
        """
        data: dict mapping reviewer i -> list of (winner, loser) pairs
        penalty: L2 regularization for item scores
        """
        self._data = data
        self._penalty = penalty
        self.device = device
        self.dtype = dtype
        self.random_seed = random_seed

    def crowdbt_objective(self, params, rev_params):
        """
        Negative penalized log-likelihood
        params: torch tensor of item scores, shape (n_items,)
        rev_params: torch tensor of reviewer reliability, shape (n_reviewers,)
        """
        val = self._penalty * torch.sum(params ** 2)

        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
#             pairs = list(pairs)  # convert set to list
            pairs = torch.tensor(list(pairs), device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            pw = _safe_exp(params[win_idx])
            pl = _safe_exp(params[los_idx])
            denom = pw + pl

#             Convert pw, pl, denom to tensors first
#             pw_tensor = torch.tensor(pw, device=rev_params[i].device)
#             pl_tensor = torch.tensor(pl, device=rev_params[i].device)
#             denom_tensor = torch.tensor(denom, device=rev_params[i].device)
# #             print(pw, pw_tensor, pl, pl_tensor)

#             prob = (rev_params[i] * pw_tensor / denom_tensor + (1 - rev_params[i]) * pl_tensor / denom_tensor)
#             prob = (rev_params[i].cpu().numpy() * pw / denom + (1 - rev_params[i].cpu().numpy()) * pl / denom)
            prob = rev_params[i] * pw / denom + (1 - rev_params[i]) * pl / denom
            val += -torch.sum(torch.log(prob))
#             print(val)

        return val


# for i, pairs in self._data.items():
#             if len(pairs) == 0:
#                 continue
#             pairs = torch.tensor(list(pairs), device=self.device, dtype=torch.long)
#             win_idx = pairs[:, 0]
#             los_idx = pairs[:, 1]

#             temp = torch.sign(self._classes[win_idx] - self._classes[los_idx]).to(self.dtype)

#             pw = torch.sigmoid(params[win_idx] - params[los_idx])
#             pr = torch.sigmoid(rev_params_r[i] * temp)

#             prob = rev_params_g[i] * pw + (1 - rev_params_g[i]) * pr
#             val += -torch.sum(torch.log(prob + 1e-12))  # small epsilon for numerical stability


    def crowdbt_gradient_scores(self, params, rev_params):
        grad = 2 * self._penalty * params.clone()
        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(pairs, device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            pw = _safe_exp(params[win_idx])
            pl = _safe_exp(params[los_idx])
            denom = pw + pl

            prob = rev_params[i] * pw / denom + (1 - rev_params[i]) * pl / denom

            # z factor vectorized
            z = (pw * pl / denom**2) * (2 * rev_params[i] - 1) / prob

            # accumulate gradient
            grad.index_add_(0, win_idx, -z)
            grad.index_add_(0, los_idx, z)

        return grad

    def crowdbt_gradient_revs(self, params, rev_params):
        grad = torch.zeros_like(rev_params, device=self.device, dtype=self.dtype)
        for i, pairs in self._data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(pairs, device=self.device, dtype=torch.long)
            win_idx = pairs[:, 0]
            los_idx = pairs[:, 1]

            pw = _safe_exp(params[win_idx])
            pl = _safe_exp(params[los_idx])
            denom = pw + pl

            prob = rev_params[i] * pw / denom + (1 - rev_params[i]) * pl / denom
            z = -(pw - pl) / denom / prob

            grad[i] += z.sum()

        return grad
    
    def alternate_optim(self, size, num_reviewers, iters=100, tol=1e-5, gtol=1e-5, lr_x=0.01, lr_y=0.01):
        """
        Alternating optimization for CrowdBT
        Returns:
            x0: item scores (torch tensor)
            y0: reviewer reliabilities (torch tensor)
        """
        device = self.device
        dtype = self.dtype

        # Initialize parameters
        x0 = torch.zeros(size, device=device, dtype=dtype, requires_grad=True)

#         y0 = torch.full((num_reviewers,), 0.7, device=device, dtype=dtype, requires_grad=True)
        # Assuming self.random is an integer seed
        gen = torch.Generator(device=device).manual_seed(self.random_seed)

        y0 = torch.rand((num_reviewers,), 
                        generator=gen, 
                        device=device, 
                        dtype=dtype, 
                        requires_grad=True)
        
        print("init done")

        for _ in tqdm(range(iters)):
            # --- Update x (item scores) with y fixed ---
            x0.requires_grad_(True)
            optimizer_x = torch.optim.SGD([x0], lr=lr_x)
            optimizer_x.zero_grad()
            loss_x = self.crowdbt_objective(x0, y0.detach())
            loss_x.backward()
            optimizer_x.step()

            # --- Update y (reviewer reliabilities) with x fixed ---
            y0.requires_grad_(True)
            optimizer_y = torch.optim.SGD([y0], lr=lr_y)
            optimizer_y.zero_grad()
            loss_y = self.crowdbt_objective(x0.detach(), y0)
            loss_y.backward()
            optimizer_y.step()
            y0.data.clamp_(0.0, 1.0)  # enforce bounds [0,1]

#             print(x0, y0)

            # Check convergence
            if loss_x.item() < tol and loss_y.item() < gtol:
                break

        return x0.detach(), y0.detach()


from tqdm import tqdm


class CrowdBT_3_0:
    """
    Fast, vectorized CrowdBT implementation
    """

    def __init__(
        self,
        data,
        device,
        random_seed=42,
        penalty=0.0,
        dtype=torch.float32,
        clamp_scores=20.0,
    ):
        """
        data: dict mapping reviewer i -> iterable of (winner, loser)
        """
        self.device = device
        self.dtype = dtype
        self.penalty = penalty
        self.random_seed = random_seed
        self.clamp_scores = clamp_scores

        # -------- Preprocess data once --------
        win_all = []
        los_all = []
        rev_all = []

        for i, pairs in data.items():
            if len(pairs) == 0:
                continue
            pairs = torch.tensor(
                list(pairs), device=device, dtype=torch.long
            )
            n = pairs.shape[0]
            win_all.append(pairs[:, 0])
            los_all.append(pairs[:, 1])
            rev_all.append(torch.full((n,), i, device=device))

        self.win_idx = torch.cat(win_all)
        self.los_idx = torch.cat(los_all)
        self.rev_idx = torch.cat(rev_all)

        self.num_pairs = self.win_idx.numel()

    # ----------------------------------------------------
    # Objective (fully vectorized)
    # ----------------------------------------------------
    def crowdbt_objective(self, scores, reliabilities):
        """
        Negative penalized log-likelihood
        """
        # Optional clamp for numerical stability
        scores = torch.clamp(scores, -self.clamp_scores, self.clamp_scores)

        pw = torch.exp(scores[self.win_idx])
        pl = torch.exp(scores[self.los_idx])
        denom = pw + pl

        r = reliabilities[self.rev_idx]
        prob = r * pw / denom + (1.0 - r) * pl / denom

        loss = -torch.sum(torch.log(prob + 1e-12))

        if self.penalty > 0:
            loss = loss + self.penalty * torch.sum(scores ** 2)

        return loss

    # ----------------------------------------------------
    # Alternating optimization
    # ----------------------------------------------------
    def alternate_optim(
        self,
        num_items,
        num_reviewers,
        iters=100,
        lr_x=0.05,
        lr_y=0.05,
        tol=1e-6,
        verbose=True,
    ):
        """
        Returns:
            scores: (num_items,)
            reliabilities: (num_reviewers,)
        """
        torch.manual_seed(self.random_seed)

        # Initialize parameters
        scores = torch.zeros(
            num_items,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        reliabilities = torch.rand(
            num_reviewers,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        # Optimizers (created ONCE)
        opt_x = torch.optim.Adam([scores], lr=lr_x)
        opt_y = torch.optim.Adam([reliabilities], lr=lr_y)

        prev_loss = None

        loop = tqdm(range(iters), disable=not verbose)
        for _ in loop:

            # ---- Update item scores ----
            opt_x.zero_grad()
            loss_x = self.crowdbt_objective(scores, reliabilities.detach())
            loss_x.backward()
            opt_x.step()

            # ---- Update reviewer reliabilities ----
            opt_y.zero_grad()
            loss_y = self.crowdbt_objective(scores.detach(), reliabilities)
            loss_y.backward()
            opt_y.step()

            # Enforce [0, 1] constraint
            reliabilities.data.clamp_(0.0, 1.0)

            loss_val = loss_x.item()
            loop.set_postfix(loss=loss_val)

            # Convergence check
            if prev_loss is not None and abs(prev_loss - loss_val) < tol:
                break
            prev_loss = loss_val

        return scores.detach(), reliabilities.detach()