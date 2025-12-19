import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class GradientEM(nn.Module):
    def __init__(self, num_items, num_workers, random_seed=42):
        super().__init__()
        
        # Set seed for reproducibility
        torch.manual_seed(random_seed)
        # If using GPU, also set these:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        
        # 1. Initialize Item Rewards (r)
        self.item_rewards = nn.Embedding(num_items, 1)
        with torch.no_grad():
            # Fill with Standard Normal: mean=0, std=1
            torch.nn.init.normal_(self.item_rewards.weight, mean=0.0, std=1.0)
            # Zero-mean centering
            self.item_rewards.weight -= self.item_rewards.weight.mean()
        
        # 2. Initialize Worker Competencies (beta)
        self.worker_betas = nn.Embedding(num_workers, 1)
        with torch.no_grad():
            # Uniform distribution between -1.0 and 1.0
            torch.nn.init.uniform_(self.worker_betas.weight, a=-1.0, b=1.0)

    def forward(self, winners, losers, annotators):
        # Retrieve parameters for the batch
        r_w = self.item_rewards(winners)      # Shape: (batch_size, 1)
        r_l = self.item_rewards(losers)       # Shape: (batch_size, 1)
        beta_s = self.worker_betas(annotators) # Shape: (batch_size, 1)
        
        # Calculate the logit: beta_s * (r_w - r_l)
        logits = beta_s * (r_w - r_l)
        return logits.squeeze()

def train_step(model, data, optimizer_r, optimizer_beta):
    winners, losers, annotators = data
    
    # 1. Update Rewards (r)
    optimizer_r.zero_grad()
    logits = model(winners, losers, annotators)
    # Target is 1 because winners are listed first
    labels = torch.ones_like(logits)
    loss_r = F.binary_cross_entropy_with_logits(logits, labels)
    loss_r.backward()
    optimizer_r.step()
    
    # Projection: sum(r) = 0
    with torch.no_grad():
        mean_r = model.item_rewards.weight.mean()
        model.item_rewards.weight.sub_(mean_r)
    
    # 2. Update Competencies (beta)
    optimizer_beta.zero_grad()
    logits = model(winners, losers, annotators)
    loss_beta = F.binary_cross_entropy_with_logits(logits, labels)
    loss_beta.backward()
    optimizer_beta.step()
    
    # Constraint: beta in [-1, 1]
    with torch.no_grad():
        model.worker_betas.weight.clamp_(-1.0, 1.0)
        
    return loss_r.item()

def train_with_convergence(model, data, opt_r, opt_beta, tol=1e-6, max_epochs=1000):
    prev_loss = float('inf')
    
    for epoch in tqdm(range(max_epochs)):
        # Perform the alternating update step
        current_loss = train_step(model, data, opt_r, opt_beta)
        
        # Calculate the absolute difference in loss
        loss_diff = abs(prev_loss - current_loss)
        
#         if epoch % 20 == 0:
#             print(f"Epoch {epoch:03d} | Loss: {current_loss:.6f} | Diff: {loss_diff:.8f}")
        
        # Convergence Check
        if loss_diff < tol:
#             print(f"\nConverged at epoch {epoch} (Loss Diff < {tol})")
            break
            
        prev_loss = current_loss
    else:
        print("\nReached max_epochs without full convergence.")
        
    return model.item_rewards.weight.data.flatten(), model.worker_betas.weight.data.flatten()