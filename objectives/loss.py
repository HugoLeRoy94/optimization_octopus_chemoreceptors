import torch
import torch.nn as nn

class InformationLoss(nn.Module):
    """
    Computes the Information Theoretic loss to maximize array diversity.
    
    Loss = - (Sum of Single Receptor Entropies) + lambda * (Sum of Squared Covariances)
    
    Goal:
    1. Make every receptor have a high entropy (broad, diverse response).
    2. Make every pair of receptors uncorrelated (they encode different things).
    """
    def __init__(self, cov_weight: float = 1.0, bandwidth_factor: float = 1.06):
        """
        Args:
            cov_weight: The lambda parameter weighting the decorrelation term.
            bandwidth_factor: The multiplier for Silverman's rule (default 1.06).
        """
        super().__init__()
        self.cov_weight = cov_weight
        self.bandwidth_factor = bandwidth_factor

    def _compute_kde_entropy(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Estimates the differential entropy of each receptor using Gaussian KDE.
        
        Args:
            activity: (Batch, N_Receptors)
            
        Returns:
            entropies: (N_Receptors,) - The estimated entropy for each receptor.
        """
        # Shapes
        B, R = activity.shape
        
        # 1. Compute Bandwidth (h) per receptor using Silverman's Rule
        # h = 1.06 * sigma * B^(-1/5)
        std = activity.std(dim=0) # (R,)
        
        # Stability: If a receptor is dead (std=0), fix h to a small value to avoid NaN
        h = self.bandwidth_factor * std * (B ** (-0.2))
        h = torch.clamp(h, min=1e-4) # Avoid division by zero
        
        # 2. Vectorized KDE
        # We need to calculate the distance between every sample i and every sample j
        # for a specific receptor r.
        
        # Expand dims for broadcasting:
        # X: (B, 1, R)
        # Y: (1, B, R)
        X = activity.unsqueeze(1)
        Y = activity.unsqueeze(0)
        
        # Differences: (B, B, R)
        diff = X - Y
        
        # Gaussian Kernel: K(u) = (1/sqrt(2pi)) * exp(-0.5 * u^2)
        # u = diff / h
        # We perform this for all R receptors in parallel
        
        # h_reshaped: (1, 1, R)
        h_reshaped = h.view(1, 1, R)
        
        u = diff / h_reshaped
        kernel_values = torch.exp(-0.5 * u**2) / (2.506628) # 2.506... is sqrt(2*pi)
        
        # Density Estimate p(x): Sum over the neighbor batch dimension (dim 1)
        # p(x_i) = (1 / (B*h)) * Sum_j K(u_ij)
        # Sum over j (dim 1) -> Result (B, R)
        density = kernel_values.sum(dim=1) / (B * h_reshaped.squeeze())
        
        # 3. Entropy H = - E[log p(x)]
        # We average log(density) over the batch dimension
        log_prob = torch.log(density + 1e-8) # Add epsilon for stability
        entropy = -torch.mean(log_prob, dim=0)
        
        return entropy

    def _compute_covariance_penalty(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Computes the sum of squared off-diagonal elements of the covariance matrix.
        We want this to be zero (uncorrelated receptors).
        """
        B, R = activity.shape
        
        # 1. Center the data
        mean = activity.mean(dim=0, keepdim=True)
        centered = activity - mean
        
        # 2. Compute Covariance Matrix: (R, R)
        # C = (X^T @ X) / (B - 1)
        cov_matrix = (centered.T @ centered) / (B - 1)
        
        # 3. Remove Diagonal (Variance)
        # We only want to penalize correlations between *different* receptors.
        # We do not want to minimize variance (in fact, we want high variance for high entropy).
        mask = ~torch.eye(R, dtype=torch.bool, device=activity.device)
        off_diagonals = cov_matrix[mask]
        
        # 4. Loss = Sum of Squares
        return (off_diagonals ** 2).sum()

    def forward(self, activity: torch.Tensor):
        """
        Args:
            activity: (Batch, N_Receptors) - The normalized response of the array.
            
        Returns:
            loss: Scalar tensor to minimize.
            stats: Dictionary containing 'entropy' and 'covariance' values for logging.
        """
        # A. Maximize Entropy -> Minimize Negative Entropy
        entropies = self._compute_kde_entropy(activity)
        mean_entropy = entropies.mean()
        loss_entropy = -mean_entropy
        
        # B. Minimize Covariance
        loss_covariance = self._compute_covariance_penalty(activity)
        
        # Total Loss
        total_loss = loss_entropy + (self.cov_weight * loss_covariance)
        
        return total_loss, {
            "loss": total_loss.detach(),
            "entropy": mean_entropy.detach(),
            "covariance": loss_covariance.detach()
        }