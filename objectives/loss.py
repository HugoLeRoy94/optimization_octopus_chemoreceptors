import torch
import torch.nn as nn
import math

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
        # 1 compute the kernel density estimator (B,R)
        density = self._compute_kde(samples=activity,query_points=activity)
        # 3. Entropy H = - E[log p(x)]
        # We average log(density) over the batch dimension
        log_prob = torch.log(density + 1e-8) # Add epsilon for stability
        entropy = -torch.mean(log_prob, dim=0)
        
        return entropy
    def _compute_kde(self, samples: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        """
        Computes the Kernel Density Estimation of the samples, evaluated at query_points.
        
        Args:
            samples: (B_samples, R) - The data defining the centers of our Gaussian kernels.
            query_points: (B_query, R) - The points where we want to know the density.
            
        Returns:
            density: (B_query, R) - The estimated probability density at each query point.
        """
        B_samples, R = samples.shape
        
        # 1. Compute Bandwidth (h) based purely on the *samples*
        std = samples.std(dim=0) # (R,)
        h = self.bandwidth_factor * std * (B_samples ** (-0.2))
        h = torch.clamp(h, min=1e-4) # Safety clamp to prevent division by zero
        
        # 2. Broadcasting Setup
        # We need pairwise differences between every query_point and every sample.
        # X shape: (B_query, 1, R)
        X = query_points.unsqueeze(1)
        
        # Y shape: (1, B_samples, R)
        Y = samples.unsqueeze(0)
        
        # diff shape: (B_query, B_samples, R)
        diff = X - Y
        
        # 3. Apply Gaussian Kernel
        h_reshaped = h.view(1, 1, R)
        u = diff / h_reshaped
        
        # K(u) = (1 / sqrt(2*pi)) * exp(-0.5 * u^2)
        norm_factor = math.sqrt(2 * math.pi)
        kernel_values = torch.exp(-0.5 * u**2) / norm_factor
        
        # 4. Sum over the samples to get the final density
        # Average over dim 1 (the samples dimension)
        density = kernel_values.sum(dim=1) / (B_samples * h_reshaped.squeeze(0))
        
        return density
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