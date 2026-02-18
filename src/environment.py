import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Tuple

class LigandEnvironment(nn.Module):
    """
    The Environment Module.
    
    Responsibilities:
    1. Holds the learnable parameters (M and Sigma) for Unit-Ligand interactions.
    2. Holds the learnable parameters for Ligand Concentrations.
    3. Implements the 'Reparameterization Trick' to allow gradients to flow
       through random sampling.
    """
    def __init__(self, n_units: int, n_families: int):
        super().__init__()
        self.n_units = n_units
        self.n_families = n_families
        
        # ======================================================================
        # 1. INTERACTION PARAMETERS (The Physics)
        # ======================================================================
        # We store Mu and Sigma for Open (0) and Closed (1) states.
        # Shape: (n_units, n_families, 2)
        
        # Initialize Mean Energies (Mu) randomly between -5 and +5 kT
        self.interaction_mu = nn.Parameter(
            torch.randn(n_units, n_families, 2) * 2.0
        )
        
        # Initialize Std Devs (Sigma). 
        # We store 'log_sigma' to ensure sigma is always positive when we exponentiate it.
        # This is a common numerical stability trick in ML.
        self.interaction_log_sigma = nn.Parameter(
            torch.randn(n_units, n_families, 2) - 1.0 # Start with small noise
        )

        # ======================================================================
        # 2. CONCENTRATION PARAMETERS (The Ecology)
        # ======================================================================
        # We assume Log-Normal distribution for concentrations.
        # log10(c) ~ Normal(mu_c, sigma_c)
        
        self.conc_mu = nn.Parameter(
            torch.zeros(n_families).uniform_(-9.0, -3.0) # 1nM to 1mM
        )
        self.conc_log_sigma = nn.Parameter(
             torch.zeros(n_families).uniform_(-1.0, 0.0) # Small spread
        )

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The Forward Pass for the Environment.
        
        Returns:
            energies: (batch_size, n_units, 2) - Sampled Interaction Energies
            concentrations: (batch_size,)      - Sampled Ligand Concentrations
            family_ids: (batch_size,)          - The identity of the ligand family
        """
        device = self.interaction_mu.device
        
        # ----------------------------------------------------------------------
        # A. Sample Ligand Identities
        # ----------------------------------------------------------------------
        # Randomly select which families appear in this batch
        family_ids = torch.randint(0, self.n_families, (batch_size,), device=device)
        
        # ----------------------------------------------------------------------
        # B. Sample Concentrations (Reparameterization Trick 1)
        # ----------------------------------------------------------------------
        # 1. Get parameters for the chosen families
        # shape: (batch_size,)
        batch_mu_c = self.conc_mu[family_ids]
        batch_sigma_c = torch.exp(self.conc_log_sigma[family_ids])
        
        # 2. Create a Normal distribution and sample with gradients enabled (rsample)
        # rsample() automatically does: mu + sigma * epsilon
        conc_dist = dist.Normal(batch_mu_c, batch_sigma_c)
        log_conc = conc_dist.rsample() 
        
        concentrations = torch.pow(10.0, log_conc)
        
        # ----------------------------------------------------------------------
        # C. Sample Interaction Energies (Reparameterization Trick 2)
        # ----------------------------------------------------------------------
        # 1. Extract params for the chosen families
        # We need shape: (batch_size, n_units, 2)
        # self.interaction_mu shape is (n_units, n_families, 2)
        # We transpose to (n_families, n_units, 2) to index easily
        
        mu_transposed = self.interaction_mu.permute(1, 0, 2)
        sigma_transposed = torch.exp(self.interaction_log_sigma.permute(1, 0, 2))
        
        # Select specific families
        batch_mus = mu_transposed[family_ids]       # (batch, units, 2)
        batch_sigmas = sigma_transposed[family_ids] # (batch, units, 2)
        
        # 2. Sample Energies
        energy_dist = dist.Normal(batch_mus, batch_sigmas)
        energies = energy_dist.rsample()
        
        # Return format: 
        # energies: (batch, units, 2)
        # conc: (batch,)
        return energies, concentrations, family_ids

    def get_covariance_loss_term(self):
        """
        Optional helper: You might want to regularize Sigma later 
        to prevent it from collapsing to zero or exploding.
        """
        return torch.exp(self.interaction_log_sigma).mean()