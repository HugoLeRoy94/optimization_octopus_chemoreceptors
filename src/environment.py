import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Tuple
from abc import ABC, abstractmethod

class ConcentrationModel(nn.Module, ABC):
    """
    Abstract Base Class for different concentration strategies.
    Subclass this to create LogNormal, Normal, Bimodal, etc.
    """
    @abstractmethod
    def sample(self, batch_size: int, family_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns concentrations for the given family_ids.
        Shape: (batch_size,)
        """
        pass

class LogNormalConcentration(ConcentrationModel):
    """
    Classic Biophysics assumption: c spans orders of magnitude.
    log10(c) ~ Normal(mu, sigma)
    """
    def __init__(self, n_families: int, init_mean=-6.0, init_scale=1.0):
        super().__init__()
        # Initialize around 10^-6 M (1 microM)
        self.mu = nn.Parameter(torch.ones(n_families) * init_mean)
        self.log_sigma = nn.Parameter(torch.ones(n_families) * torch.log(torch.tensor(init_scale)))

    def sample(self, batch_size, family_ids):
        # Gather params for this batch
        batch_mu = self.mu[family_ids]
        batch_sigma = torch.exp(self.log_sigma[family_ids])
        
        # Sample Log-Space
        dist_log = dist.Normal(batch_mu, batch_sigma)
        log_c = dist_log.rsample()
        
        # Convert to Real-Space
        return torch.pow(10.0, log_c)

class NormalConcentration(ConcentrationModel):
    """
    Simple Gaussian assumption.
    c ~ Normal(mu, sigma) clamped at 0.
    """
    def __init__(self, n_families: int, init_mean=5.0, init_scale=1.0):
        super().__init__()
        self.mu = nn.Parameter(torch.ones(n_families) * init_mean)
        self.log_sigma = nn.Parameter(torch.ones(n_families) * torch.log(torch.tensor(init_scale)))

    def sample(self, batch_size, family_ids):
        batch_mu = self.mu[family_ids]
        batch_sigma = torch.exp(self.log_sigma[family_ids])
        
        c = dist.Normal(batch_mu, batch_sigma).rsample()
        return torch.clamp(c, min=1e-6) # Physics constraint

class LigandEnvironment(nn.Module):
    def __init__(self, n_units: int, n_families: int, conc_model: ConcentrationModel):
        """
        Args:
            n_units: Number of protein units
            n_families: Number of ligand families
            conc_model: An INSTANCE of a ConcentrationModel subclass
        """
        super().__init__()
        self.n_units = n_units
        self.n_families = n_families
        
        # 1. Inject the Concentration Strategy
        self.concentration_model = conc_model
        
        # 2. Interaction Parameters (Standard MWC stuff)
        self.interaction_mu = nn.Parameter(torch.randn(n_units, n_families, 2) * 2.0)
        self.interaction_log_sigma = nn.Parameter(torch.randn(n_units, n_families, 2) - 1.0)

    def sample_batch(self, batch_size: int):
        device = self.interaction_mu.device
        
        # A. Sample Family IDs
        family_ids = torch.randint(0, self.n_families, (batch_size,), device=device)
        
        # B. Delegate Concentration Sampling
        concentrations = self.concentration_model.sample(batch_size, family_ids)
        
        # C. Sample Energies (Standard Reparameterization)
        # (Same logic as before...)
        mu_T = self.interaction_mu.permute(1, 0, 2)
        sigma_T = torch.exp(self.interaction_log_sigma.permute(1, 0, 2))
        
        batch_mus = mu_T[family_ids]
        batch_sigmas = sigma_T[family_ids]
        
        # energies[:,:,0] = E_open
        # energies[:,:,1] = E_close
        
        energies = dist.Normal(batch_mus, batch_sigmas).rsample()
        
        return energies, concentrations, family_ids