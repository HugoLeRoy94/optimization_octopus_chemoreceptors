import math
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
    @abstractmethod
    def get_expected_log_c(self) -> torch.Tensor:
        """
        Returns the expected natural logarithm of the concentration for each family.
        Shape: (n_families,)
        """
        pass
    @abstractmethod
    def get_distribution(self, family_id: int) -> dist.Distribution:
        """Returns the torch distribution object for a specific family."""
        pass

class LogNormalConcentration(ConcentrationModel):
    """
    Classic Biophysics assumption: c spans orders of magnitude.
    log10(c) ~ Normal(mu, sigma)
    """
    def __init__(self, n_families: int, init_mean=-6.0, init_scale=1.0):
        super().__init__()
        # Initialize around 10^-6 M (1 microM)
        #self.mu = nn.Parameter(torch.ones(n_families) * init_mean)
        #self.log_sigma = nn.Parameter(torch.ones(n_families) * math.log(init_scale))
        self.register_buffer('mu', torch.ones(n_families) * init_mean)
        self.register_buffer('log_sigma', torch.ones(n_families) * math.log(init_scale))

    def sample(self, batch_size, family_ids):
        # Gather params for this batch
        batch_mu = self.mu[family_ids]
        batch_sigma = torch.exp(self.log_sigma[family_ids])
        
        # Sample Log-Space
        dist_log = dist.Normal(batch_mu, batch_sigma)
        log_c = dist_log.rsample()
        
        # Convert to Real-Space
        return torch.pow(10.0, log_c)
        
    def get_expected_log_c(self):
        # mu is log10(c). To get natural log ln(c): ln(c) = log10(c) * ln(10)
        return self.mu * math.log(10.0)
    def get_distribution(self, family_id: int):
        mu = self.mu[family_id]
        sigma = torch.exp(self.log_sigma[family_id])
        return dist.Normal(mu, sigma) # Distribution of log10(c)
        
class NormalConcentration(ConcentrationModel):
    """
    Simple Gaussian assumption.
    c ~ Normal(mu, sigma) clamped at 0.
    """
    def __init__(self, n_families: int, init_mean=10**-6, init_scale=10**-7):
        super().__init__()
        #self.mu = nn.Parameter(torch.ones(n_families) * init_mean)
        #self.log_sigma = nn.Parameter(torch.ones(n_families) * math.log(init_scale))
        self.register_buffer('mu', torch.ones(n_families) * init_mean)
        self.register_buffer('log_sigma', torch.ones(n_families) * math.log(init_scale))

    def sample(self, batch_size, family_ids):
        batch_mu = self.mu[family_ids]
        batch_sigma = torch.exp(self.log_sigma[family_ids])
        
        c = dist.Normal(batch_mu, batch_sigma).rsample()
        return torch.clamp(c, min=1e-12) # Physics constraint (clamp > 0 for safe log)
        
    def get_expected_log_c(self):
        # Approximate expected log(c) as log(mu)
        return torch.log(torch.clamp(self.mu, min=1e-12))
    def get_distribution(self, family_id: int):
        mu = self.mu[family_id]
        sigma = torch.exp(self.log_sigma[family_id])
        return dist.Normal(mu, sigma) # Distribution of c

class LigandEnvironment(nn.Module):
    def __init__(self, n_units: int, n_families: int, conc_model: ConcentrationModel, sigma_init=1.0, delta_shift=4.0):
        """
        Args:
            n_units: Number of protein units
            n_families: Number of ligand families
            conc_model: An INSTANCE of a ConcentrationModel subclass
            sigma_init: Spread of the initial affinities around the mean concentration
            delta_shift: Energy penalty for the closed state to ensure Kc > Ko
        """
        super().__init__()
        self.n_units = n_units
        self.n_families = n_families
        
        # 1. Inject the Concentration Strategy
        self.concentration_model = conc_model
        
        # ----------------------------------------------------------------------
        # SMART INITIALIZATION
        # ----------------------------------------------------------------------
        # 1. Get the expected natural log concentration for each family
        # Shape: (n_families,)
        expected_log_c = self.concentration_model.get_expected_log_c()
        
        # 2. Initialize Open-State Energies (mu_open)
        # Center them around expected_log_c with an initial spread (sigma_init)
        # Broadcast expected_log_c to (n_units, n_families)
        mu_open = expected_log_c.unsqueeze(0).expand(n_units, n_families) + torch.randn(n_units, n_families) * sigma_init
        
        # 3. Initialize Closed-State Energies (mu_closed)
        # Must be strictly higher (weaker binding) than the open state so Kc > Ko
        mu_closed = mu_open + delta_shift
        
        # Combine into interaction_mu: Shape (n_units, n_families, 2)
        # Index 0 is open, Index 1 is closed
        init_mu = torch.stack([mu_open, mu_closed], dim=-1)
        self.interaction_mu = nn.Parameter(init_mu)
        
        # 4. Initialize Standard Deviations
        # Initialize log_sigma to 0.0, meaning the standard deviations start exactly at 1.0
        self.interaction_log_sigma = nn.Parameter(torch.zeros(n_units, n_families, 2))

    def sample_batch(self, batch_size: int):
        device = self.interaction_mu.device
        
        # A. Sample Family IDs
        family_ids = torch.randint(0, self.n_families, (batch_size,), device=device)
        
        # B. Delegate Concentration Sampling
        concentrations = self.concentration_model.sample(batch_size, family_ids)
        
        # C. Sample Energies (Standard Reparameterization)
        mu_T = self.interaction_mu.permute(1, 0, 2)
        sigma_T = torch.exp(self.interaction_log_sigma.permute(1, 0, 2))
        
        batch_mus = mu_T[family_ids]
        batch_sigmas = sigma_T[family_ids]
        
        # energies[:,:,0] = E_open
        # energies[:,:,1] = E_close
        
        energies = dist.Normal(batch_mus, batch_sigmas).rsample()
        
        return energies, concentrations, family_ids

    @torch.no_grad()
    def get_distribution(self, family_id=None, n_points=200):
        d = self.concentration_model.get_distribution(family_id)
        x = torch.linspace(d.mean - 3*d.stddev, d.mean + 3*d.stddev, n_points,device = self.interaction_mu.device)
        pdf = torch.exp(d.log_prob(x))
        return x.cpu().numpy(),pdf.cpu().numpy()