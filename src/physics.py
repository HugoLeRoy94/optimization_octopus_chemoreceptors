import torch
import torch.nn as nn

class Receptor(nn.Module):
    """
    The Physics Module.
    
    Responsibilities:
    1. Holds the learnable parameters for receptor 'leakiness' (epsilon).
    2. Implements the Monod-Wyman-Changeux (MWC) equation.
    3. Normalizes activity based on min/max possible opening.
    """
    def __init__(self, n_units: int, k_sub: int = 5):
        """
        Args:
            n_units: Total number of available protein subunits (e.g., 26).
            k_sub: Number of subunits in a receptor (e.g., 5 for pentamers).
        """
        super().__init__()
        self.n_units = n_units
        self.k_sub = k_sub
        
        # ======================================================================
        # 1. LEAKINESS PARAMETER (Epsilon)
        # ======================================================================
        # epsilon_r = Sum(epsilon_units). 
        # We learn one epsilon value per unit type.
        # This represents the energy cost to open the channel in absence of ligand.
        self.epsilon_units = nn.Parameter(torch.randn(n_units))

    def _compute_p_open(self, 
                        energies_k: torch.Tensor, 
                        concentrations: torch.Tensor, 
                        epsilon_r: torch.Tensor) -> torch.Tensor:
        """
        Core MWC Equation:
        p_open = Numerator / (Numerator + Denominator)
        Numerator = Product(1 + c/Ko)
        Denominator = L * Product(1 + c/Kc)
        
        Args:
            energies_k: (Batch, N_Receptors, k_sub, 2) - Interaction energies for specific units
            concentrations: (Batch, 1, 1) - Ligand concentrations
            epsilon_r: (1, N_Receptors) - Total leakiness energy per receptor
        
        Returns:
            p_open: (Batch, N_Receptors)
        """
        # Unpack Open (idx 0) and Closed (idx 1) energies
        # K = exp(E), so 1/K = exp(-E)
        # energies_k is (Batch, R, k, 2)
        E_open = energies_k[..., 0]   # (Batch, R, k)
        E_closed = energies_k[..., 1] # (Batch, R, k)
        
        # Calculate Affinity terms: (1 + c/K) = (1 + c * exp(-E))
        # We expand concentrations to match (Batch, R, k)
        c_expanded = concentrations
        
        # Term Open: Product_u (1 + c * e^{-Eu_open})
        term_open_per_unit = 1.0 + c_expanded * torch.exp(-E_open)
        term_open = torch.prod(term_open_per_unit, dim=-1) # Product over k subunits -> (Batch, R)
        
        # Term Closed: Product_u (1 + c * e^{-Eu_closed})
        term_closed_per_unit = 1.0 + c_expanded * torch.exp(-E_closed)
        term_closed = torch.prod(term_closed_per_unit, dim=-1) # -> (Batch, R)
        
        # Leakiness Factor L = exp(-epsilon_r)
        # Note: In your text you wrote e^{-epsilon}, usually L = exp(epsilon) or exp(-epsilon)
        # depending on definition of Delta E. 
        # Defined here as: epsilon_r = E_closed_state - E_open_state (without ligand)
        L = torch.exp(-epsilon_r) # (1, R)
        
        # Probability
        numerator = term_open
        denominator = term_open + (L * term_closed)
        
        return numerator / denominator

    def forward(self, 
                energies: torch.Tensor, 
                concentrations: torch.Tensor, 
                receptor_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            energies: (Batch, n_units, 2) 
                - Sampled interaction energies for ALL units.
            concentrations: (Batch,)
                - Sampled concentrations.
            receptor_indices: (N_Receptors, k_sub)
                - Integer tensor defining the stoichiometry (e.g., [[0,0,0,0,0], [0,1,1,1,1]...])
                
        Returns:
            normalized_activity: (Batch, N_Receptors) 
                - Value between 0 and 1.
        """
        batch_size = energies.shape[0]
        n_receptors = receptor_indices.shape[0]
        
        # ----------------------------------------------------------------------
        # A. GATHER ENERGIES FOR SPECIFIC RECEPTORS
        # ----------------------------------------------------------------------
        # We need to map the batch of generic unit energies to the specific 
        # combinations defined in receptor_indices.
        
        # 1. Expand receptor indices to match batch size
        # Shape: (Batch, N_Receptors, k_sub)
        batch_indices = receptor_indices.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 2. Gather energies
        # Input energies: (Batch, n_units, 2)
        # We want: (Batch, N_Receptors, k_sub, 2)
        
        # Gather requires indices to match dimensions. We expand energies for the gather.
        # This is a bit tricky in PyTorch. An easier way is F.embedding or direct indexing 
        # if we treat batch independently.
        
        # Let's use direct advanced indexing.
        # energies shape: [B, Units, 2]
        # We want to pull slices.
        
        # Reshape energies to (B, Units, 2)
        # We select specific units.
        # efficient approach: 
        # specific_energies[b, r, k, :] = energies[b, receptor_indices[r, k], :]
        
        # PyTorch "gather" on the unit dimension:
        # We construct a huge index tensor or use torch.take_along_dim? No.
        # Best way: Linearize receptor indices?
        
        # Let's use simple indexing.
        # energies: (B, Units, 2) -> (B, Units, 2)
        # indices: (R, k)
        # Flatten R and k -> (R*k)
        flat_indices = receptor_indices.view(-1)
        
        # Select: (B, R*k, 2)
        gathered_flat = energies[:, flat_indices, :]
        
        # Reshape back: (B, R, k, 2)
        energies_k = gathered_flat.view(batch_size, n_receptors, self.k_sub, 2)
        
        # ----------------------------------------------------------------------
        # B. COMPUTE LEAKINESS (Epsilon)
        # ----------------------------------------------------------------------
        # Sum of epsilons of the constituent units
        # self.epsilon_units: (n_units,)
        eps_k = self.epsilon_units[receptor_indices] # (R, k)
        epsilon_r = eps_k.sum(dim=1).unsqueeze(0)    # (1, R)
        
        # ----------------------------------------------------------------------
        # C. COMPUTE ACTIVITY
        # ----------------------------------------------------------------------
        # Reshape concentrations for broadcasting: (Batch, 1, 1)
        c_reshaped = concentrations.view(batch_size, 1, 1)
        
        # 1. Compute p_open at current concentration
        p_c = self._compute_p_open(energies_k, c_reshaped, epsilon_r)
        
        # 2. Compute p_min (c = 0)
        # At c=0, activity is 1 / (1 + L)
        # This is constant across the batch, but depends on receptor
        L = torch.exp(-epsilon_r)
        p_min = 1.0 / (1.0 + L) # (1, R)
        
        # 3. Compute p_max (c -> infinity)
        # This is the "saturated" probability.
        # Limit of (1+c/Ko) / [(1+c/Ko) + L(1+c/Kc)] as c->inf
        # = (1/Ko)^k / [(1/Ko)^k + L(1/Kc)^k]
        # = 1 / [1 + L * (Ko/Kc)^k]
        # = 1 / [1 + L * exp(k * (Eo - Ec))]
        
        # Diff E = E_open - E_closed
        delta_E = energies_k[..., 0] - energies_k[..., 1] # (B, R, k)
        sum_delta_E = delta_E.sum(dim=2) # (B, R)
        
        # p_max depends on the ligand! (Different ligands have different saturation levels)
        term_sat = L * torch.exp(sum_delta_E)
        p_max = 1.0 / (1.0 + term_sat)
        
        # ----------------------------------------------------------------------
        # D. RENORMALIZE
        # ----------------------------------------------------------------------
        # Avoid division by zero if p_max ~ p_min (e.g. silent receptor)
        denominator = p_max - p_min
        
        # Numerical stability mask: if dynamic range is too small, return 0 activity
        mask = denominator > 1e-6
        
        normalized = (p_c - p_min) / (denominator + 1e-8)
        
        # Apply mask and clamp to [0,1] just in case of float errors
        normalized = normalized * mask.float()
        return torch.clamp(normalized, 0.0, 1.0)