import torch
# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
CONFIG = {
    "n_units": 26,            # Number of available genes (CR518, CR918, etc.)
    "k_sub": 5,               # Pentamers
    "n_families": 1,         # Number of ligand families (e.g. Alcohols, Esters...)
    "n_sensors": 1000,        # Number of receptors in our array (subset of all possibilities)
    
    "batch_size": 1024,       # Number of ligands sampled per training step
    "learning_rate": 0.01,
    "epochs": 1000,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}