import torch
import torch.optim as optim
import itertools
import random
import numpy as np

# ------------------------------------------------------------------------------
# IMPORTS (Assumed to exist per your request)
# ------------------------------------------------------------------------------
# We assume these files exist in the folders we created.
from src.environment import LigandEnvironment
from src.physics import Receptor
from objectives.loss import InformationLoss

from config import *

def generate_receptor_indices(n_units, k_sub, n_sensors):
    """
    Generates the identity of the receptors in our array.
    Since 26^5 is huge, we randomly sample 'n_sensors' unique combinations 
    (with replacement, e.g., AABBB) to simulate the array.
    """
    # 1. Generate all possible combinations (approx 142k for 26 choose 5)
    # combinations_with_replacement handles stoichiometry (AAAAA, AAAAB, etc.)
    all_combos = list(itertools.combinations_with_replacement(range(n_units), k_sub))
    
    # 2. Select a subset to simulate the "Octopus Nose"
    if n_sensors > len(all_combos):
        selected = all_combos
    else:
        selected = random.sample(all_combos, n_sensors)
        
    return torch.tensor(selected, dtype=torch.long)

# ------------------------------------------------------------------------------
# MAIN TRAINING LOOP
# ------------------------------------------------------------------------------
def train():
    device = CONFIG["device"]
    print(f"Running on {device}")

    # 1. SETUP THE LAB (Initialize Models)
    # -----------------------------------------------------
    # The Environment: Holds parameters for Ligand-Unit interactions (M, Sigma)
    env = LigandEnvironment(CONFIG["n_units"], CONFIG["n_families"]).to(device)
    
    # The Physics: Holds parameters for Receptor Leakiness (epsilon)
    physics = Receptor(CONFIG["n_units"], CONFIG["k_sub"]).to(device)
    
    # The Array: Define which receptors physically exist in this run
    receptor_indices = generate_receptor_indices(
        CONFIG["n_units"], CONFIG["k_sub"], CONFIG["n_sensors"]
    ).to(device)
    
    # The Objective: Calculates Entropy and Covariance
    loss_fn = InformationLoss()

    # 2. SETUP OPTIMIZER
    # -----------------------------------------------------
    # We optimize BOTH the environment (interaction matrix) and physics (leakiness)
    optimizer = optim.Adam(
        list(env.parameters()) + list(physics.parameters()), 
        lr=CONFIG["learning_rate"]
    )

    # 3. START EXPERIMENT
    # -----------------------------------------------------
    for epoch in range(CONFIG["epochs"]):
        optimizer.zero_grad()
        
        # --- STEP A: SAMPLE THE ENVIRONMENT ---
        # "Throw a bucket of 1024 random ligands at the sensors"
        # energies: (Batch, Units, 2)
        # concs: (Batch,)
        energies, concs, fam_ids = env.sample_batch(CONFIG["batch_size"])
        
        # --- STEP B: RUN THE PHYSICS ---
        # "Measure the current (activity) of every receptor"
        # activity: (Batch, N_Sensors) -> Values between 0 and 1
        activity = physics(energies, concs, receptor_indices)
        
        # --- STEP C: MEASURE INFORMATION (LOSS) ---
        # "How diverse and informative was this activity pattern?"
        # We want to MAXIMIZE information, so we MINIMIZE the negative info.
        total_loss, stats = loss_fn(activity)
        
        # --- STEP D: OPTIMIZE ---
        # "Adjust the interaction energies and leakiness to do better next time"
        total_loss.backward()
        optimizer.step()
        
        # --- LOGGING ---
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss.item():.4f} | "
                  f"Entropy: {stats['entropy']:.4f} | "
                  f"Covariance: {stats['covariance']:.4f}")

if __name__ == "__main__":
    train()