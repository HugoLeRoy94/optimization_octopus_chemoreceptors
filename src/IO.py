import torch
import os
import json
import numpy as np
import pandas as pd
import csv
from datetime import datetime

class Logger:
    def __init__(self, base_path="/app/data/", experiment_name="optimize_array"):
        # 1. Create a unique folder for this specific run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_path, f"{experiment_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.stats_path = os.path.join(self.run_dir, "stats.csv")

    def save_config(self, config_dict):
        """Saves the CONF dictionary as a human-readable JSON."""
        path = os.path.join(self.run_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    def save_stats(self, epoch, stats):
        """Appends a single epoch's stats to a CSV file."""
        # Convert tensors to floats if necessary
        #print(stats_dict)
        #row = {k: (v.item() if hasattr(v, 'item') else v) for k, v in stats_dict.items()}
        stats['epoch'] = epoch
        
        file_exists = os.path.isfile(self.stats_path)
        
        with open(self.stats_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)
    def save_checkpoint(self, epoch, env, physics, receptor_indices, is_best=False):
        """Saves a training snapshot."""
        checkpoint = {
            "epoch": epoch,
            "env_state": env.state_dict(),
            "physics_state": physics.state_dict(),
            "receptor_indices": receptor_indices.cpu(),
        }
        
        # Save regular epoch checkpoint
        fname = f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, os.path.join(self.ckpt_dir, fname))
        
        # Also save as 'best' if flagged
        if is_best:
            torch.save(checkpoint, os.path.join(self.run_dir, "best_model.pt"))

    def load_run(self, run_folder, filename="best_model.pt"):
        """Utility to load everything back from a specific run folder."""
        path = os.path.join(run_folder, filename)
        return torch.load(path)

    def load_all_checkpoints(self):
        """Returns a list of all stats from all saved checkpoints in order."""
        files = sorted([f for f in os.listdir(self.ckpt_dir) if f.endswith('.pt')])
        all_loads = []
        for f in files:
            all_loads.append(torch.load(os.path.join(self.ckpt_dir, f)))
        return all_loads
    
    def load_objects(path, device='cpu'):
        """
        returns environment, physics, loss, indices, stats, conf file
        """
        ckpt = torch.load(path, map_location=device)
        c = ckpt['CONF']
        
        # Rebuild objects
        strat = LogNormalConcentration(n_families=c['n_families'], init_mean=5.0)
        e = LigandEnvironment(c['n_units'], c['n_families'], conc_model=strat)
        p = Receptor(c['n_units'], c['k_sub'])
        
        # Load state (buffers like 'mu' are restored here!)
        e.load_state_dict(ckpt['env_state'])
        p.load_state_dict(ckpt['physics_state'])

        l = ExactInformationLoss(k_knn=c['k_knn'])
        
        return e.to(device), p.to(device),l.to(device), ckpt['receptor_indices'], ckpt['stats'],c

    def load_history(self):
        """One-liner to get all stats for plotting."""
        return pd.read_csv(self.stats_path)