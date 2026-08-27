"""Experiment 2: dual-memory model without latent partition, explicit routing, or gradient surgery."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config, deep_update
from data.dataloader import build_dataloaders
from training.train_cfsr import build_model
from training.trainer import CFSRTrainer
from utils.seed import set_seed

if __name__ == "__main__":
    cfg=load_config("config.yaml"); set_seed(cfg.get("seed",42))
    cfg=deep_update(cfg,{"model":{"use_partition_gate":False,"use_routing":False},"training":{"use_gradient_surgery":False,"log_path":"results/logs/dual_memory.csv","checkpoint_path":"results/checkpoints/dual_memory.pt"},"loss":{"lambda_timing":0.0,"lambda_peak":0.0}})
    tr,va,_,_=build_dataloaders(cfg); CFSRTrainer(build_model(cfg),cfg).fit(tr,va)
