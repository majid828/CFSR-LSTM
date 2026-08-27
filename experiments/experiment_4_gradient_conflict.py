"""Experiment 4: full CFSR-LSTM with conflict diagnostics and asymmetric projected-gradient surgery."""
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
    cfg=deep_update(cfg,{"training":{"use_gradient_surgery":True,"log_path":"results/logs/full_cfsr.csv","checkpoint_path":"results/checkpoints/full_cfsr.pt"}})
    tr,va,_,_=build_dataloaders(cfg); CFSRTrainer(build_model(cfg),cfg).fit(tr,va)
