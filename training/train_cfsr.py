import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

from data.synthetic_generator import generate_synthetic_basin, save_npz
from data.dataloader import build_dataloaders
from models.cfsr_lstm import CFSRLSTM
from training.trainer import CFSRTrainer
from utils.config_loader import load_config
from utils.seed import set_seed


def build_model(cfg):
    mc = cfg["model"]
    return CFSRLSTM(
        dynamic_dim=mc["dynamic_dim"], static_dim=mc["static_dim"], latent_dim=mc["latent_dim"],
        fast_hidden=mc["fast_hidden"], slow_hidden=mc["slow_hidden"], max_lag=mc["max_lag"], dt=mc["dt"],
        activation=mc.get("activation", "tanh"), use_partition_gate=mc.get("use_partition_gate", True),
        use_cross_memory=mc.get("use_cross_memory", True), use_routing=mc.get("use_routing", True),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    path = Path(cfg["data"]["path"])
    if not path.exists() and cfg["data"].get("auto_generate_synthetic", True):
        save_npz(path, generate_synthetic_basin(cfg["data"].get("synthetic_steps", 5000), cfg.get("seed", 42),
                                                dynamic_dim=cfg["model"]["dynamic_dim"], static_dim=cfg["model"]["static_dim"]))
    train_loader, val_loader, _, _ = build_dataloaders(cfg)
    trainer = CFSRTrainer(build_model(cfg), cfg, args.device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
