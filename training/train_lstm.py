import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
import torch
from torch.optim import Adam
from tqdm import tqdm

from data.synthetic_generator import generate_synthetic_basin, save_npz
from data.dataloader import build_dataloaders
from models.standard_lstm import StandardLSTM
from losses.discharge_loss import discharge_huber_loss, log_discharge_loss
from losses.peak_loss import peak_magnitude_loss
from losses.timing_loss import differentiable_peak_timing_loss
from utils.config_loader import load_config
from utils.seed import set_seed


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config.yaml"); p.add_argument("--peak-aware", action="store_true"); args = p.parse_args()
    cfg = load_config(args.config); set_seed(cfg.get("seed", 42))
    path = Path(cfg["data"]["path"])
    if not path.exists():
        save_npz(path, generate_synthetic_basin(cfg["data"].get("synthetic_steps", 5000), cfg.get("seed", 42)))
    tr, va, _, _ = build_dataloaders(cfg)
    mc, lc, tc = cfg["model"], cfg["loss"], cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() and tc.get("use_cuda", True) else "cpu")
    model = StandardLSTM(mc["dynamic_dim"], mc["static_dim"], mc.get("baseline_hidden", 64)).to(device)
    opt = Adam(model.parameters(), lr=tc.get("baseline_lr", 1e-3))
    best = float("inf")
    for epoch in range(tc["epochs"]):
        model.train()
        for b in tqdm(tr, desc=f"baseline {epoch+1}/{tc['epochs']}"):
            b = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}
            out = model(b["x"], b["static"]); q = out["q_total"]
            loss = lc["lambda_q"]*discharge_huber_loss(q,b["q"],lc["delta_q"]) + lc["lambda_logq"]*log_discharge_loss(q,b["q"],lc["eps_q"])
            if args.peak_aware:
                loss = loss + lc["lambda_peak"]*peak_magnitude_loss(q,b["q"],b["event_id"],lc["delta_peak"],lc["eps_q"]) + lc["lambda_timing"]*differentiable_peak_timing_loss(q,b["q"],b["event_id"],lc["kappa_timing"],lc["delta_timing"],lc["eps_q"])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip_norm", 5.0)); opt.step()
        model.eval(); vals=[]
        with torch.no_grad():
            for b in va:
                b={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}; q=model(b["x"],b["static"])["q_total"]
                vals.append(float(discharge_huber_loss(q,b["q"],lc["delta_q"])))
        v=sum(vals)/max(len(vals),1)
        if v<best:
            best=v; Path("results/checkpoints").mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"val":v},"results/checkpoints/lstm_best.pt")

if __name__ == "__main__": main()
