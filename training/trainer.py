from pathlib import Path
from typing import Dict, List
import torch
from torch.optim import Adam
from tqdm import tqdm

from losses.total_loss import compute_cfsr_losses
from models.gradient_surgery import asymmetric_project, combine_shared_gradients, cosine_alignment
from training.validation import move_batch, validate_cfsr
from training.checkpoint import save_checkpoint
from utils.logger import CSVLogger


def _autograd(loss, params, retain_graph=True):
    if not params:
        return []
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return [g for g in grads]


def _set_grads(params, grads):
    for p, g in zip(params, grads):
        p.grad = None if g is None else g.detach().clone()


class CFSRTrainer:
    """Implements Eqs. (71)-(93) with staged activation of specialization and surgery."""

    def __init__(self, model, cfg, device=None):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() and cfg["training"].get("use_cuda", True) else "cpu"))
        self.model.to(self.device)
        groups = model.parameter_groups()
        self.params = {k: list(v) for k, v in groups.items()}
        tc = cfg["training"]
        self.optimizers = {
            "shared": Adam(self.params["shared"], lr=tc["lr_shared"], weight_decay=tc.get("weight_decay", 0.0)),
            "fast": Adam(self.params["fast"], lr=tc["lr_fast"], weight_decay=tc.get("weight_decay", 0.0)),
            "slow": Adam(self.params["slow"], lr=tc["lr_slow"], weight_decay=tc.get("weight_decay", 0.0)),
            "routing": Adam(self.params["routing"], lr=tc["lr_routing"], weight_decay=tc.get("weight_decay", 0.0)),
        }
        self.logger = CSVLogger(tc.get("log_path", "results/logs/train.csv"))
        self.global_step = 0

    def _phase(self, epoch):
        tc = self.cfg["training"]
        if epoch < tc["warmup_epochs"]:
            return 1
        if epoch < tc["surgery_start_epoch"]:
            return 2
        return 3

    def train_step(self, batch, epoch):
        self.model.train()
        batch = move_batch(batch, self.device)
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)
        out = self.model(batch["x"], batch["static"])
        bundle = compute_cfsr_losses(out, batch, self.cfg)
        phase = self._phase(epoch)
        lc = self.cfg["loss"]

        if phase == 1:
            # Representation warm-up: all active forward modules learn only the general objective.
            all_named = [(name, self.params[name]) for name in ["shared", "fast", "slow", "routing"]]
            for i, (name, params) in enumerate(all_named):
                grads = _autograd(bundle.general, params, retain_graph=(i < len(all_named) - 1))
                _set_grads(params, grads)
            cos = torch.tensor(0.0, device=self.device); projected = False
        else:
            # Shared: g_A = grad(L_G + L_E), g_S = grad(L_S), then asymmetric surgery if Phase III.
            g_a = _autograd(bundle.general + bundle.event, self.params["shared"], retain_graph=True)
            g_s = _autograd(bundle.slow, self.params["shared"], retain_graph=True)
            cos = cosine_alignment(g_a, g_s, self.cfg["training"].get("gradient_eps", 1e-12))
            if phase >= 3 and self.cfg["training"].get("use_gradient_surgery", True):
                g_a_tilde, cos, projected = asymmetric_project(g_a, g_s, self.cfg["training"].get("gradient_eps", 1e-12))
            else:
                g_a_tilde, projected = g_a, False
            _set_grads(self.params["shared"], combine_shared_gradients(g_a_tilde, g_s))

            # Branch-specific objectives, exactly matching Sections 16.1-16.3.
            j_fast = bundle.general + bundle.event
            # J_R = lambda_Q L_Q + event; intentionally excludes L_logQ according to Eq. (90).
            j_route = lc["lambda_q"] * bundle.components["LQ"] + bundle.event
            j_slow = bundle.general + bundle.slow
            _set_grads(self.params["fast"], _autograd(j_fast, self.params["fast"], retain_graph=True))
            _set_grads(self.params["routing"], _autograd(j_route, self.params["routing"], retain_graph=True))
            _set_grads(self.params["slow"], _autograd(j_slow, self.params["slow"], retain_graph=False))

        clip = self.cfg["training"].get("grad_clip_norm")
        if clip is not None and clip > 0:
            for params in self.params.values():
                torch.nn.utils.clip_grad_norm_(params, clip)
        for opt in self.optimizers.values():
            opt.step()

        self.global_step += 1
        return {
            "loss": float(bundle.total_scalarized.detach().cpu()),
            "general": float(bundle.general.detach().cpu()),
            "event": float(bundle.event.detach().cpu()),
            "slow": float(bundle.slow.detach().cpu()),
            "grad_cosine": float(cos.detach().cpu()),
            "projected": float(projected),
            "phase": phase,
            **{k: float(v.detach().cpu()) for k, v in bundle.components.items()},
        }

    def fit(self, train_loader, val_loader=None):
        tc = self.cfg["training"]
        best = float("inf")
        history = []
        for epoch in range(tc["epochs"]):
            rows = []
            bar = tqdm(train_loader, desc=f"epoch {epoch+1}/{tc['epochs']}")
            for batch in bar:
                row = self.train_step(batch, epoch)
                rows.append(row)
                bar.set_postfix(loss=f"{row['loss']:.4f}", cos=f"{row['grad_cosine']:.3f}")
            epoch_row = {k: sum(r[k] for r in rows) / max(len(rows), 1) for k in rows[0]} if rows else {}
            epoch_row.update({"epoch": epoch + 1})
            if val_loader is not None:
                val = validate_cfsr(self.model, val_loader, self.cfg, self.device)
                epoch_row.update({f"val_{k}": v for k, v in val.items()})
                target = val.get("total", epoch_row.get("loss", float("inf")))
            else:
                target = epoch_row.get("loss", float("inf"))
            self.logger.log(epoch_row); history.append(epoch_row)
            if target < best:
                best = target
                save_checkpoint(tc.get("checkpoint_path", "results/checkpoints/cfsr_best.pt"), self.model, self.optimizers, epoch + 1, {"best_val": best})
        return history
