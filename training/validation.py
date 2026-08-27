import torch
from losses.total_loss import compute_cfsr_losses


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def validate_cfsr(model, loader, cfg, device):
    model.eval()
    sums = {}; n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["x"], batch["static"])
        bundle = compute_cfsr_losses(out, batch, cfg)
        vals = {"total": bundle.total_scalarized, "general": bundle.general, "event": bundle.event, "slow": bundle.slow, **bundle.components}
        for k, v in vals.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach().cpu())
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}
