import torch
import torch.nn.functional as F


def _zero_like(x):
    return x.sum() * 0.0


def peak_magnitude_loss(q_pred, q_obs, event_id, delta: float = 1.0, eps_q: float = 1e-6):
    """Eq. (49)-(51), averaged over observed event windows in a batch."""
    losses = []
    for b in range(q_pred.size(0)):
        ids = torch.unique(event_id[b])
        for eid in ids:
            if int(eid.item()) < 0:
                continue
            m = event_id[b] == eid
            if not torch.any(m):
                continue
            obs_peak = q_obs[b][m].max()
            pred_peak = q_pred[b][m].max()
            rel = (pred_peak - obs_peak) / (obs_peak + eps_q)
            losses.append(F.huber_loss(rel, torch.zeros_like(rel), delta=delta, reduction="mean"))
    return torch.stack(losses).mean() if losses else _zero_like(q_pred)
