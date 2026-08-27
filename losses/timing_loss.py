import torch
import torch.nn.functional as F


def differentiable_peak_timing_loss(q_pred, q_obs, event_id, kappa: float = 10.0, delta: float = 1.0, eps_q: float = 1e-6):
    """Eq. (52)-(55): soft predicted peak time, hard observed target."""
    losses = []
    for b in range(q_pred.size(0)):
        ids = torch.unique(event_id[b])
        for eid in ids:
            if int(eid.item()) < 0:
                continue
            idx = torch.where(event_id[b] == eid)[0]
            if idx.numel() == 0:
                continue
            obs_event = q_obs[b, idx]
            pred_event = q_pred[b, idx]
            obs_peak = obs_event.max().detach()
            logits = kappa * pred_event / (obs_peak + eps_q)
            p = torch.softmax(logits, dim=0)
            # Use indices within the event window. Translation cancels in the timing difference.
            local_t = torch.arange(idx.numel(), device=q_pred.device, dtype=q_pred.dtype)
            pred_t = torch.sum(local_t * p)
            obs_t = torch.argmax(obs_event).to(q_pred.dtype)
            normalized = (pred_t - obs_t) / float(idx.numel())
            losses.append(F.huber_loss(normalized, torch.zeros_like(normalized), delta=delta, reduction="mean"))
    return torch.stack(losses).mean() if losses else q_pred.sum() * 0.0


def hydrograph_shape_loss(q_pred, q_obs, event_id, eps_q: float = 1e-6):
    """Optional Eq. (56)-(60): discrete Wasserstein-1-type shape loss."""
    losses = []
    for b in range(q_pred.size(0)):
        for eid in torch.unique(event_id[b]):
            if int(eid.item()) < 0:
                continue
            idx = torch.where(event_id[b] == eid)[0]
            if idx.numel() == 0:
                continue
            obs = q_obs[b, idx]
            pred = q_pred[b, idx]
            p_obs = obs / (obs.sum() + eps_q)
            p_mod = pred / (pred.sum() + eps_q)
            losses.append(torch.mean(torch.abs(torch.cumsum(p_mod, 0) - torch.cumsum(p_obs, 0))))
    return torch.stack(losses).mean() if losses else q_pred.sum() * 0.0
