import torch
import torch.nn.functional as F


def discharge_huber_loss(q_pred, q_obs, delta: float = 1.0, mask=None):
    loss = F.huber_loss(q_pred, q_obs, delta=delta, reduction="none")
    if mask is None:
        return loss.mean()
    w = mask.to(loss.dtype)
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def log_discharge_loss(q_pred, q_obs, eps_q: float = 1e-6, mask=None):
    err = (torch.log(q_pred + eps_q) - torch.log(q_obs + eps_q)) ** 2
    if mask is None:
        return err.mean()
    w = mask.to(err.dtype)
    return (err * w).sum() / w.sum().clamp_min(1.0)
