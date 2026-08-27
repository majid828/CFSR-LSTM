import torch
import torch.nn.functional as F


def recession_mask(precip, q_obs, p_threshold: float = 0.1):
    mask = torch.zeros_like(q_obs, dtype=torch.bool)
    if q_obs.size(1) > 1:
        mask[:, 1:] = (precip[:, 1:] < p_threshold) & (q_obs[:, 1:] <= q_obs[:, :-1])
    return mask


def baseflow_loss(q_slow, baseflow_target, delta: float = 1.0):
    valid = torch.isfinite(baseflow_target)
    if not torch.any(valid):
        return q_slow.sum() * 0.0
    return F.huber_loss(q_slow[valid], baseflow_target[valid], delta=delta, reduction="mean")


def slow_recession_losses(q_slow, recession_k, precip, q_obs, p_threshold: float = 0.1):
    """Eq. (62)-(66). k_t multiplies Q^B_{t-1} for t>=1."""
    mask = recession_mask(precip, q_obs, p_threshold)
    if not torch.any(mask):
        z = q_slow.sum() * 0.0
        return z, z, mask
    diff = torch.zeros_like(q_slow)
    rec = torch.zeros_like(q_slow)
    diff[:, 1:] = (q_slow[:, 1:] - q_slow[:, :-1]) ** 2
    rec[:, 1:] = (q_slow[:, 1:] - recession_k[:, 1:] * q_slow[:, :-1]) ** 2
    smooth = diff[mask].mean()
    recession = rec[mask].mean()
    return recession, smooth, mask


def slow_objective(q_slow, recession_k, precip, q_obs, baseflow_target=None, *, lambda_b=1.0, lambda_r=1.0, lambda_m=1.0, delta_b=1.0, p_threshold=0.1):
    l_rec, l_smooth, mask = slow_recession_losses(q_slow, recession_k, precip, q_obs, p_threshold)
    if baseflow_target is not None and torch.isfinite(baseflow_target).any():
        l_b = baseflow_loss(q_slow, baseflow_target, delta_b)
    else:
        l_b = q_slow.sum() * 0.0
    total = lambda_b * l_b + lambda_r * l_rec + lambda_m * l_smooth
    return total, {"baseflow": l_b, "recession": l_rec, "smooth": l_smooth, "recession_fraction": mask.float().mean()}
