from dataclasses import dataclass
from losses.discharge_loss import discharge_huber_loss, log_discharge_loss
from losses.peak_loss import peak_magnitude_loss
from losses.timing_loss import differentiable_peak_timing_loss, hydrograph_shape_loss
from losses.slow_flow_loss import slow_objective


@dataclass
class LossBundle:
    total_scalarized: object
    general: object
    event: object
    slow: object
    components: dict


def compute_cfsr_losses(outputs, batch, cfg):
    lc = cfg["loss"]
    q = outputs["q_total"]
    qobs = batch["q"]
    mask = batch.get("mask")
    lq = discharge_huber_loss(q, qobs, lc["delta_q"], mask)
    llog = log_discharge_loss(q, qobs, lc["eps_q"], mask)
    lpeak = peak_magnitude_loss(q, qobs, batch["event_id"], lc["delta_peak"], lc["eps_q"])
    ltime = differentiable_peak_timing_loss(q, qobs, batch["event_id"], lc["kappa_timing"], lc["delta_timing"], lc["eps_q"])
    lshape = hydrograph_shape_loss(q, qobs, batch["event_id"], lc["eps_q"]) if lc.get("lambda_shape", 0.0) > 0 else q.sum() * 0.0
    lslow, slow_parts = slow_objective(
        outputs["q_slow"], outputs["recession_k"], batch["precip"], qobs, batch.get("baseflow"),
        lambda_b=lc["lambda_baseflow"], lambda_r=lc["lambda_recession"], lambda_m=lc["lambda_smooth"],
        delta_b=lc["delta_baseflow"], p_threshold=lc["precip_recession_threshold"],
    )
    general = lc["lambda_q"] * lq + lc["lambda_logq"] * llog
    event = lc["lambda_peak"] * lpeak + lc["lambda_timing"] * ltime + lc.get("lambda_shape", 0.0) * lshape
    total = general + event + lslow
    parts = {"LQ": lq, "LlogQ": llog, "Lpeak": lpeak, "Ltime": ltime, "Lshape": lshape, "LS": lslow, **{f"LS_{k}": v for k,v in slow_parts.items()}}
    return LossBundle(total, general, event, lslow, parts)
