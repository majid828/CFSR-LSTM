from typing import Iterable, List, Optional, Tuple
import torch

TensorOpt = Optional[torch.Tensor]


def _dot(gs1: Iterable[TensorOpt], gs2: Iterable[TensorOpt]):
    terms = [torch.sum(a * b) for a, b in zip(gs1, gs2) if a is not None and b is not None]
    if not terms:
        return None
    return torch.stack(terms).sum()


def _norm_sq(gs: Iterable[TensorOpt], reference: torch.Tensor):
    terms = [torch.sum(g * g) for g in gs if g is not None]
    return torch.stack(terms).sum() if terms else reference.new_zeros(())


def cosine_alignment(event_grads: List[TensorOpt], slow_grads: List[TensorOpt], eps: float = 1e-12) -> torch.Tensor:
    dot = _dot(event_grads, slow_grads)
    if dot is None:
        ref = next((g for g in event_grads + slow_grads if g is not None), None)
        return torch.tensor(0.0) if ref is None else ref.new_zeros(())
    na = torch.sqrt(_norm_sq(event_grads, dot))
    ns = torch.sqrt(_norm_sq(slow_grads, dot))
    return dot / (na * ns + eps)


def asymmetric_project(event_grads: List[TensorOpt], slow_grads: List[TensorOpt], eps: float = 1e-12) -> Tuple[List[TensorOpt], torch.Tensor, bool]:
    """Eq. (78): remove only the locally opposing component of g_A along g_S."""
    dot = _dot(event_grads, slow_grads)
    ref = next((g for g in event_grads + slow_grads if g is not None), None)
    if ref is None:
        return event_grads, torch.tensor(0.0), False
    cosine = cosine_alignment(event_grads, slow_grads, eps)
    if dot is None or dot.detach().item() >= 0:
        return event_grads, cosine, False
    slow_norm_sq = _norm_sq(slow_grads, ref)
    if slow_norm_sq.detach().item() <= 0:
        return event_grads, cosine, False
    coeff = dot / (slow_norm_sq + eps)
    projected = []
    for ga, gs in zip(event_grads, slow_grads):
        if ga is None:
            projected.append(None)
        elif gs is None:
            projected.append(ga)
        else:
            projected.append(ga - coeff * gs)
    return projected, cosine, True


def combine_shared_gradients(projected_event: List[TensorOpt], slow_grads: List[TensorOpt]) -> List[TensorOpt]:
    out = []
    for ga, gs in zip(projected_event, slow_grads):
        if ga is None and gs is None:
            out.append(None)
        elif ga is None:
            out.append(gs)
        elif gs is None:
            out.append(ga)
        else:
            out.append(ga + gs)
    return out
