from pathlib import Path
import torch


def save_checkpoint(path, model, optimizers=None, epoch=None, extra=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "epoch": epoch, "extra": extra or {}}
    if optimizers:
        state["optimizers"] = {k: o.state_dict() for k, o in optimizers.items()}
    torch.save(state, path)


def load_checkpoint(path, model, optimizers=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizers and "optimizers" in state:
        for k, o in optimizers.items():
            if k in state["optimizers"]:
                o.load_state_dict(state["optimizers"][k])
    return state
