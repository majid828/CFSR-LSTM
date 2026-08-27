from copy import deepcopy
from pathlib import Path
import yaml


def load_config(path="config.yaml"):
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base, updates):
    out = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
