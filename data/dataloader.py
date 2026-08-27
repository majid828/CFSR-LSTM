from torch.utils.data import DataLoader
from data.dataset import load_npz_datasets


def build_dataloaders(cfg):
    dc = cfg["data"]
    train_ds, val_ds, test_ds, scaler = load_npz_datasets(
        dc["path"], seq_len=dc["seq_len"], stride=dc["stride"],
        train_fraction=dc["train_fraction"], val_fraction=dc["val_fraction"],
    )
    common = dict(batch_size=dc["batch_size"], num_workers=dc.get("num_workers", 0), pin_memory=dc.get("pin_memory", False))
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
        scaler,
    )
