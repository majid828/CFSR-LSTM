from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocessing import StandardScaler, chronological_split


class HydroSequenceDataset(Dataset):
    def __init__(self, dynamic, static, q, event_id, baseflow=None, seq_len=168, stride=24, dynamic_scaler=None, start=0, stop=None):
        self.dynamic = np.asarray(dynamic, np.float32)
        self.static = np.asarray(static, np.float32)
        self.q = np.asarray(q, np.float32)
        self.event_id = np.asarray(event_id, np.int64)
        self.baseflow = None if baseflow is None else np.asarray(baseflow, np.float32)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.start = int(start)
        self.stop = len(self.q) if stop is None else int(stop)
        self.scaler = dynamic_scaler
        self.starts = list(range(self.start, max(self.start, self.stop - self.seq_len + 1), self.stride))
        if self.stop - self.seq_len >= self.start and (not self.starts or self.starts[-1] != self.stop - self.seq_len):
            self.starts.append(self.stop - self.seq_len)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]; j = i + self.seq_len
        x = self.dynamic[i:j].copy()
        if self.scaler is not None:
            x = self.scaler.transform(x).astype(np.float32)
        q = self.q[i:j]
        eid = self.event_id[i:j].copy()
        # IDs are only labels; keeping global IDs is fine because losses compare equality within sample.
        base = np.full_like(q, np.nan, dtype=np.float32) if self.baseflow is None else self.baseflow[i:j]
        return {
            "x": torch.from_numpy(x),
            "static": torch.from_numpy(self.static.copy()),
            "q": torch.from_numpy(q),
            "baseflow": torch.from_numpy(base),
            "event_id": torch.from_numpy(eid),
            "precip": torch.from_numpy(self.dynamic[i:j, 0].astype(np.float32)),
            "mask": torch.ones(self.seq_len, dtype=torch.bool),
            "start_index": torch.tensor(i, dtype=torch.long),
        }


def load_npz_datasets(path, seq_len=168, stride=24, train_fraction=0.7, val_fraction=0.15):
    d = np.load(Path(path), allow_pickle=False)
    dynamic, static, q, event_id = d["dynamic"], d["static"], d["q"], d["event_id"]
    baseflow = d["baseflow"] if "baseflow" in d.files else None
    tr, va, te = chronological_split(len(q), train_fraction, val_fraction)
    scaler = StandardScaler().fit(dynamic[tr])
    kwargs = dict(dynamic=dynamic, static=static, q=q, event_id=event_id, baseflow=baseflow, seq_len=seq_len, stride=stride, dynamic_scaler=scaler)
    return (
        HydroSequenceDataset(**kwargs, start=tr.start, stop=tr.stop),
        HydroSequenceDataset(**kwargs, start=va.start, stop=va.stop),
        HydroSequenceDataset(**kwargs, start=te.start, stop=te.stop),
        scaler,
    )
