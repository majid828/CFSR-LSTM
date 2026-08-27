import argparse
from pathlib import Path
import numpy as np


def _route_signal(signal, delay_kernel):
    out = np.convolve(signal, delay_kernel, mode="full")[: len(signal)]
    return out


def generate_synthetic_basin(n_steps=5000, seed=42, dt=1.0, dynamic_dim=5, static_dim=6):
    """Generate a reproducible rainfall-runoff toy basin with fast routed and slow storage components.

    Dynamic columns: precipitation, temperature, radiation, wind, PET.
    This generator is intended for code validation, not scientific benchmarking.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    # Static basin attributes (scaled generic descriptors).
    static = rng.uniform(0.1, 1.0, size=static_dim).astype(np.float32)
    area, slope = static[0], static[1]

    # Meteorology with intermittent storms.
    temperature = 12 + 9 * np.sin(2 * np.pi * t / (24 * 365)) + rng.normal(0, 2, n_steps)
    radiation = np.clip(250 + 180 * np.sin(2 * np.pi * (t % 24) / 24) + rng.normal(0, 35, n_steps), 0, None)
    wind = np.clip(rng.gamma(2.0, 1.2, n_steps), 0, None)
    pet = np.clip(0.02 * radiation + 0.15 * np.maximum(temperature, 0) + rng.normal(0, 0.3, n_steps), 0, None)
    rain_occurs = rng.random(n_steps) < 0.07
    precipitation = rain_occurs * rng.gamma(shape=1.7, scale=8.0, size=n_steps)
    for i in range(1, n_steps):
        if precipitation[i - 1] > 4 and rng.random() < 0.5:
            precipitation[i] += rng.gamma(1.4, 4.0)

    # Antecedent storage and nonlinear runoff generation.
    storage = np.zeros(n_steps, dtype=np.float64)
    baseflow = np.zeros(n_steps, dtype=np.float64)
    quick_gen = np.zeros(n_steps, dtype=np.float64)
    k_s = 0.985 + 0.01 * static[2]
    for i in range(1, n_steps):
        recharge = 0.12 * precipitation[i] * (0.5 + 0.5 * static[3])
        evap = 0.018 * pet[i]
        storage[i] = max(0.0, k_s * storage[i - 1] + recharge - evap)
        baseflow[i] = 0.018 * storage[i]
        wetness = 1.0 - np.exp(-storage[i - 1] / 25.0)
        quick_gen[i] = precipitation[i] * (0.08 + 0.5 * wetness + 0.18 * slope)

    # Causal routing kernel: larger basins tend to have larger mean lag.
    mean_lag = 1.5 + 4.0 * area / max(0.2, slope)
    max_lag = 24
    lags = np.arange(max_lag + 1)
    kernel = np.exp(-0.5 * ((lags - mean_lag) / (1.2 + 1.3 * area)) ** 2)
    kernel /= kernel.sum()
    quickflow = _route_signal(quick_gen, kernel)
    q = np.maximum(0.0, quickflow + baseflow + rng.normal(0, 0.02 + 0.01 * np.sqrt(quickflow + baseflow + 1), n_steps))

    dynamic = np.stack([precipitation, temperature, radiation, wind, pet], axis=-1).astype(np.float32)
    if dynamic_dim != 5:
        if dynamic_dim < 5:
            dynamic = dynamic[:, :dynamic_dim]
        else:
            extra = rng.normal(size=(n_steps, dynamic_dim - 5)).astype(np.float32)
            dynamic = np.concatenate([dynamic, extra], axis=-1)

    # Event windows are based only on observed forcings/discharge, for synthetic supervision.
    threshold = np.quantile(q, 0.90)
    event_id = np.full(n_steps, -1, dtype=np.int64)
    active = (q >= threshold) | (precipitation > 2.0)
    # Expand each active point by a causal/local response window.
    expanded = active.copy()
    for lag in range(1, 7):
        expanded[lag:] |= active[:-lag]
        expanded[:-lag] |= active[lag:]
    eid = 0
    i = 0
    while i < n_steps:
        if not expanded[i]:
            i += 1; continue
        j = i
        while j + 1 < n_steps and expanded[j + 1]:
            j += 1
        event_id[i:j + 1] = eid
        eid += 1
        i = j + 1

    return {
        "dynamic": dynamic,
        "static": static,
        "q": q.astype(np.float32),
        "baseflow": baseflow.astype(np.float32),
        "event_id": event_id,
        "dt": np.float32(dt),
    }


def save_npz(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/synthetic_basin.npz")
    p.add_argument("--n-steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    save_npz(args.output, generate_synthetic_basin(args.n_steps, args.seed))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
