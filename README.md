# CFSR-LSTM Paper Code

Reference implementation of the **Conflict-Aware Fast-Slow Routing LSTM (CFSR-LSTM)** described in the supplied methodology draft. The code follows the proposed decomposition into: (1) a shared hydrometeorological encoder, (2) a learned latent fast/slow partition gate, (3) separate fast and slow recurrent memories, (4) controlled bidirectional cross-memory communication, (5) nonnegative rapid-response and slow-flow outputs, (6) state-dependent causal routing for peak timing, and (7) asymmetric projected-gradient conflict handling on shared parameters.

## Method-to-code map

- Eq. (11): `models/cfsr_lstm.py` shared encoder.
- Eq. (12)-(15): `models/partition_gate.py`.
- Eq. (16)-(27): `models/fast_slow_memory.py` via independent `nn.LSTMCell` modules.
- Eq. (28)-(31): `models/cross_memory.py`; both communicated states use the same preliminary states.
- Eq. (35): `fast_response_head` + `softplus` in `models/cfsr_lstm.py`.
- Eq. (36)-(42): `models/routing_module.py`; routing uses `Q_F[t] = sum_l pi[t-l,l] R_F[t-l]` exactly.
- Eq. (43)-(45): nonnegative slow output and additive total discharge in `models/cfsr_lstm.py`.
- Eq. (46)-(70): `losses/`.
- Eq. (71)-(87): `models/gradient_surgery.py` and `training/trainer.py`.
- Eq. (88)-(93): branch-specific objectives in `training/trainer.py`.
- Staged schedule (Sections 18 and 27): configured by `warmup_epochs` and `surgery_start_epoch`.

## Important scientific interpretation

`q_slow` is a learned **slow-flow component**, not automatically physical groundwater baseflow. The latent partition gate is also not a water-mass partition. Synthetic data are included only as a reproducible software validation example, not as evidence for the research hypotheses.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python data/synthetic_generator.py --output data/synthetic_basin.npz
python training/train_cfsr.py --config config.yaml
```

The training script can auto-generate the synthetic dataset when it does not exist.

## Ablations

```bash
python experiments/experiment_1_baseline.py
python experiments/experiment_2_dual_memory.py
python experiments/experiment_3_routing.py
python experiments/experiment_4_gradient_conflict.py
python experiments/full_comparison.py
```

These scripts mirror the recommended development/ablation logic: standard LSTM; dual memory; partition/routing; and full conflict-aware CFSR-LSTM.

## Expected real-data NPZ format

The default loader expects one continuous basin series in an `.npz` file with:

- `dynamic`: `[time, dynamic_dim]`; column 0 must be precipitation for recession masking.
- `static`: `[static_dim]`.
- `q`: `[time]` observed total discharge.
- `event_id`: `[time]`, with `-1` outside training event windows and nonnegative integer IDs inside events.
- `baseflow`: optional `[time]`; if omitted, slow-flow training uses recession consistency/smoothness only.

For multi-basin studies, extend `HydroSequenceDataset` to index basin IDs and basin-specific static attributes. The core model already accepts a batch of static vectors.

## Training details

The full trainer partitions parameters as required by the methodology:

- `theta_sh`: encoder + partition gate + cross-memory communication.
- `theta_F`: fast LSTM + rapid-response head.
- `theta_S`: slow LSTM + slow-flow head + recession-coefficient head.
- `theta_R`: routing distribution head.

During specialization/fine-tuning:

- `g_A = grad_theta_sh(L_G + L_E)`.
- `g_S = grad_theta_sh(L_S)`.
- If `g_A^T g_S < 0`, only `g_A` is projected away from the opposing `g_S` direction.
- Shared update uses `g_sh = g_S + g_A_tilde`.
- Fast parameters use `J_F = L_G + L_E`.
- Routing parameters use `J_R = lambda_Q L_Q + L_E` (the log-flow term is intentionally excluded to match Eq. 90).
- Slow parameters use `J_S = L_G + L_S`.

## Reproducibility and validation

Set `seed` in `config.yaml`. The repository is designed to run on CPU or CUDA. Before scientific use, validate event-window construction, units, discharge normalization, time-step `dt`, maximum routing lag, slow-flow supervision, and hyperparameters for the target data set.
