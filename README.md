# Compute-Efficient Sensor-Aided Millimeter-Wave Beam Prediction for Real-Time Edge Deployment

Code, configs, frozen splits, trained checkpoints, and the on-device benchmark harness
for the paper of the same title.

**TL;DR.** A **0.59 M-parameter** camera+GPS *selective state-space* model predicts the
serving mmWave beam and runs its **full sensor-to-decision pipeline in real time —
49.5 ms / 20 Hz end-to-end (23.6 ms model inference) on a \$55 Raspberry Pi 4**, sustained
over a 150 s soak without thermal throttling. A latency-faithful reconstruction of a
recent SSM baseline's architecture needs **2.58 s per inference (~109× slower)** on the
same board. On a **leakage-free, vehicle-pass-disjoint split** the model reaches
**DBA 0.865**; GPS position alone reaches **0.853 at 3.7 ms**, and the top-1 beam retains
**98 % of optimal received power** despite a modest beam-index accuracy.

Every number in the paper maps to a config + seed + commit and can be regenerated with the
commands below.

---

## 1. Environment

```bash
git clone https://github.com/himansh24dev/compute-efficient-beam-prediction.git
cd compute-efficient-beam-prediction
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
Tested with Python 3.11–3.13, PyTorch ≥ 2.2 (CUDA optional — only training needs a GPU;
all evaluation below runs on CPU using the included checkpoints). On the Raspberry Pi only
`onnxruntime`, `numpy` and `Pillow` are needed (no PyTorch on-device).

## 2. Data

DeepSense 6G real-world V2I scenarios **31–34** (not redistributed here; obtain from
<https://deepsense6g.net>). Place the scenario folders at the repo root
(`Scenario31/`, …, `Scenario34/`), then build the memory-mapped cache once:

```bash
.venv/bin/python scripts/build_cache.py --scenarios 31 32 33 34
.venv/bin/python scripts/validate_data.py          # optional integrity check
```
The frozen split manifest (`data/derived/splits/v1.json`) and the operational definition
of a *pass* (a maximal contiguous same-sequence traversal, ≥4 frames) are versioned in the
repo, so the train/test partition is reproducible.

---

## 3. Reproduce the paper's results

### 3a. On your workstation (CPU unless noted)

Two trained checkpoints ship in `experiments/edge_deploy/results/deploy_ssm_w2_*_s1337.pt`,
so the evaluation results below reproduce **without retraining**.

| Paper item | Command | Notes |
|---|---|---|
| Deployed W=2 model (DBA 0.865) | `python -m experiments.edge_deploy.train_deploy --config configs/experiments/deploy_w2.yaml` | GPU; writes the two checkpoints |
| **Table 1** accuracy vs size + **Table 2** split leakage | `python -m experiments.bemamba_compare.run_compare` | GPU; window-random & episode-random |
| **Pass-level bootstrap CI** (0.865 ± 0.017) | `python -m experiments.edge_deploy.passlevel_ci` | CPU; uses the shipped checkpoint |
| **Communication metrics** (98 % / 99.6 % power retained) | `python -m experiments.edge_deploy.comm_metrics` | CPU; uses the DeepSense power vectors |
| **Delay-aligned / future-beam** (staleness penalty) | `python -m experiments.edge_deploy.future_beam_deploy --epochs 50` | GPU; trains t / t+1 / t+2 |
| **Beam-index staleness** over the latency | `python -m experiments.edge_deploy.beam_drift` | CPU; raw timestamps only |
| **Compute profile** (params / GMACs; 27× / 120–240×) | `python -m experiments.edge_deploy.config_audit` | CPU (fvcore) |
| GPS-only k-NN baseline (0.748) | `python -m experiments.edge_deploy.position_baseline` | CPU |
| **Ablations** (Table 4: cores / depth+fusion / modalities) | `python -m experiments.edge_deploy.ablate_cores` · `ablate_variants` · `ablate_modalities` | GPU |
| **Sensor robustness** (Fig. 4) | `python -m experiments.edge_deploy.robustness_full` | CPU; test-time perturbations |
| Multi-seed CI (SSM vs Transformer) | `python -m experiments.edge_deploy.multiseed --mode ci --cores ssm transformer --seeds 1337 2024 7 42 123 --protocol episode-random` | GPU |
| Regenerate all figures | `python -m experiments.edge_deploy.make_figs` | writes `figs/*.pdf` |

### 3b. On the Raspberry Pi 4 (on-device latency / thermal)

Export to ONNX on the workstation, copy `experiments/edge_deploy/artifacts/` to the Pi,
then run the harness there (`onnxruntime` + `numpy` + `Pillow` only):

```bash
# workstation: export fp32 + INT8 ONNX from the checkpoint, plus the GPS-only and replica models
.venv/bin/python -m experiments.edge_deploy.export_onnx
.venv/bin/python -m experiments.edge_deploy.export_gps_only
.venv/bin/python -m experiments.edge_deploy.bemamba_replica
```

| Paper item | Command (on the Pi) |
|---|---|
| Forward-pass latency vs window (**Table 3**) | `python bench_auto.py --model artifacts/deploy_w2/beam_ssm_fp32.onnx --window 2 --threads 4` |
| Thread scaling (**Fig. 2**) | `python bench_auto.py --model … --threads 1` (repeat for 1–4) |
| End-to-end pipeline (49.5 ms / 20 Hz) | `python bench_e2e.py --model artifacts/deploy_w2/beam_ssm_fp32.onnx --window 2 --threads 4 --runs 300 --frame 960x540` |
| Sustained thermal soak (150 s, 73.5 °C, no throttle) | `python soak_e2e.py --model artifacts/deploy_w2/beam_ssm_fp32.onnx --window 2 --threads 4 --seconds 150` |
| Per-operator profiling (98 % conv on the replica) | `python bench_profile.py --model artifacts/bemamba_replica.onnx --window 5 --threads 4` |
| GPS-only end-to-end (3.7 ms / 267 Hz) | `python bench_auto.py --model artifacts/gps_only/beam_gps_fp32.onnx --window 2 --threads 4` |
| INT8 quantization (12.8 / 18.8 ms) | `python bench_auto.py --model artifacts/deploy_w2/beam_ssm_int8.onnx --window 2 --threads 4` |

INT8 models are produced by `python -m experiments.edge_deploy.quantize_int8`.

---

## 4. Repository layout

```
models/                 selective-SSM core (+ Transformer/GRU/LSTM/MLP), gated fusion, encoders
data/                   DeepSense loaders, memmap cache, frozen-split logic, GPS/LiDAR/radar preprocessing
data/derived/splits/    frozen split manifest (versioned — defines the experimental protocol)
experiments/
  edge_deploy/          ONNX export, on-device benchmarks (latency/e2e/soak/profile), INT8,
                        ablations, robustness, pass-level CI, comm metrics, delay-aligned, replica
  bemamba_compare/      same-protocol accuracy comparison + BeMamba-architecture replica
  context_window/       input-window sweep (short effective memory)
configs/                YAML configs — every reported number maps to a config + seed + commit
scripts/                cache builder, data validation
```

## 5. Reproducibility notes

- **Determinism.** Seeds are fixed and logged (default 1337); GPS is standardized with
  train-only statistics; the ONNX export is verified numerically equivalent to PyTorch
  (max abs error < 5×10⁻⁶). The shipped checkpoint reproduces the exact headline
  DBA 0.865 via `passlevel_ci`.
- **Uncertainty.** The flagship number carries a **pass-level** bootstrap CI (resampling
  held-out passes, not frames), the honest interval for this correlated data.
- **On-device numbers** were measured on a passively cooled Raspberry Pi 4 Model B
  (BCM2711, quad Cortex-A72 @ 1.5 GHz, 4 GB), `onnxruntime` 1.28, performance governor.

## 6. Citation

```bibtex
@article{himanshu_beam_edge,
  title   = {Compute-Efficient Sensor-Aided Millimeter-Wave Beam Prediction for Real-Time Edge Deployment},
  author  = {Himanshu},
  year    = {2026},
  note    = {Manuscript; bibliographic details added on publication}
}
```

## 7. License

Released under the MIT License (see `LICENSE`). The DeepSense 6G dataset is subject to its
own terms; obtain it from <https://deepsense6g.net>.
