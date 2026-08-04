"""Reviewer (Sci Rep, ChatGPT #8): DBA/Top-K are proxies over the beam INDEX. Here we
report the physical communication cost using the measured 64-beam received-power vectors:
for the deployed W=2 camera+GPS model on the pass-disjoint test set, the fraction of
optimal beamforming gain retained, the gain loss in dB, the achievable-rate (spectral-
efficiency) ratio at reference SNRs, and the outage rate -- for the top-1 predicted beam
and for a top-K mini-sweep (the system tries its K best guesses and keeps the strongest).

  .venv/bin/python -m experiments.edge_deploy.comm_metrics
"""
from __future__ import annotations

import numpy as np
import torch

from data.config import load_data_config
from data.index import build_sample_index
from data.dataset import make_snapshot_dataset
from data.paths import resolve_relative
from data.preprocess import beam as beam_pp
from data.preprocess.stats import load_gps_stats
from models.beam_model import BeamModel
from utils.device import autocast_context, get_device
from utils.yaml_io import load_yaml, save_json
from torch.utils.data import DataLoader

from experiments.phase0.train import _apply_modality_subset, _move
from experiments.bemamba_compare.splits_compare import _all_episodes

CKPT = "experiments/edge_deploy/results/deploy_ssm_w2_episode-random_s1337.pt"
SEED = 1337


def main():
    exp = load_yaml("configs/experiments/deploy_w2.yaml")
    cfg = load_data_config(exp["data_config"])
    device = get_device(cfg)
    cols = cfg.columns
    mods = ["gps", "camera"]
    _apply_modality_subset(cfg, mods)
    cfg.snapshot["future_horizons"] = []
    cfg.snapshot["window"] = 2
    cfg.snapshot.pop("anchor_window", None)
    gps_stats = load_gps_stats(cfg)

    eps = _all_episodes(cfg, list(exp["compare_scenarios"]))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(eps)); n_tr = int(round(0.8 * len(eps)))
    test_eps = [eps[i] for i in order[n_tr:]]

    ck = torch.load(CKPT, map_location=device)
    model = BeamModel(mods, int(cfg.beam["num_beams"]), "ssm", exp["model"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    P_best, P_top = [], {1: [], 3: [], 5: []}
    dfs = {}
    with torch.no_grad():
        for ep in test_eps:
            sid = ep.scenario_id
            if sid not in dfs:
                dfs[sid] = build_sample_index(cfg, sid, allow_heldout=True)
            df = dfs[sid]
            anchors = list(range(ep.row_start + 1, ep.row_end))       # window=2, stride=1, max_h=0
            ds = make_snapshot_dataset(cfg, [ep], gps_stats, allow_heldout=True)
            if len(ds) == 0:
                continue
            assert len(ds) == len(anchors), f"anchor mismatch {len(ds)} vs {len(anchors)}"
            dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=6)
            preds, targets = [], []
            for batch in dl:
                inp = _move(batch["inputs"], device, False)
                with autocast_context(cfg, device):
                    out = model(inp)
                    logits = (out[0] if isinstance(out, tuple) else out).float()
                preds.append(logits.topk(5, dim=-1).indices.cpu().numpy())
                targets.append(batch["beam"].numpy())
            preds = np.concatenate(preds); targets = np.concatenate(targets)
            for i, t in enumerate(anchors):
                pv = beam_pp.load_power_vector(resolve_relative(cfg, sid, df.iloc[t][cols["pwr"]]), 64)
                assert int(np.argmax(pv)) == int(targets[i]), "power/label misalignment"
                P_best.append(float(pv.max()))
                for k in (1, 3, 5):
                    P_top[k].append(float(pv[preds[i, :k]].max()))

    P_best = np.array(P_best)
    valid = P_best > 0                                     # drop frames with an all-zero/invalid power file
    n_all = len(P_best); P_best = P_best[valid]
    for k in (1, 3, 5):
        P_top[k] = np.array(P_top[k])[valid]
    n = len(P_best)
    res = {"n_frames": int(n), "n_dropped_invalid": int(n_all - n),
           "ckpt_dba": ck["val"]["dba"], "ckpt_top1": ck["val"]["top1"]}
    print(f"frames={n} (dropped {n_all-n} invalid)  (deployed DBA {ck['val']['dba']:.3f}, "
          f"Top-1 {ck['val']['top1']:.3f})\n")
    print(f"{'metric':34s} {'Top-1':>9s} {'Top-3':>9s} {'Top-5':>9s}")

    def row(name, fn):
        vals = [fn(np.array(P_top[k]) / P_best) for k in (1, 3, 5)]
        print(f"{name:34s} " + " ".join(f"{v:9.3f}" for v in vals))
        return {f"top{k}": round(float(v), 4) for k, v in zip((1, 3, 5), vals)}

    res["rel_power_mean"] = row("relative received power (mean)", lambda r: r.mean())
    res["rel_power_median"] = row("relative received power (median)", lambda r: np.median(r))
    res["gain_loss_db_mean"] = row("beamforming-gain loss dB (mean)", lambda r: (-10 * np.log10(r)).mean())
    res["gain_loss_db_p95"] = row("beamforming-gain loss dB (p95)", lambda r: np.percentile(-10 * np.log10(r), 95))
    res["outage_gt1dB"] = row("outage  P(gain loss > 1 dB)", lambda r: (-10 * np.log10(r) > 1).mean())
    res["outage_gt3dB"] = row("outage  P(gain loss > 3 dB)", lambda r: (-10 * np.log10(r) > 3).mean())

    print("\nachievable-rate (spectral-efficiency) ratio, best beam at reference SNR:")
    res["se_ratio"] = {}
    for snr_db in (0, 10, 20):
        rho = 10 ** (snr_db / 10.0)
        def se(r):  # rate(pred)/rate(best) with best-beam SNR = rho, others scaled by rel power
            return (np.log2(1 + rho * r) / np.log2(1 + rho)).mean()
        res["se_ratio"][f"{snr_db}dB"] = row(f"  SE ratio @ {snr_db} dB (mean)", se)

    save_json(res, "experiments/edge_deploy/results/comm_metrics.json")
    print("\nsaved -> experiments/edge_deploy/results/comm_metrics.json")


if __name__ == "__main__":
    raise SystemExit(main())
