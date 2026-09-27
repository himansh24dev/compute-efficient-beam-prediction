"""Communication-level metrics (Table 4) from the saved test predictions of the five
validation-selected camera+GPS models, using the measured 64-beam received-power
vectors. Same definitions as experiments/edge_deploy/comm_metrics.py; point values are
five-seed means and intervals are 95% pass-level bootstrap intervals (B = 5000).

  .venv/bin/python -m experiments.revision.comm_metrics
"""
from __future__ import annotations

import json

import numpy as np

from data.config import load_data_config
from data.index import build_sample_index
from data.paths import resolve_relative
from data.preprocess import beam as beam_pp
from experiments.revision.analyze import RUNS, SEEDS, boot_weights, load_run

OUT = "experiments/revision/results/comm_metrics.json"


def main():
    cfg = load_data_config("configs/data.yaml")
    col = cfg.columns["pwr"]
    ref = load_run("A_ssm_2M_i1337")
    dfs, P = {}, []
    for sid, row, tgt in zip(ref["scenario_id"], ref["row"], ref["target"]):
        sid = int(sid)
        if sid not in dfs:
            dfs[sid] = build_sample_index(cfg, sid, allow_heldout=True)
        pv = beam_pp.load_power_vector(resolve_relative(cfg, sid, dfs[sid].iloc[int(row)][col]), 64)
        P.append(pv)
    P = np.asarray(P, dtype=np.float64)
    best = P.max(1)
    valid = best > 0
    assert np.all(P[valid].argmax(1) == ref["target"][valid]), "power/label misalignment"
    order = sorted(set(ref["passes"].tolist())); pidx = {k: i for i, k in enumerate(order)}
    pi = np.array([pidx[k] for k in ref["passes"]])[valid]
    W = boot_weights(len(order))

    def per_pass_mean(x):
        s = np.zeros(len(order)); c = np.zeros(len(order))
        np.add.at(s, pi, x); np.add.at(c, pi, 1)
        return s, c

    names = [f"A_ssm_2M_i{s}" for s in SEEDS if (RUNS / f"A_ssm_2M_i{s}.npz").is_file()]
    stats = {}
    for n in names:
        r = load_run(n)
        assert np.array_equal(r["row"], ref["row"])
        top = np.argsort(-r["prob"].astype(np.float32), axis=1)[:, :5][valid]
        for k in (1, 3, 5):
            rel = np.take_along_axis(P[valid], top[:, :k], 1).max(1) / best[valid]
            loss = -10 * np.log10(np.clip(rel, 1e-12, None))
            q = {"rel_power_mean": rel, "gain_loss_db_mean": loss,
                 "outage_gt1dB": (loss > 1).astype(float), "outage_gt3dB": (loss > 3).astype(float)}
            for snr in (0, 10, 20):
                rho = 10 ** (snr / 10)
                q[f"se_ratio_{snr}dB"] = np.log2(1 + rho * rel) / np.log2(1 + rho)
            for m, x in q.items():
                s_, c_ = per_pass_mean(x)
                stats.setdefault((m, k), []).append((s_.sum() / c_.sum(), (W @ s_) / (W @ c_)))
            stats.setdefault(("rel_power_median", k), []).append((float(np.median(rel)), None))
    res = {"n_models": len(names), "n_windows": int(valid.sum()),
           "n_dropped_invalid_power": int((~valid).sum()), "metrics": {}}
    for (m, k), v in stats.items():
        pt = float(np.mean([a for a, _ in v]))
        d = {"mean": round(pt, 4)}
        if v[0][1] is not None:
            reps = np.mean([b for _, b in v], 0)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            d.update(lo=round(float(lo), 4), hi=round(float(hi), 4))
        res["metrics"].setdefault(m, {})[f"top{k}"] = d
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
