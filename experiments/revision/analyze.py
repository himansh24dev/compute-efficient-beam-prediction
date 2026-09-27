"""Revision analysis: every accuracy table + its uncertainty, from saved predictions.

Pass-level bootstrap (reviewer comments 3, 8). The held-out vehicle PASS is the
independent unit, so all intervals resample whole test passes with replacement
(B = 5000). For a configuration trained with five initialisations on the same
split, each bootstrap replicate averages the metric over the five seeds, so the
interval covers pass sampling while the point estimate is the five-seed mean.

Paired differences (comment 3). For two configurations A, B evaluated on the SAME
held-out passes, each replicate draws one set of passes and computes
mean_seeds(metric_A) - mean_seeds(metric_B) on it; the 2.5/97.5 percentiles give
the 95% CI of the difference, and the two-sided bootstrap p-value is
2 * min(P(diff <= 0), P(diff >= 0)).

  .venv/bin/python -m experiments.revision.analyze
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path("experiments/revision/results")
RUNS = ROOT / "runs"
DELTA, K_DBA, B = 5.0, 3, 5000
SEEDS = [1337, 2024, 7, 42, 123]


# ---------------------------------------------------------------- per-sample stats
def sample_stats(prob: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(N, 5) columns: dba_loss, hit1, hit2, hit3, hit5. DBA = 1 - mean(dba_loss)."""
    top = np.argsort(-prob.astype(np.float32), axis=1)[:, :5]
    t = target[:, None]
    hit = top == t
    dist = np.clip(np.abs(top[:, :K_DBA] - t) / DELTA, None, 1.0)
    cmin = np.minimum.accumulate(dist, axis=1)
    loss = cmin.mean(axis=1)                                   # (1/K) sum_k cummin_k
    return np.stack([loss, hit[:, :1].any(1), hit[:, :2].any(1),
                     hit[:, :3].any(1), hit[:, :5].any(1)], axis=1).astype(np.float64)


METRICS = ["dba", "top1", "top2", "top3", "top5"]


def to_metrics(sum_stats: np.ndarray, n: np.ndarray) -> np.ndarray:
    """sum_stats (..., 5), n (...,) -> metrics (..., 5) in METRICS order."""
    m = sum_stats / n[..., None]
    m[..., 0] = 1.0 - m[..., 0]
    return m


VARIANT = ""                      # "" = validation-selected checkpoint, "__last" = final epoch


def load_run(name: str):
    z = np.load(RUNS / f"{name}{VARIANT}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


_BEAM = {}


def future_target(run, e: int, h_saved: int = 0) -> np.ndarray:
    """Beam label e frames after each test anchor row. The run's saved target is the
    beam h_saved frames ahead (its training horizon); that is checked against the
    cached labels before returning the requested horizon."""
    from data.config import load_data_config
    from data.cache import scenario_cache_dir
    cfg = load_data_config("configs/data.yaml")
    out = np.empty_like(run["target"])
    for sid in np.unique(run["scenario_id"]):
        if sid not in _BEAM:
            _BEAM[sid] = np.load(scenario_cache_dir(cfg, int(sid)) / "beam.npy")
        m = run["scenario_id"] == sid
        rows = run["row"][m]
        assert np.array_equal(_BEAM[sid][rows + h_saved], run["target"][m]), "label/cache mismatch"
        out[m] = _BEAM[sid][rows + e]
    return out


def per_pass(run, pass_order=None):
    """-> (pass_keys, sums (P,5), counts (P,))."""
    st = sample_stats(run["prob"], run["target"])
    keys = run["passes"]
    order = pass_order if pass_order is not None else sorted(set(keys.tolist()))
    idx = {k: i for i, k in enumerate(order)}
    sums = np.zeros((len(order), 5)); cnt = np.zeros(len(order))
    pi = np.fromiter((idx[k] for k in keys), dtype=np.int64, count=len(keys))
    np.add.at(sums, pi, st); np.add.at(cnt, pi, 1)
    return order, sums, cnt


def boot_weights(n_pass: int, seed: int = 0) -> np.ndarray:
    """(B, P) multiplicity of each pass in each bootstrap replicate."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_pass, size=(B, n_pass))
    W = np.zeros((B, n_pass))
    np.add.at(W, (np.repeat(np.arange(B), n_pass), draws.ravel()), 1)
    return W


def group_boot(names: list[str], W: np.ndarray, order):
    """Seed-averaged metrics: point (5,), replicates (B,5), per-seed points (S,5)."""
    pts, reps = [], []
    for nm in names:
        _, s, c = per_pass(load_run(nm), order)
        pts.append(to_metrics(s.sum(0), np.array(c.sum())))
        reps.append(to_metrics(W @ s, W @ c))
    return np.mean(pts, 0), np.mean(reps, 0), np.array(pts)


def ci(reps: np.ndarray):
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    return lo, hi


def summarize(point, reps, per_seed=None):
    lo, hi = ci(reps)
    out = OrderedDict()
    for i, m in enumerate(METRICS):
        d = {"mean": round(float(point[i]), 4), "lo": round(float(lo[i]), 4),
             "hi": round(float(hi[i]), 4), "half": round(float((hi[i] - lo[i]) / 2), 4)}
        if per_seed is not None and len(per_seed) > 1:
            d["seed_sd"] = round(float(per_seed[:, i].std(ddof=1)), 4)
            d["per_seed"] = [round(float(x), 4) for x in per_seed[:, i]]
        out[m] = d
    return out


def paired(a_names, b_names, W, order):
    pa, ra, sa = group_boot(a_names, W, order)
    pb, rb, sb = group_boot(b_names, W, order)
    diff = ra - rb
    lo, hi = ci(diff)
    out = OrderedDict()
    for i, m in enumerate(METRICS):
        p = 2 * min((diff[:, i] <= 0).mean(), (diff[:, i] >= 0).mean())
        wins = int((sa[:, i] > sb[:, i]).sum()) if len(sa) == len(sb) else None
        out[m] = {"diff": round(float(pa[i] - pb[i]), 4), "lo": round(float(lo[i]), 4),
                  "hi": round(float(hi[i]), 4), "p_boot": round(float(min(p, 1.0)), 4),
                  "seed_wins_A": wins, "n_seeds": len(sa)}
    return out


def ensemble(names, W, order):
    runs = [load_run(n) for n in names]
    for r in runs[1:]:
        assert np.array_equal(r["row"], runs[0]["row"]), "ensemble members must share the test set"
    prob = np.mean([r["prob"].astype(np.float32) for r in runs], axis=0)
    ens = {**runs[0], "prob": prob}
    _, s, c = per_pass(ens, order)
    return summarize(to_metrics(s.sum(0), np.array(c.sum())), to_metrics(W @ s, W @ c))


def names(prefix):
    return [f"{prefix}_i{s}" for s in SEEDS if (RUNS / f"{prefix}_i{s}{VARIANT}.npz").is_file()]


def main(variant: str = ""):
    global VARIANT
    VARIANT = variant
    res = OrderedDict()
    res["variant"] = "final-epoch model (no selection)" if variant else "validation-selected checkpoint"
    ref = load_run("A_ssm_2M_i1337")
    order = sorted(set(ref["passes"].tolist()))
    W = boot_weights(len(order))
    res["test_set"] = {"n_passes": len(order), "n_windows": int(len(ref["target"])),
                       "bootstrap_B": B}

    # --- configurations on the headline split (five inits each) ---
    configs = OrderedDict([
        ("ssm_cam+gps (deployed)", "A_ssm_2M"), ("ssm_gps_only", "A_ssm_gps"),
        ("ssm_camera_only", "A_ssm_cam"), ("transformer", "A_transformer_2M"),
        ("gru", "A_gru_2M"), ("lstm", "A_lstm_2M"), ("mlp", "A_mlp_2M"),
        ("ssm_+lidar", "B_ssm_3M"), ("ssm_+lidar+radar", "B_ssm_4M"),
        ("ssm_L2", "B_ssm_L2"), ("ssm_L6", "B_ssm_L6"),
        ("fusion_mean", "B_fuse_mean"), ("fusion_concat", "B_fuse_concat"),
        ("pretrained_encoder", "B_pretrained"),
        ("int8_whole", "Q_int8_whole"), ("int8_selective", "Q_int8_selective")])
    res["configs"] = OrderedDict()
    for label, pref in configs.items():
        nm = names(pref)
        if not nm:
            continue
        pt, rp, ps = group_boot(nm, W, order)
        res["configs"][label] = {"n_seeds": len(nm), **summarize(pt, rp, ps)}
    # single deployed checkpoint (init 1337), the one exported to ONNX
    pt, rp, _ = group_boot(["A_ssm_2M_i1337"], W, order)
    res["deployed_single_seed1337"] = summarize(pt, rp)

    # --- paired differences on the same held-out passes (reviewer comment 3) ---
    base = names("A_ssm_2M")
    res["paired_vs_deployed"] = OrderedDict()
    for label, pref in configs.items():
        if pref == "A_ssm_2M" or not names(pref):
            continue
        res["paired_vs_deployed"][label] = paired(base, names(pref), W, order)

    # --- five-seed ensemble on the pass-disjoint split ---
    if len(base) == 5:
        res["ensemble_5seed"] = ensemble(base, W, order)

    # --- delay-aligned matrix (one fixed anchor set) ---
    d = OrderedDict()
    for h in (0, 1, 2):
        nm = names(f"D_h{h}")
        if nm:
            _, dorder = None, sorted(set(load_run(nm[0])["passes"].tolist()))
            Wd = boot_weights(len(dorder))
            pt, rp, ps = group_boot(nm, Wd, dorder)
            d[f"trained_t+{h}"] = {"n_seeds": len(nm), **summarize(pt, rp, ps)}
    res["delay_aligned_selfscore"] = d
    mat = OrderedDict()
    for h in (0, 1, 2):
        nm = names(f"D_h{h}")
        if not nm:
            continue
        dorder = sorted(set(load_run(nm[0])["passes"].tolist()))
        Wd = boot_weights(len(dorder))
        row = OrderedDict()
        for e in (0, 1, 2):
            pts, reps = [], []
            for n_ in nm:
                r0 = load_run(n_)
                r = {**r0, "target": future_target(r0, e, h_saved=h)}
                _, s_, c_ = per_pass(r, dorder)
                pts.append(to_metrics(s_.sum(0), np.array(c_.sum()))); reps.append(to_metrics(Wd @ s_, Wd @ c_))
            row[f"eval_t+{e}"] = summarize(np.mean(pts, 0), np.mean(reps, 0))
        mat[f"trained_t+{h}"] = row
    res["delay_aligned_matrix"] = mat

    # --- window sweep (controlled sample set) ---
    ws = OrderedDict()
    for Wn in (2, 4, 8, 16):
        nm = [f"E_w{Wn}_i{s}" for s in SEEDS[:3] if (RUNS / f"E_w{Wn}_i{s}.npz").is_file()]
        if nm:
            eorder = sorted(set(load_run(nm[0])["passes"].tolist()))
            We = boot_weights(len(eorder))
            pt, rp, ps = group_boot(nm, We, eorder)
            ws[f"W={Wn}"] = {"n_seeds": len(nm), **summarize(pt, rp, ps)}
    res["window_sweep"] = ws

    # --- split-protocol leakage (seed drives split + init; unpaired across protocols) ---
    leak = OrderedDict()
    for Wn in (2, 5):
        for p in ("window-random", "episode-random"):
            vals = []
            for s in SEEDS:
                nm = f"C_w{Wn}_{p}_s{s}"
                if Wn == 2 and p == "episode-random" and s == 1337:
                    nm = "A_ssm_2M_i1337"
                f = RUNS / f"{nm}.json"
                if f.is_file():
                    t = json.load(open(f))["test_last_epoch" if VARIANT else "test"]
                    vals.append([t["dba"], t["top3"]])
            if vals:
                v = np.array(vals)
                leak[f"W{Wn}_{p}"] = {"n_seeds": len(v), "dba": round(float(v[:, 0].mean()), 4),
                                      "dba_sd": round(float(v[:, 0].std(ddof=1)), 4) if len(v) > 1 else None,
                                      "top3": round(float(v[:, 1].mean()), 4),
                                      "top3_sd": round(float(v[:, 1].std(ddof=1)), 4) if len(v) > 1 else None,
                                      "per_seed_dba": [round(float(x), 4) for x in v[:, 0]]}
    from scipy import stats as _st
    for Wn in (2, 5):
        a, b = leak.get(f"W{Wn}_window-random"), leak.get(f"W{Wn}_episode-random")
        if a and b and a["n_seeds"] == b["n_seeds"] == 5:
            wr = np.array([json.load(open(RUNS / f"C_w{Wn}_window-random_s{s}.json"))["test_last_epoch" if VARIANT else "test"]["dba"] for s in SEEDS])
            er = []
            for s in SEEDS:
                nm = "A_ssm_2M_i1337" if (Wn == 2 and s == 1337) else f"C_w{Wn}_episode-random_s{s}"
                er.append(json.load(open(RUNS / f"{nm}.json"))["test_last_epoch" if VARIANT else "test"]["dba"])
            d = wr - np.array(er); h = _st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            leak[f"W{Wn}_inflation_dba"] = {"mean": round(float(d.mean()), 4), "t95_lo": round(float(d.mean() - h), 4),
                                            "t95_hi": round(float(d.mean() + h), 4), "per_seed": [round(float(x), 4) for x in d]}
    res["leakage"] = leak
    t1 = OrderedDict()
    for m in ("3M", "4M"):
        v = [json.load(open(RUNS / f"C_w5_window-random_{m}_s{s}.json"))["test"]
             for s in SEEDS if (RUNS / f"C_w5_window-random_{m}_s{s}.json").is_file()]
        if v:
            t1[m] = {"n_seeds": len(v), "dba": round(float(np.mean([x["dba"] for x in v])), 4),
                     "top3": round(float(np.mean([x["top3"] for x in v])), 4)}
    res["table1_window_random_w5"] = t1

    (ROOT).mkdir(parents=True, exist_ok=True)
    out = ROOT / ("analysis_last.json" if variant else "analysis.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"saved -> {out}")
    return res


if __name__ == "__main__":
    import sys
    main("__last" if "--last" in sys.argv else "")
