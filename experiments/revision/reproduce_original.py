"""Sanity check before the revision re-run: the shipped deployment checkpoint,
evaluated on the ORIGINAL split with the ORIGINAL GPS statistics (fit on the day
train/val manifest, data/derived/splits/v1.json), must reproduce the submitted
DBA 0.865 / Top-3 0.809 on the 23 held-out passes (3,887 windows).

  .venv/bin/python -m experiments.revision.reproduce_original
"""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.config import load_data_config
from data.dataset import make_snapshot_dataset
from data.splits import resolve_splits
from data.preprocess.stats import fit_gps_stats
from models.beam_model import BeamModel
from utils.yaml_io import load_yaml
from experiments.phase0.train import _apply_modality_subset
from experiments.bemamba_compare.splits_compare import _all_episodes
from experiments.bemamba_compare.metrics_full import new_accumulator, add, finalize, full_metrics

exp = load_yaml("configs/experiments/deploy_w2.yaml")
cfg = load_data_config(exp["data_config"])
gps_stats = fit_gps_stats(cfg, resolve_splits(cfg).train)          # original stats (cached as gps_v1.json)
_apply_modality_subset(cfg, ["gps", "camera"])
cfg.snapshot["future_horizons"] = []; cfg.snapshot["window"] = 2; cfg.snapshot.pop("anchor_window", None)
eps = _all_episodes(cfg, exp["compare_scenarios"])
order = np.random.default_rng(1337).permutation(len(eps)); n_tr = int(round(0.8 * len(eps)))
test_eps = [eps[i] for i in order[n_tr:]]
ds = make_snapshot_dataset(cfg, test_eps, gps_stats, allow_heldout=True)
dev = torch.device("cuda")
ck = torch.load("experiments/edge_deploy/results/deploy_ssm_w2_episode-random_s1337.pt", map_location=dev)
m = BeamModel(["gps", "camera"], 64, "ssm", exp["model"]).to(dev); m.load_state_dict(ck["model"]); m.eval()
acc = new_accumulator()
with torch.no_grad():
    for b in DataLoader(ds, batch_size=256, num_workers=8):
        out = m({k: v.to(dev) for k, v in b["inputs"].items()})
        add(acc, full_metrics((out[0] if isinstance(out, tuple) else out).float(), b["beam"].to(dev)))
r = finalize(acc)
res = {"n_passes": len(test_eps), "n_windows": len(ds), "dba": r["dba"], "top1": r["top1"],
       "top3": r["top3"], "top5": r["top5"], "submitted": {"dba": 0.8652, "top1": 0.4350, "top3": 0.8091},
       "test_passes": sorted(e.key for e in test_eps)}
json.dump(res, open("experiments/revision/results/reproduce_original.json", "w"), indent=1)
print({k: v for k, v in res.items() if k != "test_passes"})
