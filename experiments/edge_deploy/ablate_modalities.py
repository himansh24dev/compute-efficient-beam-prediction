"""Unimodal ablation: camera-only and GPS-only at W=2 (episode-random), to
quantify each modality's contribution and justify the cam+GPS fusion.
Reuses the tested trainer; compare against the cam+gps number from deploy_w2.
"""
from __future__ import annotations

from pathlib import Path

from data.config import load_data_config
from utils.device import get_device
from utils.yaml_io import load_yaml, save_json
from experiments.bemamba_compare.train_compare import train_one


def main():
    exp = load_yaml("configs/experiments/deploy_w2.yaml")
    cfg = load_data_config(exp["data_config"])
    device = get_device(cfg)
    seed = int(exp["sweep"]["seeds"][0]); W = int(exp["window"])
    out = Path(exp["results_dir"]); out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for mods in (["camera"], ["gps"]):
        name = "+".join(mods)
        print(f"\n=== ablate {name} | W={W} | episode-random ===", flush=True)
        r = train_one(cfg, exp, "ssm", mods, "episode-random", seed, device=device)
        t = r["test"]
        print(f"  RESULT {name}: top1={t['top1']:.4f} top3={t['top3']:.4f} "
              f"dba={t['dba']:.4f} params={r['params']:,}", flush=True)
        summary[name] = {"top1": float(t["top1"]), "top3": float(t["top3"]),
                         "top5": float(t["top5"]), "dba": float(t["dba"]),
                         "params": int(r["params"])}
    save_json(summary, out / "ablation_unimodal_w2.json")
    print("\nunimodal ablation done.", flush=True)


if __name__ == "__main__":
    main()
