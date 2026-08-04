"""Test-time robustness: the deployed model is trained with modality dropout, so
it should degrade gracefully when a sensor drops out. Evaluate the deployment
checkpoint on the honest test set with (a) both modalities, (b) camera zeroed
(GPS-only at test), (c) GPS zeroed (camera-only at test). Runs on CPU by default
so it does not contend with a GPU training sweep.
"""
from __future__ import annotations

import argparse

import torch

from data.config import load_data_config
from utils.yaml_io import load_yaml, save_json
from models.beam_model import BeamModel
from experiments.bemamba_compare.splits_compare import build_compare_loaders
from experiments.bemamba_compare.metrics_full import new_accumulator, add, finalize, full_metrics
from experiments.edge_deploy.export_onnx import infer_cfg


@torch.no_grad()
def eval_masked(model, te, device, drop):
    model.eval(); acc = new_accumulator()
    for batch in te:
        inp = {k: v.to(device) for k, v in batch["inputs"].items()}
        if drop:
            inp[drop] = torch.zeros_like(inp[drop])
        tgt = batch["beam"].to(device)
        logits = model(inp)
        logits = (logits[0] if isinstance(logits, tuple) else logits).float()
        add(acc, full_metrics(logits, tgt))
    return finalize(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/edge_deploy/results/deploy_ssm_w2_episode-random_s1337.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    exp = load_yaml("configs/experiments/deploy_w2.yaml")
    cfg = load_data_config(exp["data_config"])
    device = torch.device(args.device)
    obj = torch.load(args.ckpt, map_location=device)
    c = infer_cfg(obj["model"])
    mcfg = {"dim": c["dim"], "core_layers": c["core_layers"], "camera": {"width": 32},
            "gps": {"in_dim": 3, "hidden": 128},
            "ssm": {"d_state": c["d_state"], "d_conv": c["d_conv"],
                    "expand": c["expand"], "dt_rank": c["dt_rank"]}}
    model = BeamModel(["gps", "camera"], 64, "ssm", mcfg).to(device)
    model.load_state_dict(obj["model"])
    _, te, _ = build_compare_loaders(cfg, exp, ["gps", "camera"], "episode-random", 1337)

    out = {}
    for drop, name in [(None, "both"), ("camera", "camera_dropped"), ("gps", "gps_dropped")]:
        r = eval_masked(model, te, device, drop)
        out[name] = {"top1": r["top1"], "top3": r["top3"], "dba": r["dba"]}
        print(f"  {name:16s}: top1={r['top1']:.4f} top3={r['top3']:.4f} dba={r['dba']:.4f}", flush=True)
    save_json(out, "experiments/edge_deploy/results/robustness_missing_modality.json")
    print("robustness eval done.")


if __name__ == "__main__":
    main()
