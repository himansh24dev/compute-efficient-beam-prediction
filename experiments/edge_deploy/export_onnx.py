"""Stage 0 (edge deploy): export the trained cam+GPS SSM beam predictor to ONNX,
verify numerical equivalence vs PyTorch, and produce an INT8-quantized copy.

The ONNX file is what actually runs on the Raspberry Pi (via onnxruntime, which
has lightweight aarch64 wheels — no PyTorch needed on-device). Architecture dims
are read straight from the checkpoint so the rebuilt model always matches.

Usage:
  .venv/bin/python -m experiments.edge_deploy.export_onnx \
      --ckpt experiments/phase1/results/ckpt_e367fd32de97_gps+camera_ssm_s1337.pt \
      --window 8 --out experiments/edge_deploy/artifacts
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from models.beam_model import BeamModel


def infer_cfg(sd: dict) -> dict:
    """Read architecture dims from the state-dict tensors themselves."""
    dim = sd["head.weight"].shape[1]                       # 128
    num_beams = sd["head.weight"].shape[0]                 # 64
    A_log = sd["core.layers.0.mixer.A_log"]                # (d_inner, d_state)
    d_inner, d_state = A_log.shape
    d_conv = sd["core.layers.0.mixer.conv1d.weight"].shape[2]
    x_proj_rows = sd["core.layers.0.mixer.x_proj.weight"].shape[0]  # dt_rank + 2*d_state
    dt_rank = x_proj_rows - 2 * d_state
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("core.layers."))
    return {"dim": int(dim), "num_beams": int(num_beams), "d_state": int(d_state),
            "expand": int(d_inner // dim), "d_conv": int(d_conv), "dt_rank": int(dt_rank),
            "core_layers": int(n_layers)}


class ExportWrapper(nn.Module):
    """Tensor-in / tensor-out wrapper so ONNX gets named inputs (gps, camera)."""

    def __init__(self, model: BeamModel):
        super().__init__()
        self.model = model

    def forward(self, gps: torch.Tensor, camera: torch.Tensor) -> torch.Tensor:
        out = self.model({"gps": gps, "camera": camera})
        return out[0] if isinstance(out, tuple) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--window", type=int, default=8, help="input frames (checkpoint trained at 8)")
    ap.add_argument("--out", default="experiments/edge_deploy/artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    obj = torch.load(args.ckpt, map_location="cpu")
    sd = obj["model"]
    modalities = obj.get("modalities", ["gps", "camera"])
    cfg = infer_cfg(sd)
    print(f"checkpoint: {args.ckpt}")
    print(f"  modalities={modalities}  val={ {k: round(v,4) if isinstance(v,float) else v for k,v in obj.get('val',{}).items()} }")
    print(f"  inferred arch: {cfg}")

    model_cfg = {"dim": cfg["dim"], "core_layers": cfg["core_layers"],
                 "camera": {"width": 32}, "gps": {"in_dim": 3, "hidden": 128},
                 "ssm": {"d_state": cfg["d_state"], "d_conv": cfg["d_conv"],
                         "expand": cfg["expand"], "dt_rank": cfg["dt_rank"]}}
    model = BeamModel(modalities, cfg["num_beams"], "ssm", model_cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"state_dict mismatch: missing={missing} unexpected={unexpected}"
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  rebuilt + loaded OK: {n_params:,} params ({n_params/1e6:.2f}M)")

    wrap = ExportWrapper(model).eval()
    W = args.window
    gps = torch.randn(1, W, 3)
    camera = torch.randn(1, W, 3, 224, 224)
    with torch.no_grad():
        ref = wrap(gps, camera).numpy()

    fp32_path = os.path.join(args.out, "beam_ssm_fp32.onnx")
    torch.onnx.export(
        wrap, (gps, camera), fp32_path,
        input_names=["gps", "camera"], output_names=["logits"],
        dynamic_axes={"gps": {0: "batch"}, "camera": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"\nexported fp32 ONNX -> {fp32_path}  ({os.path.getsize(fp32_path)/1e6:.2f} MB)")

    # numerical equivalence check
    import onnxruntime as ort
    sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"gps": gps.numpy(), "camera": camera.numpy()})[0]
    max_err = float(np.abs(onnx_out - ref).max())
    top1_match = int(onnx_out.argmax()) == int(ref.argmax())
    print(f"  PyTorch vs ONNX: max_abs_err={max_err:.2e}  argmax_match={top1_match}")
    assert max_err < 1e-3 and top1_match, "ONNX output diverges from PyTorch!"

    # INT8 (dynamic) — shrinks weights ~4x; a quick first pass. Static/QDQ w/
    # calibration is the follow-up for max Conv speedup on ARM. Non-fatal: fp32
    # is the essential artifact; if dynamic quant trips on shape inference we
    # still ship fp32 and refine INT8 (static) later.
    int8_path = os.path.join(args.out, "beam_ssm_int8.onnx")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process
        prep_path = os.path.join(args.out, "_beam_ssm_prep.onnx")
        quant_pre_process(fp32_path, prep_path, skip_symbolic_shape=False)
        quantize_dynamic(prep_path, int8_path, weight_type=QuantType.QInt8)
        os.remove(prep_path)
        sess8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
        out8 = sess8.run(None, {"gps": gps.numpy(), "camera": camera.numpy()})[0]
        print(f"\nquantized INT8 (dynamic) -> {int8_path}  ({os.path.getsize(int8_path)/1e6:.2f} MB)"
              f"  argmax_match_vs_fp32={int(out8.argmax())==int(ref.argmax())}")
    except Exception as e:
        print(f"\n[INT8 dynamic quant skipped — {type(e).__name__}: {str(e)[:120]}]")
        print(" fp32 ONNX is ready; INT8 will be done via static/calibrated quant later.")

    print("\nStage-0 export OK. fp32 artifact in", args.out)


if __name__ == "__main__":
    main()
