"""Per-component parameters / MACs of the BeMamba-scale reconstruction (reviewer
comment 5). fvcore counts one fused multiply-add as one operation, so its totals are
MACs; element-wise ops (activations, the selective-scan update) are not counted.

  .venv/bin/python -m experiments.revision.replica_breakdown
"""
import json
import torch
from fvcore.nn import FlopCountAnalysis
from experiments.edge_deploy.bemamba_replica import BeMambaReplica

W = 5
m = BeMambaReplica(window=W).eval()
inp = (torch.randn(1, W, 3, 224, 224), torch.randn(1, W, 3),
       torch.randn(1, W, 3, 128, 128), torch.randn(1, W, 2, 128, 128))
fa = FlopCountAnalysis(m, inp); fa.unsupported_ops_warnings(False); fa.uncalled_modules_warnings(False)
bym = fa.by_module()
tot_p = sum(p.numel() for p in m.parameters()); tot_g = fa.total()
rows = []
for name in ("cam", "lidar", "radar", "gps", "fuse", "core", "head"):
    p = sum(x.numel() for x in getattr(m, name).parameters())
    rows.append({"component": name, "params": p, "params_pct": round(100 * p / tot_p, 2),
                 "gmacs": round(bym[name] / 1e9, 3), "gmacs_pct": round(100 * bym[name] / tot_g, 2)})
layers = []
x = torch.randn(1, 3, 224, 224)
for i, layer in enumerate(m.cam.net):
    x = layer(x); layers.append({"idx": i, "type": layer.__class__.__name__ if not isinstance(layer, torch.nn.Sequential) else "Conv3x3-BN-ReLU", "out_shape": list(x.shape)})
out = {"window": W, "total_params": tot_p, "total_gmacs": round(tot_g / 1e9, 3),
       "components": rows, "camera_encoder_layers": layers,
       "head": [str(l) for l in m.head], "ssm_layers": 4, "dim": 256,
       "inputs": {"camera": [W, 3, 224, 224], "gps": [W, 3], "lidar_bev": [W, 3, 128, 128], "radar_maps": [W, 2, 128, 128]}}
json.dump(out, open("experiments/revision/results/replica_breakdown.json", "w"), indent=1)
print(json.dumps(out, indent=1))
