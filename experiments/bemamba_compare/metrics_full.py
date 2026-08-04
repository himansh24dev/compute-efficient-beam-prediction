"""Full beam-prediction metrics, matching the DeepSense / BeMamba definitions so
our numbers are directly comparable to their Table II.

- Top-K accuracy for K in {1,2,3,5}.
- DBA score EXACTLY as BeMamba Eq. (16) / the ITU-2022 challenge: K=3, Delta=5.
    DBA = (1/K) * sum_k eta_k
    eta_k = 1 - mean_i min_{1<=j<=k} min(|beam_j(pred) - beam(gt)| / Delta, 1)
  i.e. cumulative-min over the top-j predicted beam indices of the clipped
  beam-distance, averaged over samples, averaged over k.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def full_metrics(logits: torch.Tensor, target: torch.Tensor,
                 ks=(1, 2, 3, 5), dba_k: int = 3, delta: float = 5.0) -> dict:
    """logits: (N, num_beams); target: (N,) 0-indexed. Returns dict of scalars
    (sums / counts are handled by the caller via accumulate())."""
    maxk = max(max(ks), dba_k)
    _, pred = logits.topk(maxk, dim=-1)                        # (N, maxk) sorted best-first
    hit = pred.eq(target.unsqueeze(-1))                        # (N, maxk)

    out = {}
    for k in ks:
        out[f"top{k}"] = hit[:, :k].any(dim=-1).float().sum().item()

    # DBA: clipped beam distance of each top-j prediction, cumulative min over j.
    dist = (pred[:, :dba_k].float() - target.unsqueeze(-1).float()).abs() / delta
    dist = dist.clamp(max=1.0)                                 # (N, dba_k)
    cummin = torch.cummin(dist, dim=1).values                 # (N, dba_k): min over top-1..j
    # store per-k cumulative-min sums; eta_k finalized in aggregate()
    for k in range(1, dba_k + 1):
        out[f"_dba_cmin_sum{k}"] = cummin[:, k - 1].sum().item()
    out["n"] = target.numel()
    out["_dba_k"] = dba_k
    return out


def new_accumulator(ks=(1, 2, 3, 5), dba_k=3) -> dict:
    acc = {f"top{k}": 0.0 for k in ks}
    for k in range(1, dba_k + 1):
        acc[f"_dba_cmin_sum{k}"] = 0.0
    acc["n"] = 0
    acc["_dba_k"] = dba_k
    return acc


def add(acc: dict, batch_metrics: dict) -> None:
    for k, v in batch_metrics.items():
        if k in ("_dba_k",):
            continue
        acc[k] = acc.get(k, 0.0) + v


def finalize(acc: dict, ks=(1, 2, 3, 5)) -> dict:
    n = max(acc["n"], 1)
    dba_k = acc["_dba_k"]
    res = {f"top{k}": acc[f"top{k}"] / n for k in ks}
    etas = [1.0 - acc[f"_dba_cmin_sum{k}"] / n for k in range(1, dba_k + 1)]
    res["dba"] = sum(etas) / len(etas)
    res["dba_per_k"] = [round(e, 4) for e in etas]
    res["n"] = int(acc["n"])
    return res
