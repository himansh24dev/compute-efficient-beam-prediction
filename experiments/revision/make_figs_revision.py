"""Revised figures (Scientific Reports revision): deploy.pdf and robustness.pdf.

deploy     : the reconstruction is plotted at its verified 23.41 GMACs (no x error bar:
             the FLOP/MAC ambiguity concerns the published budget, not our
             reconstruction); an independent ResNet-50 x 5-frame point and, when
             measured, a half-budget reconstruction point are added.
robustness : five checkpoints x ten realizations per stochastic perturbation, mean with
             95% band; three GPS error processes; every camera level reported.

  .venv/bin/python -m experiments.revision.make_figs_revision
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt

from experiments.edge_deploy.make_figs import (INK, TEAL, CORAL, SAND, INDIGO, SLATE, GRID,
                                               HALO, _clean)

OUT = "../manuscript_revised/figs"
RES = "experiments/revision/results"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{RES}/{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def deploy():
    pi = json.load(open(f"{RES}/pi/pi_summary.json"))
    info = json.load(open("experiments/revision/pi/bundle/models/models_info.json"))
    ours = pi["fwd_deploy_fp32_t4_control"]["p50"]          # same session as the reconstructions
    full = (info["replica_full"]["gmacs"], pi["replica_full_t4"]["p50"])
    half = (info["replica_half"]["gmacs"], pi["replica_half_t4"]["p50"])
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.05, 75); ax.set_ylim(9, 6000)
    ax.axhspan(9, 100, color=TEAL, alpha=0.10, zorder=0)
    ax.axhspan(100, 6000, color=CORAL, alpha=0.08, zorder=0)
    ax.axhline(100, color=SLATE, ls=(0, (6, 4)), lw=1.3, zorder=1)
    ax.text(0.06, 11.5, "real-time zone", color=TEAL, fontsize=8.5, va="bottom", ha="left",
            style="italic", alpha=0.7)
    ax.text(64, 5200, "too slow", color=CORAL, fontsize=8.5, va="top", ha="right",
            style="italic", alpha=0.7)
    ax.text(19, 108, "100 ms deadline", color=SLATE, fontsize=8.2, va="bottom", ha="right")
    ax.plot([0.097, full[0]], [ours, full[1]], ls=(0, (1, 1.8)), color="#AEBBC0", lw=1.7, zorder=1)
    ax.text(2.4, 430, r"$\approx$%d$\times$" % round(full[1] / ours), color=SLATE, fontsize=11.5, ha="center",
            va="center", fontweight="bold", path_effects=HALO, zorder=3)
    ax.scatter([0.097], [ours], s=150, marker="o", color=TEAL, edgecolor="white",
               linewidth=1.6, zorder=6, path_effects=HALO,
               label=f"Ours (measured model)\n0.59 M · 0.097 GMACs\n{ours:.1f} ms · {1000/ours:.0f} Hz")
    ax.scatter([full[0]], [full[1]], s=150, marker="o", color=CORAL, edgecolor="white",
               linewidth=1.6, zorder=6, path_effects=HALO,
               label=f"Reconstruction at the\npublished SOTA budget\n16.27 M · {full[0]:.2f} GMACs\n{full[1]:.0f} ms · {1000/full[1]:.2f} Hz")
    rn = pi["resnet50_5frames_t4"]["p50"]
    ax.scatter([20.56], [rn], s=90, marker="s", color=SLATE, edgecolor="white",
               linewidth=1.4, zorder=5, path_effects=HALO,
               label=f"ResNet-50 × 5 frames\n20.56 GMACs · {rn:.0f} ms")
    if half is not None:
        ax.scatter([half[0]], [half[1]], s=110, marker="D", color=SAND, edgecolor="white",
                   linewidth=1.4, zorder=5, path_effects=HALO,
                   label=f"Reconstruction at half budget\n(\"23.4 GFLOPs\" as true FLOPs)\n{half[0]:.2f} GMACs · {half[1]:.0f} ms")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.6, frameon=True, framealpha=0.94,
                    edgecolor="#CDD6DA", borderpad=0.7, labelspacing=0.9,
                    handletextpad=0.8, borderaxespad=0.7)
    leg.get_frame().set_linewidth(0.9)
    ax.set_xlabel("compute per inference  (GMACs, log)", fontsize=10.5)
    ax.set_ylabel("Raspberry Pi 4 forward-pass latency  (ms, log)", fontsize=10)
    ax.set_xticks([0.1, 1, 5, 20]); ax.set_xticklabels(["0.1", "1", "5", "20"])
    ax.set_yticks([10, 100, 1000]); ax.set_yticklabels(["10", "100", "1000"])
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    _clean(ax)
    save(fig, "deploy")


def threads():
    pi = json.load(open(f"{RES}/pi/pi_summary.json"))
    t = [1, 2, 3, 4]; lat = [pi[f"fwd_deploy_fp32_t{k}"]["p50"] for k in t]; hz = [1000 / v for v in lat]
    fig, ax = plt.subplots(figsize=(5.5, 3.9))
    ax.bar(t, lat, width=0.6, color=TEAL, alpha=0.85, edgecolor=INK, lw=0.8, hatch="////", zorder=3)
    for bi, (v, h) in enumerate(zip(lat, hz)):
        ax.text(t[bi], v + 1.0, r"$\bf{%.1f}$ ms" % v + f"\n{h:.0f} Hz", ha="center", va="bottom",
                fontsize=9, color=INK, linespacing=1.25)
    ax.set_ylim(0, 58); ax.set_xlim(0.45, 4.55); ax.set_xticks(t)
    ax.set_xlabel("CPU threads", fontsize=10.5); ax.set_ylabel("Pi 4 forward pass  (ms, p50)", fontsize=10.5)
    yspan = 51.0
    ax.annotate("", xy=(4, yspan), xytext=(1, yspan), arrowprops=dict(arrowstyle="<|-|>", color=SLATE,
                lw=1.1, alpha=0.85, shrinkA=0, shrinkB=0))
    ax.text(2.5, yspan + 1.4, r"$%.1f\times$ faster (1$\to$4 cores)" % (lat[0] / lat[-1]), fontsize=9.2,
            color=SLATE, style="italic", ha="center", va="bottom")
    ax.annotate("real-time even\non a single core", xy=(1.30, lat[0] - 1.5), xytext=(2.6, 35.5), fontsize=8.8,
                color="#1F7A6E", style="italic", ha="center", va="center", linespacing=1.2,
                arrowprops=dict(arrowstyle="->", color="#1F7A6E", lw=1.1, alpha=0.85, connectionstyle="arc3,rad=0.22"))
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    _clean(ax)
    save(fig, "threads")


def robustness():
    r = json.load(open(f"{RES}/robustness.json"))
    clean = r["clean"][0]["dba"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    styles = {"gps_iid_window": ("independent per window", CORAL, "o", "-"),
              "gps_gauss_markov": (r"Gauss–Markov, $\tau$ = 10 s", INDIGO, "s", "--"),
              "gps_pass_bias": ("constant bias per pass", INK, "^", ":")}
    for k, (lab, col, mk, ls) in styles.items():
        rows = sorted(r[k], key=lambda x: x["level"])
        x = [0] + [e["level"] for e in rows]
        y = [clean] + [e["dba"] for e in rows]
        lo = [clean] + [e["dba_lo"] for e in rows]; hi = [clean] + [e["dba_hi"] for e in rows]
        ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0)
        ax.plot(x, y, ls=ls, marker=mk, ms=4.5, color=col, lw=1.6, label=lab)
    ax.set_xlabel(r"GPS error, per-axis std $\sigma$ (m)")
    ax.set_ylabel("DBA (pass-disjoint test)")
    ax.set_xticks([0, 1, 2, 3, 5, 10, 20]); ax.set_ylim(0, 0.95)
    ax.grid(True, color=GRID, lw=0.8); ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("GPS position error", fontsize=10.5)
    _clean(ax)

    ax = axes[1]
    fams = [("cam_blur", "blur k", lambda v: f"{int(v)}"),
            ("cam_occlude", "occlusion p", lambda v: f"{v:g}"),
            ("cam_gauss", r"noise $\sigma$", lambda v: f"{v:g}"),
            ("cam_bright", "brightness Δ", lambda v: f"{v:+g}")]
    labels, vals, los, his, cols = [], [], [], [], []
    palette = {"cam_blur": TEAL, "cam_occlude": INDIGO, "cam_gauss": CORAL, "cam_bright": SAND}
    for k, name, fmt in fams:
        for e in r[k]:
            labels.append(f"{name} {fmt(e['level'])}"); vals.append(e["dba"])
            los.append(e["dba"] - e["dba_lo"]); his.append(e["dba_hi"] - e["dba"])
            cols.append(palette[k])
    yy = list(range(len(vals)))[::-1]
    ax.barh(yy, vals, color=cols, alpha=0.85, height=0.7,
            xerr=[los, his], error_kw={"elinewidth": 0.9, "capsize": 2, "ecolor": SLATE})
    ax.axvline(clean, color=INK, ls=(0, (4, 3)), lw=1.1)
    ax.text(clean + 0.005, len(vals) - 0.3, f"clean\n{clean:.3f}", fontsize=7.4, ha="left", va="bottom", color=INK)
    a = json.load(open(f"{RES}/analysis.json"))["configs"]["ssm_gps_only"]["dba"]["mean"]
    ax.axvline(a, color=CORAL, ls=(0, (1, 1.5)), lw=1.3)
    ax.text(a - 0.005, -1.15, f"GPS-only\nmodel {a:.3f}", fontsize=7.4, ha="right", va="bottom", color=CORAL)
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=7.6)
    ax.set_xlim(0, 0.98); ax.set_ylim(-1.3, len(vals) - 0.3 + 0.9); ax.set_xlabel("DBA (pass-disjoint test)")
    ax.set_title("Camera degradation", fontsize=10.5)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    _clean(ax)
    fig.tight_layout()
    save(fig, "robustness")


if __name__ == "__main__":
    deploy()
    threads()
    if os.path.isfile(f"{RES}/robustness.json"):
        robustness()
