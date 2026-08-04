"""Generate modern, refined publication figures with matplotlib.

Design: cohesive muted palette, light typography with weight *hierarchy* (bold
reserved for headline numbers only), soft fills, thin gridlines, no top/right
spines, subtle white marker halos. Saved as PDF (vector, for LaTeX) + PNG (view).
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUT = "figs"
os.makedirs(OUT, exist_ok=True)

# --- cohesive modern palette (muted, not neon) ---
INK   = "#264653"   # dark slate — primary text/axis
TEAL  = "#2A9D8F"   # "ours" / good
CORAL = "#E76F51"   # baseline / too-slow
SAND  = "#E9C46A"   # highlight / star
INDIGO= "#6D6AC9"   # secondary series
SLATE = "#5B6B73"   # secondary text
GRID  = "#E9EEF0"   # very light grid

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Nimbus Sans", "Liberation Sans"],
    "font.size": 11, "font.weight": "normal",
    "axes.edgecolor": INK, "axes.linewidth": 1.0,
    "text.color": INK, "axes.labelcolor": INK, "axes.labelweight": "normal",
    "xtick.color": SLATE, "ytick.color": SLATE,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": False,
})
HALO = [pe.withStroke(linewidth=3, foreground="white")]      # subtle text/marker halo


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK); ax.spines["bottom"].set_color(INK)
    ax.tick_params(labelsize=10, length=3, colors=SLATE)


def _callout(ax, xy, xytext, title, sub, color, fill, arrow=False, ha="left"):
    """Elegant two-tier callout: bold title line + light detail lines."""
    txt = r"$\bf{%s}$" % title.replace(" ", r"\ ") + "\n" + sub
    kw = dict(arrowprops=dict(arrowstyle="-", color=color, lw=1.1, alpha=0.7)) if arrow else {}
    ax.annotate(txt, xy=xy, xytext=xytext, fontsize=9.5, color=color, va="center", ha=ha,
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.4", fc=fill, ec=color,
                          lw=1.0, alpha=0.96), **kw)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _lead(ax, xy, xytext, text, color, ha="left"):
    """Compact leader-line annotation (academic style, no filled box)."""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=8.3, color=color, va="center", ha=ha,
                linespacing=1.3, path_effects=HALO,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.9, alpha=0.65,
                                shrinkA=2, shrinkB=4))


def deploy():
    fig, ax = plt.subplots(figsize=(5.9, 4.2))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.05, 75); ax.set_ylim(9, 6000)
    # zones + deadline (present enough that the figure does not read washed-out)
    ax.axhspan(9, 100, color=TEAL, alpha=0.10, zorder=0)
    ax.axhspan(100, 6000, color=CORAL, alpha=0.08, zorder=0)
    ax.axhline(100, color=SLATE, ls=(0, (6, 4)), lw=1.3, zorder=1)
    ax.text(64, 11.5, "real-time zone", color=TEAL, fontsize=8.5, va="bottom", ha="right",
            style="italic", alpha=0.7)
    ax.text(64, 5200, "too slow", color=CORAL, fontsize=8.5, va="top", ha="right",
            style="italic", alpha=0.7)
    ax.text(19, 108, "100 ms deadline", color=SLATE, fontsize=8.2, va="bottom", ha="right")
    # connecting line + speedup on it (halo so it sits cleanly over the line)
    ax.plot([0.097, 23.4], [23.6, 2580], ls=(0, (1, 1.8)), color="#AEBBC0", lw=1.7, zorder=1)
    ax.text(2.4, 430, r"$\approx$109$\times$", color=SLATE, fontsize=11.5, ha="center",
            va="center", fontweight="bold", path_effects=HALO, zorder=3)
    # baseline reports "23.4 GFLOPs", FLOP/MAC convention unstated: as MACs 23.4 GMACs,
    # as true FLOPs 11.7 GMACs -> show the range with an error bar, don't pick one.
    ax.errorbar([23.4], [2580], xerr=[[23.4 - 11.7], [0.0]], fmt="none",
                ecolor=CORAL, elinewidth=1.8, capsize=4, capthick=1.8, zorder=4)
    # both points as circles; the legend box (not in-figure text) carries the detail
    ax.scatter([0.097], [23.6], s=150, marker="o", color=TEAL, edgecolor="white",
               linewidth=1.6, zorder=6, path_effects=HALO,
               label="Ours\n0.59 M · 0.10 GMACs\n23.6 ms · 42 Hz")
    ax.scatter([23.4], [2580], s=150, marker="o", color=CORAL, edgecolor="white",
               linewidth=1.6, zorder=6, path_effects=HALO,
               label="BeMamba replica\n16.19 M · 12–23 GMACs\n2580 ms · 0.4 Hz")
    leg = ax.legend(loc="upper left", fontsize=8.4, frameon=True, framealpha=0.94,
                    edgecolor="#CDD6DA", borderpad=0.8, labelspacing=1.05,
                    handletextpad=0.8, borderaxespad=0.8)
    leg.get_frame().set_linewidth(0.9)
    ax.set_xlabel("compute per inference  (GMACs, log)", fontsize=10.5)
    ax.set_ylabel("Raspberry Pi 4 latency  (ms, log)", fontsize=10.5)
    ax.set_xticks([0.1, 1, 5, 20]); ax.set_xticklabels(["0.1", "1", "5", "20"])
    ax.set_yticks([10, 100, 1000]); ax.set_yticklabels(["10", "100", "1000"])
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    _clean(ax)
    save(fig, "deploy")


def context():
    W = [2, 4, 8, 16]; x = list(range(len(W)))
    ssm = [0.507, 0.491, 0.491, 0.489]; ssm_e = [0.006, 0.003, 0.003, 0.010]
    xf = [0.469, 0.444, 0.448, 0.470]; xf_e = [0.003, 0.003, 0.001, 0.004]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.axvspan(-0.32, 0.32, color=SAND, alpha=0.16, zorder=0)
    ax.errorbar(x, ssm, yerr=ssm_e, fmt="-o", color=TEAL, lw=2.2, ms=8, mec="white", mew=1.6,
                capsize=3.5, elinewidth=1.3, ecolor=TEAL, label="SSM (ours)", zorder=4)
    ax.errorbar(x, xf, yerr=xf_e, fmt="--s", color=INDIGO, lw=2, ms=7, mec="white", mew=1.3,
                capsize=3.5, elinewidth=1.1, ecolor=INDIGO, label="Transformer (matched)", zorder=3)
    ax.scatter([0], [0.507], s=300, marker="*", color=SAND, edgecolor="white", lw=1.4,
               zorder=6, path_effects=HALO)
    ax.annotate("best (W=2)", xy=(0, 0.507), xytext=(0.55, 0.523), fontsize=10,
                color="#B07D2B", ha="center",
                arrowprops=dict(arrowstyle="-", color="#B07D2B", lw=1.1, alpha=0.7))
    ax.set_xticks(x); ax.set_xticklabels(W)
    ax.set_ylim(0.43, 0.535)
    ax.set_xlabel("input window  W  (frames)", fontsize=10.5)
    ax.set_ylabel("Val. Top-1 accuracy", fontsize=10.5)
    ax.grid(True, axis="y", color=GRID, lw=0.9, zorder=0)
    ax.legend(fontsize=10, frameon=False, loc="upper right", handlelength=1.8)
    _clean(ax)
    save(fig, "context")


def threads():
    t = [1, 2, 3, 4]; lat = [42.2, 26.1, 24.5, 23.6]; hz = [1000 / v for v in lat]
    fig, ax = plt.subplots(figsize=(5.5, 3.9))
    ax.bar(t, lat, width=0.6, color=TEAL, alpha=0.85, edgecolor=INK, lw=0.8,
           hatch="////", zorder=3)
    for bi, (v, h) in enumerate(zip(lat, hz)):
        ax.text(t[bi], v + 1.0, r"$\bf{%.1f}$ ms" % v + f"\n{h:.0f} Hz", ha="center", va="bottom",
                fontsize=9, color=INK, linespacing=1.25)
    ax.set_ylim(0, 58)
    ax.set_xlim(0.45, 4.55)
    ax.set_xticks(t)
    ax.set_xlabel("CPU threads", fontsize=10.5)
    ax.set_ylabel("Pi 4 latency  (ms, p50)", fontsize=10.5)
    # speedup shown as a clean horizontal span high above every bar/label
    yspan = 51.0
    ax.annotate("", xy=(4, yspan), xytext=(1, yspan),
                arrowprops=dict(arrowstyle="<|-|>", color=SLATE, lw=1.1, alpha=0.85,
                                shrinkA=0, shrinkB=0))
    ax.text(2.5, yspan + 1.4, r"$1.8\times$ faster (1$\to$4 cores)", fontsize=9.2, color=SLATE,
            style="italic", ha="center", va="bottom")
    # single-core note in the empty band above the short bars, short arrow to the 1-thread bar
    ax.annotate("real-time even\non a single core", xy=(1.30, 40.5), xytext=(2.6, 35.5),
                fontsize=8.8, color="#1F7A6E", style="italic", ha="center", va="center",
                linespacing=1.2,
                arrowprops=dict(arrowstyle="->", color="#1F7A6E", lw=1.1, alpha=0.85,
                                connectionstyle="arc3,rad=0.22"))
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    _clean(ax)
    save(fig, "threads")


def pareto():
    p = [0.23, 0.39, 0.59, 1.15, 1.90]; dba = [0.857, 0.867, 0.865, 0.866, 0.870]
    fig, ax = plt.subplots(figsize=(5.7, 3.9))
    ax.set_xscale("log")
    ax.axvspan(0.6, 40, color=SAND, alpha=0.12, zorder=0)
    ax.text(2.4, 0.847, "over-parameterised", fontsize=9.5, color="#B07D2B", ha="center",
            style="italic")
    ax.axhline(0.872, ls=(0, (5, 5)), color="#AAB7BD", lw=1.1, zorder=1)
    ax.text(0.135, 0.8735, "accuracy ceiling", fontsize=9, color=SLATE, va="bottom")
    ax.plot(p, dba, "-o", color=TEAL, lw=2.2, ms=8, mec="white", mew=1.5, zorder=4)
    ax.scatter([0.59], [0.865], s=340, marker="*", color=SAND, edgecolor="white", lw=1.5,
               zorder=6, path_effects=HALO)
    ax.annotate("ours (deployed)", xy=(0.59, 0.865), xytext=(0.72, 0.851), fontsize=10,
                color="#B07D2B", arrowprops=dict(arrowstyle="-", color="#B07D2B", lw=1.1, alpha=0.7))
    ax.annotate("saturates by\n~0.4 M params", xy=(0.39, 0.867), xytext=(0.145, 0.884),
                fontsize=9.5, color=TEAL)
    ax.axvline(16, ls=(0, (1, 2)), color=CORAL, lw=1.6, zorder=3)
    ax.text(16, 0.8415, r"BeMamba" + "\n" + r"16 M (27$\times$)", color=CORAL, fontsize=9.5,
            ha="center", va="bottom")
    ax.set_xlim(0.13, 40); ax.set_ylim(0.84, 0.895)
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 16]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "16"])
    ax.set_xlabel("parameters  (millions, log)", fontsize=10.5)
    ax.set_ylabel("DBA  (honest split)", fontsize=10.5)
    ax.grid(True, axis="y", color=GRID, lw=0.9, zorder=0)
    _clean(ax)
    save(fig, "pareto")


def robustness():
    """Test-time sensor robustness: DBA collapses with GPS error but is nearly
    flat under camera degradation (the model is GPS-anchored, camera-robust)."""
    gps_m = [0, 1, 2, 5, 10, 20]
    gps_dba = [0.865, 0.808, 0.706, 0.474, 0.297, 0.173]
    fig, ax = plt.subplots(figsize=(5.9, 4.0))
    # shaded context bands (subtle in-plot; the legend box carries their meaning)
    ax.axvspan(1, 5, color=TEAL, alpha=0.10, zorder=0)
    ax.axhspan(0.775, 0.865, color=INDIGO, alpha=0.10, zorder=0)
    # GPS-error sensitivity curve — thin line, small clean markers, no glow
    ax.plot(gps_m, gps_dba, "-", color=CORAL, lw=1.7, zorder=3)
    ax.plot(gps_m, gps_dba, "o", color=CORAL, ms=5.0, mec="white", mew=0.8, zorder=4)
    # clean reference point (0 m) as a clean filled circle
    ax.plot([0], [0.865], "o", color=TEAL, ms=7.0, mec="white", mew=1.0, zorder=5)
    ax.set_xlim(-0.7, 20.7); ax.set_ylim(0.12, 0.95)
    ax.set_xlabel("GPS position error  (m)", fontsize=10.5)
    ax.set_ylabel("DBA  (pass-disjoint split)", fontsize=10.5)
    ax.set_xticks([0, 1, 2, 5, 10, 15, 20])
    ax.grid(True, axis="y", color=GRID, lw=0.9, zorder=0)
    # legend box (elements described here instead of in the plot area)
    handles = [
        Line2D([0], [0], color=CORAL, lw=1.7, marker="o", ms=5, mec="white", mew=0.8,
               label="DBA vs. GPS error"),
        Line2D([0], [0], color="none", marker="o", ms=7, markerfacecolor=TEAL,
               mec="white", mew=1.0, label="clean (no perturbation): 0.865"),
        mpatches.Patch(facecolor=TEAL, alpha=0.35, edgecolor="none",
                       label="consumer GPS (1–5 m)"),
        mpatches.Patch(facecolor=INDIGO, alpha=0.35, edgecolor="none",
                       label="camera blur / occlusion / noise (0.78–0.86)"),
    ]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=8.3, frameon=True,
                    framealpha=0.96, edgecolor="#CDD6DA", borderpad=0.8, labelspacing=0.85,
                    handletextpad=0.7, borderaxespad=0.7)
    leg.get_frame().set_linewidth(0.9)
    _clean(ax)
    save(fig, "robustness")


if __name__ == "__main__":
    deploy(); context(); threads(); pareto(); robustness()
    print("figures written to", OUT)
