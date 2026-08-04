"""Draw the schematic diagrams (system scenario, architecture) in matplotlib and
save each as a standalone vector PDF + PNG, for \\includegraphics in the paper.

Design goals (learned the hard way): FLAT modern cards — no drop shadows, thin
soft borders; every edge/shape label carries its own white background so no arrow
or box ever shows *through* the text; muted palette shared with the charts.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

OUT = "paper/latex/figs"
os.makedirs(OUT, exist_ok=True)

INK   = "#264653"; TEAL = "#2A9D8F"; CORAL = "#E76F51"; SAND = "#C99A2E"
INDIGO= "#6D6AC9"; SLATE = "#66757C"
fTEAL = "#E7F4F0"; fSAND = "#FBF2DC"; fIND = "#EEEDF9"; fCOR = "#FCEBE4"; fGRY = "#F0F4F5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Nimbus Sans", "Liberation Sans"],
    "font.size": 11, "text.color": INK,
})


def _save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", pad_inches=0.03, dpi=220)
    plt.close(fig)


def card(ax, cx, cy, w, h, text, ec, fc, fs=7.5, tc=INK, dashed=False, lw=0.9):
    """Flat rounded card — no shadow, thin border."""
    p = mp.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0,rounding_size=0.08",
                          linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3,
                          linestyle=(0, (4, 2)) if dashed else "solid")
    ax.add_patch(p)
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
                zorder=4, linespacing=1.28)
    return cx, cy, w, h


def arrow(ax, p1, p2, color=SLATE, lw=1.1, dashed=False, rad=0.0, alpha=1.0):
    ax.annotate("", xy=p2, xytext=p1, zorder=2,
                arrowprops=dict(arrowstyle="-|>", mutation_scale=10, color=color, lw=lw,
                                alpha=alpha, linestyle=(0, (4, 2)) if dashed else "solid",
                                shrinkA=3, shrinkB=3, connectionstyle=f"arc3,rad={rad}"))


def cap(ax, x, y, s, fs=6.5, color=None, ha="center"):
    """Plain caption text (icon labels, panel notes) — no background."""
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color or SLATE, zorder=5)


def edge(ax, x, y, s, fs=6.3, color=INK):
    """Edge / shape label that punches a clean white hole so no line shows through."""
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=1.0))


# ------------------------------- icons -------------------------------
def cam_icon(ax, cx, cy):
    ax.add_patch(mp.FancyBboxPatch((cx - 0.42, cy - 0.3), 0.84, 0.6,
                 boxstyle="round,pad=0,rounding_size=0.06", fc=fGRY, ec=INK + "55", lw=0.7, zorder=3))
    ax.add_patch(mp.Rectangle((cx - 0.4, cy + 0.0), 0.8, 0.28, fc=TEAL, alpha=0.32, zorder=3))
    ax.add_patch(mp.Rectangle((cx - 0.4, cy - 0.28), 0.8, 0.28, fc=INK, alpha=0.22, zorder=3))
    ax.add_patch(mp.Rectangle((cx + 0.05, cy - 0.16), 0.2, 0.13, fc=CORAL, alpha=0.8, zorder=4))


def gps_icon(ax, cx, cy):
    ax.add_patch(mp.Circle((cx, cy + 0.06), 0.2, fc=CORAL, ec="none", zorder=3))
    ax.add_patch(mp.Polygon([[cx - 0.14, cy + 0.02], [cx + 0.14, cy + 0.02], [cx, cy - 0.26]],
                 closed=True, fc=CORAL, ec="none", zorder=3))
    ax.add_patch(mp.Circle((cx, cy + 0.08), 0.075, fc="white", ec="none", zorder=4))


def grid_icon(ax, cx, cy):
    ax.add_patch(mp.FancyBboxPatch((cx - 0.36, cy - 0.28), 0.72, 0.56,
                 boxstyle="round,pad=0,rounding_size=0.05", fc="#F4F7F8", ec=INK + "44", lw=0.7, zorder=3))
    for gx in [-0.18, 0, 0.18]:
        ax.plot([cx + gx, cx + gx], [cy - 0.26, cy + 0.26], color=INK, alpha=0.15, lw=0.6, zorder=4)
    for gy in [-0.13, 0.13]:
        ax.plot([cx - 0.34, cx + 0.34], [cy + gy, cy + gy], color=INK, alpha=0.15, lw=0.6, zorder=4)
    ax.add_patch(mp.Rectangle((cx + 0.02, cy + 0.0), 0.14, 0.12, fc=TEAL, zorder=5))


# ------------------------------- architecture -------------------------------
def architecture():
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 13.2); ax.set_ylim(0.0, 9.8); ax.axis("off"); ax.set_aspect("equal")
    R = 7.0  # main-row y for panel (a)

    ax.text(0.2, 9.35, "(a)  Pipeline", fontsize=9, fontweight="bold", color=INK, ha="left")
    # icons
    cam_icon(ax, 0.75, R + 1.4); cap(ax, 0.75, R + 0.82, "camera  $3\\times224^2$")
    gps_icon(ax, 0.75, R); cap(ax, 0.75, R - 0.58, "GPS  $(x,y,z)$")
    grid_icon(ax, 0.75, R - 1.4); cap(ax, 0.75, R - 1.95, "LiDAR / radar  (opt.)")
    # encoders
    ec = card(ax, 3.0, R + 1.4, 1.9, 0.95, "depthwise CNN\n$\\phi_c$", TEAL, fTEAL)
    eg = card(ax, 3.0, R, 1.9, 0.95, "MLP\n$\\phi_g$", TEAL, fTEAL)
    er = card(ax, 3.0, R - 1.4, 1.9, 0.95, "shared CNN\n$\\phi_r$", TEAL, fTEAL, dashed=True)
    for icy, c in [(R + 1.4, ec), (R, eg), (R - 1.4, er)]:
        arrow(ax, (1.25, icy), (c[0] - c[2] / 2, icy), dashed=(c is er))
    # per-frame bracket
    ax.add_patch(mp.FancyBboxPatch((1.9, R - 1.95), 2.2, 4.15, boxstyle="round,pad=0,rounding_size=0.1",
                 fc="none", ec=SLATE, lw=0.8, ls=(0, (4, 3)), alpha=0.55, zorder=1))
    cap(ax, 3.0, R - 2.5, "per-frame, window $t\\!-\\!W\\!+\\!1\\!:\\!t$")
    # fusion
    fu = card(ax, 5.65, R, 1.55, 1.15, "gated\nfusion  $\\odot$", SAND, fSAND)
    arrow(ax, (ec[0] + ec[2] / 2, R + 1.4), (fu[0] - 0.5, fu[1] + 0.38))
    arrow(ax, (eg[0] + eg[2] / 2, R), (fu[0] - fu[2] / 2, fu[1]))
    arrow(ax, (er[0] + er[2] / 2, R - 1.4), (fu[0] - 0.5, fu[1] - 0.38), dashed=True)
    edge(ax, 4.42, R + 0.46, "$d\\!=\\!128$")
    # SSM stack
    card(ax, 8.2, R + 0.22, 1.9, 1.15, "", INDIGO, "#DEDCF2")
    card(ax, 8.1, R + 0.11, 1.9, 1.15, "", INDIGO, "#E7E5F5")
    ssm = card(ax, 8.0, R, 1.9, 1.15, "selective\nSSM  $\\times4$", INDIGO, fIND)
    arrow(ax, (fu[0] + fu[2] / 2, R), (ssm[0] - ssm[2] / 2, R)); edge(ax, 6.85, R + 0.46, "$(W,d)$")
    # head
    hd = card(ax, 10.45, R, 1.5, 0.95, "linear\nhead", CORAL, fCOR)
    arrow(ax, (ssm[0] + ssm[2] / 2 + 0.2, R), (hd[0] - hd[2] / 2, R)); edge(ax, 9.55, R + 0.46, "$\\mathbf{h}_W$")
    arrow(ax, (hd[0] + hd[2] / 2, R), (hd[0] + hd[2] / 2 + 0.75, R)); edge(ax, 11.55, R + 0.42, "$64$")
    # beam fan
    bx, by = 12.6, R
    for a, col in [(38, INK + "2E"), (19, INK + "2E"), (0, TEAL), (-19, INK + "2E"), (-38, INK + "2E")]:
        ax.add_patch(mp.Wedge((bx, by), 0.72, a - 5, a + 5, fc=col, ec="none", zorder=3))
    cap(ax, bx + 0.05, by - 0.98, "beam $\\hat{y}_t$")

    # ---- panel (b) ----
    ax.text(0.2, 3.35, "(b)  Selective-SSM block", fontsize=9, fontweight="bold", color=INK, ha="left")
    y0, yt = 1.35, 2.05
    nm = card(ax, 0.95, y0, 1.35, 0.8, "RMS\nNorm", INK, fGRY)
    ip = card(ax, 2.75, y0, 1.35, 0.8, "in_proj", TEAL, fTEAL)
    cv = card(ax, 4.6, yt, 1.55, 0.8, "Conv1d\n+SiLU", TEAL, fTEAL)
    xp = card(ax, 6.5, yt, 1.55, 0.8, "$x_{\\mathrm{proj}}$\n$\\Delta,B,C$", TEAL, fTEAL)
    sc = card(ax, 8.5, yt, 1.7, 0.8, "selective\nscan $(A,D)$", INDIGO, fIND)
    gt = card(ax, 10.4, y0, 1.3, 0.8, "$\\otimes$ gate", SAND, fSAND)
    op = card(ax, 12.1, y0, 1.3, 0.8, "out_proj", TEAL, fTEAL)
    arrow(ax, (0.12, y0), (nm[0] - nm[2] / 2, y0))
    arrow(ax, (nm[0] + nm[2] / 2, y0), (ip[0] - ip[2] / 2, y0))
    arrow(ax, (ip[0] + ip[2] / 2, y0), (cv[0], cv[1] - 0.4), rad=-0.15); edge(ax, 3.78, yt + 0.02, "$x$", fs=6.5)
    arrow(ax, (cv[0] + cv[2] / 2, yt), (xp[0] - xp[2] / 2, yt))
    arrow(ax, (xp[0] + xp[2] / 2, yt), (sc[0] - sc[2] / 2, yt))
    arrow(ax, (sc[0] + sc[2] / 2, yt), (gt[0], gt[1] + 0.4), rad=-0.15)
    arrow(ax, (ip[0], ip[1] - 0.4), (gt[0], gt[1] - 0.4), rad=-0.26)
    cap(ax, 6.5, 0.68, "$z \\to \\mathrm{SiLU}$", fs=7.5)
    arrow(ax, (gt[0] + gt[2] / 2, y0), (op[0] - op[2] / 2, y0))
    arrow(ax, (op[0] + op[2] / 2, y0), (op[0] + op[2] / 2 + 0.75, y0))
    # residual arc (over the top; label rides on it with a clean background)
    ax.annotate("", xy=(op[0] + 0.66, y0 + 0.44), xytext=(0.14, y0 + 0.44), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=SLATE, alpha=0.5, lw=1.0, ls=(0, (4, 3)),
                                connectionstyle="arc3,rad=-0.16"))
    edge(ax, 6.4, 2.82, "residual", fs=7.5, color=SLATE)
    _save(fig, "arch")


# ------------------------------- system-scene helpers -------------------------------
def _car(ax, cx, cy, col=CORAL):
    """Sleek minimal car silhouette."""
    ax.add_patch(mp.FancyBboxPatch((cx - 0.82, cy), 1.64, 0.4,
                 boxstyle="round,pad=0,rounding_size=0.18", fc=col, ec="none", zorder=5))
    ax.add_patch(mp.Polygon([[cx - 0.52, cy + 0.36], [cx + 0.5, cy + 0.36],
                 [cx + 0.34, cy + 0.74], [cx - 0.34, cy + 0.74]], closed=True,
                 fc=col, ec="none", zorder=5))
    ax.add_patch(mp.Polygon([[cx - 0.44, cy + 0.4], [cx + 0.42, cy + 0.4],
                 [cx + 0.3, cy + 0.68], [cx - 0.28, cy + 0.68]], closed=True,
                 fc="white", ec="none", alpha=0.55, zorder=6))
    for wx in (cx - 0.46, cx + 0.46):
        ax.add_patch(mp.Circle((wx, cy - 0.02), 0.17, fc=INK, zorder=6))
        ax.add_patch(mp.Circle((wx, cy - 0.02), 0.07, fc="white", zorder=7))


def _chip(ax, cx, cy, s=0.34, color=TEAL):
    """Edge-SoC chip glyph (rounded die + pins)."""
    for t in (-s * 0.28, 0, s * 0.28):
        ax.plot([cx + t, cx + t], [cy + s / 2, cy + s / 2 + s * 0.22], color=color, lw=1.0, zorder=6)
        ax.plot([cx + t, cx + t], [cy - s / 2, cy - s / 2 - s * 0.22], color=color, lw=1.0, zorder=6)
        ax.plot([cx - s / 2, cx - s / 2 - s * 0.22], [cy + t, cy + t], color=color, lw=1.0, zorder=6)
        ax.plot([cx + s / 2, cx + s / 2 + s * 0.22], [cy + t, cy + t], color=color, lw=1.0, zorder=6)
    ax.add_patch(mp.FancyBboxPatch((cx - s / 2, cy - s / 2), s, s,
                 boxstyle="round,pad=0,rounding_size=0.05", fc="white", ec=color, lw=1.4, zorder=7))
    ax.add_patch(mp.FancyBboxPatch((cx - s * 0.22, cy - s * 0.22), s * 0.44, s * 0.44,
                 boxstyle="round,pad=0,rounding_size=0.03", fc=color, ec="none", alpha=0.9, zorder=8))


def _gpspin(ax, cx, cy, col=TEAL):
    ax.add_patch(mp.Circle((cx, cy + 0.06), 0.16, fc=col, ec="none", zorder=7))
    ax.add_patch(mp.Polygon([[cx - 0.11, cy + 0.02], [cx + 0.11, cy + 0.02], [cx, cy - 0.2]],
                 closed=True, fc=col, ec="none", zorder=7))
    ax.add_patch(mp.Circle((cx, cy + 0.07), 0.06, fc="white", ec="none", zorder=8))


# ------------------------------- system scenario -------------------------------
def system():
    import numpy as np
    fig, ax = plt.subplots(figsize=(4.0, 2.35))
    ax.set_xlim(0, 11.2); ax.set_ylim(-2.75, 3.35); ax.axis("off"); ax.set_aspect("equal")

    # --- road with subtle perspective + dashed lane line ---
    ax.add_patch(mp.Polygon([[0.5, 0], [11.0, 0], [11.0, -0.42], [0.5, -0.24]], closed=True,
                 fc=INK, alpha=0.06, ec="none", zorder=0))
    ax.plot([0.5, 11.0], [0, 0], color=INK, alpha=0.32, lw=1.1, zorder=1, solid_capstyle="round")
    ax.plot([2.6, 10.8], [-0.14, -0.14], color=INK, alpha=0.16, lw=1.0, ls=(0, (6, 5)), zorder=1)

    # --- base station: pole, antenna panel, sensor head, RF glow ---
    bx = 1.4
    ax.plot([bx, bx], [0, 2.32], color=INK, lw=2.0, zorder=3, solid_capstyle="round")
    ax.add_patch(mp.FancyBboxPatch((bx - 0.15, 1.5), 0.3, 0.82,
                 boxstyle="round,pad=0,rounding_size=0.05", fc=INK, ec="none", alpha=0.88, zorder=4))
    apex = (bx + 0.2, 2.0)
    for r, a in [(0.62, 0.05), (0.44, 0.07), (0.27, 0.11)]:      # RF glow
        ax.add_patch(mp.Circle(apex, r, fc=TEAL, ec="none", alpha=a, zorder=2))
    ax.add_patch(mp.FancyBboxPatch((bx - 0.24, 2.36), 0.48, 0.3,
                 boxstyle="round,pad=0,rounding_size=0.05", fc=TEAL, ec="none", zorder=5))
    ax.add_patch(mp.Circle((bx - 0.07, 2.51), 0.055, fc="white", zorder=6))   # camera lens
    ax.add_patch(mp.Circle((bx + 0.11, 2.51), 0.03, fc="white", alpha=0.8, zorder=6))
    cap(ax, bx + 0.05, 2.92, "roadside base station", 7.0, color=INK)
    cap(ax, bx + 0.05, -0.34, "camera + GPS sensors", 6.6)

    # --- candidate beams (codebook) fanning out, one selected ---
    carx, cary = 8.35, 0.28
    for ex, ey in [(10.7, 2.5), (10.8, 1.7), (10.8, 0.95), (10.4, -0.15)]:
        ax.plot([apex[0], ex], [apex[1], ey], color=INK, alpha=0.13, lw=1.0, zorder=1)
    # selected beam: soft glow + bright core + spread wedge
    tip = (carx, cary + 0.5)
    ax.add_patch(mp.Polygon([apex, (carx + 0.3, cary + 0.95), (carx + 0.3, cary + 0.05)],
                 closed=True, fc=TEAL, alpha=0.10, ec="none", zorder=1))
    ax.plot([apex[0], tip[0]], [apex[1], tip[1]], color=TEAL, lw=7, alpha=0.12,
            solid_capstyle="round", zorder=2)
    ax.plot([apex[0], tip[0]], [apex[1], tip[1]], color=TEAL, lw=2.4, alpha=0.9,
            solid_capstyle="round", zorder=3)
    ang = np.degrees(np.arctan2(tip[1] - apex[1], tip[0] - apex[0]))
    ax.text((apex[0] + tip[0]) / 2 + 0.1, (apex[1] + tip[1]) / 2 + 0.34,
            "selected beam $\\hat{y}_t$", fontsize=7.5, color="#1F7A6E", rotation=ang,
            rotation_mode="anchor", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))

    # --- vehicle + GPS + motion ---
    _car(ax, carx, cary)
    _gpspin(ax, carx, cary + 1.12)
    cap(ax, carx, cary + 1.5, "GPS", 6.6, color="#1F7A6E")
    cap(ax, carx, -0.34, "vehicle", 6.6)
    ax.annotate("", xy=(carx + 1.7, cary + 0.18), xytext=(carx + 1.02, cary + 0.18),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=10, color=SLATE, lw=1.2))
    cap(ax, carx + 1.9, cary + 0.18, "motion", 6.6, ha="left")

    # --- edge-compute badge (the paper's headline), auto-sized so text never overflows ---
    chip_cx = 3.05
    _chip(ax, chip_cx, -1.82)
    t1 = ax.text(chip_cx + 0.6, -1.63, "on-device edge inference", fontsize=7.4,
                 color=INK, ha="left", va="center", fontweight="bold", zorder=6)
    t2 = ax.text(chip_cx + 0.6, -2.03, "no over-the-air beam sweep", fontsize=6.7,
                 color=SLATE, ha="left", va="center", zorder=6)
    fig.canvas.draw()                                        # enable text measurement
    inv = ax.transData.inverted(); rend = fig.canvas.get_renderer()
    rights, bots, tops = [], [], []
    for t in (t1, t2):
        bb = t.get_window_extent(renderer=rend)
        (_, y0) = inv.transform((bb.x0, bb.y0)); (x1, y1) = inv.transform((bb.x1, bb.y1))
        rights.append(x1); bots.append(y0); tops.append(y1)
    x_l = chip_cx - 0.55                                     # include the chip glyph
    x_r = max(rights) + 0.35
    y_b = min(bots) - 0.24; y_t = max(tops) + 0.24
    ax.add_patch(mp.FancyBboxPatch((x_l, y_b), x_r - x_l, y_t - y_b,
                 boxstyle="round,pad=0,rounding_size=0.13", fc=fTEAL, ec=TEAL, lw=1.1, zorder=4))
    # connector: predictor runs at the base station
    ax.annotate("", xy=(x_l + 0.55, y_t), xytext=(bx + 0.15, 0.05),
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=1.0, alpha=0.5,
                                ls=(0, (4, 3)), connectionstyle="arc3,rad=-0.2"), zorder=2)
    _save(fig, "system")


if __name__ == "__main__":
    # NOTE: BOTH schematic figures are now sourced from the design tool and live at
    # paper/latex/figs/{system,arch}.pdf (sources under figs/src/). We no longer
    # regenerate either here so this script cannot overwrite them. The matplotlib
    # fallbacks remain as functions (system(), architecture()) if ever needed.
    print("Fig 1 (system) and Fig 2 (arch) are design-tool figures; nothing regenerated.")
