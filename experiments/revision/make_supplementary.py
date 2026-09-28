"""Generate the Supplementary Information tables (LaTeX) from the revision results, so
no number is transcribed by hand.

  .venv/bin/python -m experiments.revision.make_supplementary
writes ../manuscript_revised/supp_tables.tex (included by supplementary.tex).
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path("experiments/revision/results")
OUT = Path("../manuscript_revised/supp_tables.tex")

ROWS = [("Modalities", "camera only", "ssm_camera_only"), ("", "GPS only", "ssm_gps_only"),
        ("", "+ LiDAR", "ssm_+lidar"), ("", "+ LiDAR + radar", "ssm_+lidar+radar"),
        ("Encoder", "ImageNet-pretrained", "pretrained_encoder"),
        ("Temporal core", "MLP (no temporal)", "mlp"), ("", "GRU", "gru"), ("", "LSTM", "lstm"),
        ("", "Transformer (matched)", "transformer"),
        ("SSM depth", "2 layers", "ssm_L2"), ("", "6 layers", "ssm_L6"),
        ("Fusion", "mean", "fusion_mean"), ("", "learned concat", "fusion_concat")]


def f3(x):
    return f"{x:+.4f}" if abs(x) < 0.0005 and x != 0 else f"{x:+.3f}"


def delta(p):
    d = p["dba"]; lo, hi = -d["hi"], -d["lo"]          # X minus deployed
    txt = f"${f3(-d['diff'])}$ [${f3(lo)}$, ${f3(hi)}$]"
    return ("{\\boldmath " + txt + "}") if (lo > 0 or hi < 0) else txt


def table_s1():
    L = json.load(open(R / "analysis_last.json"))
    V = json.load(open(R / "analysis.json"))
    c = L["configs"]; dep = c["ssm_cam+gps (deployed)"]
    out = [r"\begin{table}[ht]\centering",
           r"\caption{Ablations without checkpoint selection: the same five-seed runs as Table~9 of the main text, "
           r"each scored at its final (50th) training epoch. $\Delta$DBA is the paired difference to the deployed "
           r"configuration on the same 23 held-out passes with its 95\% pass-level bootstrap interval; intervals in bold "
           r"exclude zero. The last column repeats the validation-selected $\Delta$DBA of Table~9 for comparison.}",
           r"\label{tab:s1}\footnotesize\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{llccc}\toprule",
           r"Factor & Setting & DBA / Top-3 & $\Delta$DBA, final epoch & $\Delta$DBA, val.\ selected \\ \midrule",
           f"& deployed configuration & {dep['dba']['mean']:.3f} / {dep['top3']['mean']:.3f} & -- & -- \\\\ \\midrule"]
    for fac, lab, k in ROWS:
        if k not in c:
            continue
        out.append(f"{fac} & {lab} & {c[k]['dba']['mean']:.3f} / {c[k]['top3']['mean']:.3f} & "
                   f"{delta(L['paired_vs_deployed'][k])} & {delta(V['paired_vs_deployed'][k])} \\\\")
    out += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(out)


def table_seeds():
    V = json.load(open(R / "analysis.json")); L = json.load(open(R / "analysis_last.json"))
    out = [r"\begin{table}[ht]\centering",
           r"\caption{Per-seed test DBA of the main configurations (initializations 1337, 2024, 7, 42, 123 on the same split). "
           r"V: validation-selected checkpoint; F: final epoch.}",
           r"\label{tab:s2}\small\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{llcccccc}\toprule",
           r"Configuration & & 1337 & 2024 & 7 & 42 & 123 & SD \\ \midrule"]
    for lab, k in (("Camera+GPS SSM (deployed)", "ssm_cam+gps (deployed)"), ("GPS only", "ssm_gps_only"),
                   ("Camera only", "ssm_camera_only"), ("Transformer", "transformer"), ("GRU", "gru"),
                   ("LSTM", "lstm"), ("MLP", "mlp")):
        for tag, A in (("V", V), ("F", L)):
            d = A["configs"][k]["dba"]
            out.append(f"{lab if tag == 'V' else ''} & {tag} & " + " & ".join(f"{x:.3f}" for x in d["per_seed"])
                       + f" & {d['seed_sd']:.3f} \\\\")
    out += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(out)


def table_robust():
    r = json.load(open(R / "robustness.json"))
    names = {"gps_iid_window": "GPS, independent per window", "gps_gauss_markov": "GPS, Gauss--Markov ($\\tau$=10\\,s)",
             "gps_pass_bias": "GPS, constant bias per pass", "cam_gauss": "Camera noise $\\sigma$",
             "cam_blur": "Camera box blur $k$", "cam_bright": "Camera brightness $\\delta$",
             "cam_occlude": "Camera occlusion $p$", "drop_camera": "Camera replaced by mean colour",
             "drop_gps": "Position replaced by training mean"}
    out = [r"\begin{table}[ht]\centering",
           r"\caption{Test-time robustness, all settings (five deployed checkpoints; ten realizations per stochastic "
           r"setting; mean DBA and Top-3 with the 2.5--97.5\% range of DBA over checkpoints and realizations). "
           r"GPS levels are the per-axis standard deviation in metres; camera levels are in ImageNet-standardized units. "
           f"Clean: DBA {r['clean'][0]['dba']:.3f}, Top-3 {r['clean'][0]['top3']:.3f}.}}",
           r"\label{tab:s3}\small\setlength{\tabcolsep}{5pt}",
           r"\begin{tabular}{llccc}\toprule Perturbation & Level & $n$ & DBA [range] & Top-3 \\ \midrule"]
    for k, lab in names.items():
        for i, e in enumerate(sorted(r[k], key=lambda x: x["level"])):
            lv = "--" if k.startswith("drop") else f"{e['level']:g}"
            out.append(f"{lab if i == 0 else ''} & {lv} & {e['n_evals']} & {e['dba']:.3f} [{e['dba_lo']:.3f}, {e['dba_hi']:.3f}] & {e['top3']:.3f} \\\\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule\end{tabular}\end{table}"
    return "\n".join(out)


def table_split():
    m = json.load(open("data/derived/splits/pass_disjoint_80_20.json"))["seed_1337"]
    cnt = {p: {} for p in ("train", "val", "test")}
    for p in cnt:
        for k in m[p]:
            sc = k[1:3]; cnt[p][sc] = cnt[p].get(sc, 0) + 1
    out = [r"\begin{table}[ht]\centering",
           r"\caption{Passes per scenario in the training, validation and test partitions of the headline "
           r"pass-disjoint split (split seed 1337; frozen manifest in the code repository).}",
           r"\label{tab:s4}\small",
           r"\begin{tabular}{lccccc}\toprule Partition & S31 & S32 & S33 & S34 & Total \\ \midrule"]
    for p in ("train", "val", "test"):
        v = [cnt[p].get(s, 0) for s in ("31", "32", "33", "34")]
        out.append(f"{p.capitalize()} & " + " & ".join(map(str, v)) + f" & {sum(v)} \\\\")
    out += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(out)


def table_pi():
    pi = json.load(open(R / "pi" / "pi_summary.json"))
    rows = [("Deployed, fp32, 1 thread", "fwd_deploy_fp32_t1"), ("Deployed, fp32, 2 threads", "fwd_deploy_fp32_t2"),
            ("Deployed, fp32, 3 threads", "fwd_deploy_fp32_t3"), ("Deployed, fp32, 4 threads", "fwd_deploy_fp32_t4"),
            ("Deployed, fp32 (control, session 2)", "fwd_deploy_fp32_t4_control"),
            ("Deployed, INT8 whole model", "fwd_deploy_int8_t4"), ("Deployed, INT8 convolutions only", "fwd_deploy_int8_selective_t4"),
            ("GPS only", "fwd_gps_fp32_t4"), ("Camera only", "fwd_cam_fp32_t4"),
            ("Core: Transformer", "fwd_core_transformer_t4"), ("Core: GRU", "fwd_core_gru_t4"),
            ("Core: LSTM", "fwd_core_lstm_t4"), ("Core: MLP", "fwd_core_mlp_t4"),
            ("Window $W{=}4$", "fwd_win_w4_t4"), ("Window $W{=}5$", "fwd_win_w5_t4"),
            ("Window $W{=}8$", "fwd_win_w8_t4"), ("Window $W{=}16$", "fwd_win_w16_t4"),
            ("+ LiDAR", "fwd_abl_lidar_t4"), ("+ LiDAR + radar", "fwd_abl_lidar_radar_t4"),
            ("ImageNet-pretrained encoder", "fwd_abl_pretrained_t4"), ("2 SSM layers", "fwd_abl_L2_t4"),
            ("6 SSM layers", "fwd_abl_L6_t4"), ("Mean fusion", "fwd_abl_fuse_mean_t4"),
            ("Concat fusion", "fwd_abl_fuse_concat_t4"),
            ("Reconstruction, full budget (23.41 GMACs)", "replica_full_t4"),
            ("Reconstruction, half budget (11.58 GMACs)", "replica_half_t4"),
            ("ResNet-50 over 5 frames (20.56 GMACs)", "resnet50_5frames_t4")]
    sess = {"run1": "1", "run2": "2", "run3": "2"}
    out = [r"\begin{table}[ht]\centering",
           r"\caption{All forward-pass measurements on the Raspberry~Pi~4 (fp32 unless stated, four threads unless "
           r"stated, batch 1; latency percentiles in ms). Session~1 runs were not thermally controlled but the board's "
           r"sticky throttle-history flags were clear after them; session~2 runs started at $\le$50$^\circ$C with "
           r"1\,s thermal logging, and none showed an active throttling flag or a clock below 1.5~GHz. Session~2 tail "
           r"percentiles include the small overhead of the 1\,s thermal sampler.}",
           r"\label{tab:s5}\footnotesize\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lcccccc}\toprule Model & p50 & p95 & p99 & Rate (Hz) & Runs & Session \\ \midrule"]
    for lab, k in rows:
        v = pi[k]
        out.append(f"{lab} & {v['p50']:.2f} & {v['p95']:.2f} & {v['p99']:.2f} & {v['hz']:.1f} & {v['runs']} & {sess[pi['source'][k]]} \\\\")
    out.append(r"\midrule \multicolumn{7}{l}{\emph{Processing pipeline (stored 960$\times$540 JPEG to beam decision; total latency in ms)}} \\")
    for lab, k in (("fp32, OpenCV (as in training)", "pipe_deploy_fp32_cv2"), ("fp32, PIL bilinear", "pipe_deploy_fp32_pil"),
                   ("INT8 whole model, OpenCV", "pipe_deploy_int8_cv2"), ("INT8 convolutions only, OpenCV", "pipe_deploy_int8_selective_cv2"),
                   ("GPS only", "pipe_gps_fp32"), ("150\\,s thermal soak, fp32, OpenCV", "soak_deploy_fp32_cv2")):
        v = pi[k]
        out.append(f"{lab} & {v['total_p50']:.2f} & {v['total_p95']:.2f} & {v['total_p99']:.2f} & {v['rate_hz']:.1f} & {v['steps']} & {sess[pi['source'][k]]} \\\\")
    out.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(out)


def main():
    parts = [table_s1(), table_seeds(), table_robust(), table_split(), table_pi()]
    OUT.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
