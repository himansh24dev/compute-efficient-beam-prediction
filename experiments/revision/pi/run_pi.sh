#!/usr/bin/env bash
# Raspberry Pi 4 runs for the Scientific Reports revision. Run from the bundle folder:
#   python3 -m venv ~/pienv && ~/pienv/bin/pip install onnxruntime==1.28.* numpy opencv-python-headless pillow
#   sudo cpufreq-set -g performance   # or: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
#   PY=~/pienv/bin/python bash run_pi.sh
# Results land in results/; copy that folder back to experiments/revision/results/pi/.
set -euo pipefail
PY=${PY:-python3}
mkdir -p results
cooldown() { echo "cooldown $1 s"; sleep "$1"; vcgencmd measure_temp || true; }
{ uname -a; cat /proc/device-tree/model; echo; vcgencmd get_throttled; vcgencmd measure_temp;
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; $PY -c "import onnxruntime,numpy,cv2;print(onnxruntime.__version__,numpy.__version__,cv2.__version__)"; } \
  > results/env.txt 2>&1

# 1) forward pass: deployed fp32 / INT8 / selective INT8 (retrained weights; same architecture)
for m in deploy_fp32 deploy_int8 deploy_int8_selective; do
  for t in 1 2 3 4; do
    [ "$m" != deploy_fp32 ] && [ "$t" != 4 ] && continue
    $PY bench_auto.py --model models/$m.onnx --window 2 --threads $t --runs 300 --out results/fwd_${m}_t$t.json
  done
done
$PY bench_auto.py --model models/gps_fp32.onnx --window 2 --threads 4 --runs 300 --out results/fwd_gps_fp32_t4.json
$PY bench_auto.py --model models/cam_fp32.onnx --window 2 --threads 4 --runs 300 --out results/fwd_cam_fp32_t4.json
for c in transformer gru lstm mlp; do
  $PY bench_auto.py --model models/core_${c}_fp32.onnx --window 2 --threads 4 --runs 300 --out results/fwd_core_${c}_t4.json
done
cooldown 60

# 2) processing pipeline on real frames: training-matched cv2 (primary) and original PIL
for m in deploy_fp32 deploy_int8 deploy_int8_selective; do
  $PY bench_pipeline.py --model models/$m.onnx --frames-dir frames --backend cv2 --runs 300 --out results/pipe_${m}_cv2.json
done
$PY bench_pipeline.py --model models/deploy_fp32.onnx --frames-dir frames --backend pil --runs 300 --out results/pipe_deploy_fp32_pil.json
$PY bench_pipeline.py --model models/gps_fp32.onnx --frames-dir frames --gps-only --runs 300 --out results/pipe_gps_fp32.json
cooldown 120

# 3) thermal soak, 150 s, processing pipeline (cv2)
$PY bench_pipeline.py --model models/deploy_fp32.onnx --frames-dir frames --backend cv2 --soak-seconds 150 --out results/soak_deploy_fp32_cv2.json
cooldown 180

# 4) reconstructions: full (23.41 GMACs) and half budget (11.58 GMACs), forward pass
$PY bench_auto.py --model models/replica_full.onnx --window 5 --threads 4 --runs 30 --out results/replica_full_t4.json
cooldown 60
$PY bench_auto.py --model models/replica_half.onnx --window 5 --threads 4 --runs 30 --out results/replica_half_t4.json

# 5) per-operator profiles
$PY bench_profile.py --model models/replica_full.onnx --window 5 --threads 4 --runs 10 > results/profile_replica_full.txt
$PY bench_profile.py --model models/replica_half.onnx --window 5 --threads 4 --runs 10 > results/profile_replica_half.txt
$PY bench_profile.py --model models/deploy_fp32.onnx --window 2 --threads 4 --runs 100 > results/profile_deploy_fp32.txt
echo "ALL DONE"; ls -la results
