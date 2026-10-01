#!/usr/bin/env bash
# second Pi batch (forward pass of every remaining variant)
set -euo pipefail
PY=${PY:-python3}
mkdir -p results_extra
$PY bench_auto.py --model models/win_w4.onnx --window 4 --threads 4 --runs 300 --out results_extra/fwd_win_w4_t4.json
sleep 20
$PY bench_auto.py --model models/win_w5.onnx --window 5 --threads 4 --runs 300 --out results_extra/fwd_win_w5_t4.json
sleep 20
$PY bench_auto.py --model models/win_w8.onnx --window 8 --threads 4 --runs 100 --out results_extra/fwd_win_w8_t4.json
sleep 20
$PY bench_auto.py --model models/win_w16.onnx --window 16 --threads 4 --runs 100 --out results_extra/fwd_win_w16_t4.json
sleep 20
$PY bench_auto.py --model models/abl_lidar.onnx --window 2 --threads 4 --runs 300 --out results_extra/fwd_abl_lidar_t4.json
sleep 20
$PY bench_auto.py --model models/abl_lidar_radar.onnx --window 2 --threads 4 --runs 300 --out results_extra/fwd_abl_lidar_radar_t4.json
sleep 20
$PY bench_auto.py --model models/abl_pretrained.onnx --window 2 --threads 4 --runs 100 --out results_extra/fwd_abl_pretrained_t4.json
sleep 20
$PY bench_auto.py --model models/abl_L2.onnx --window 2 --threads 4 --runs 300 --out results_extra/fwd_abl_L2_t4.json
sleep 20
$PY bench_auto.py --model models/abl_L6.onnx --window 2 --threads 4 --runs 300 --out results_extra/fwd_abl_L6_t4.json
sleep 20
$PY bench_auto.py --model models/abl_fuse_mean.onnx --window 2 --threads 4 --runs 300 --out results_extra/fwd_abl_fuse_mean_t4.json
sleep 20
$PY bench_auto.py --model models/abl_fuse_concat.onnx --window 2 --threads 4 --runs 300 --out results_extra/fwd_abl_fuse_concat_t4.json
echo ALL EXTRA DONE
