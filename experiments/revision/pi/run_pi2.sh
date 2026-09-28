#!/usr/bin/env bash
# Thermally controlled re-run (revision, 2026-09-28). The first run's thermal soak started
# hot (72 C, performance governor at idle) and hit the soft limit / frequency capping, which
# invalidated everything measured after it. Here every benchmark:
#   1. idles on the ondemand governor until the SoC is <= T_START (50 C by default,
#      matching the submission's soak start of 48.7 C),
#   2. switches to the performance governor and runs,
#   3. is accompanied by a 1 s sampler logging time, temperature, ARM clock and the
#      vcgencmd throttle word; bits 0-3 (currently under-volted / capped / throttled /
#      soft-limited) must stay 0 during a valid measurement.
# Prerequisite (once, as root): chmod o+w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Run from ~/ :  PY=~/pienv/bin/python nohup bash run_pi2.sh > run2.log 2>&1 < /dev/null &
set -uo pipefail
PY=${PY:-python3}
T_START=${T_START:-50}
GOV=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
OUT=~/results2; mkdir -p "$OUT"
temp() { vcgencmd measure_temp | sed -E "s/temp=([0-9.]+).*/\1/"; }
cool() { echo ondemand > $GOV; local t0=$SECONDS; while awk -v t="$(temp)" -v m="$T_START" 'BEGIN{exit !(t>m)}'; do
           sleep 10; (( SECONDS - t0 > 1800 )) && { echo "WARN cooldown timeout at $(temp) C"; break; }; done
         echo "cooled to $(temp) C in $((SECONDS - t0)) s"; }
sample_start() { ( while true; do echo "$(date +%s.%N | cut -c1-14),$(temp),$(vcgencmd measure_clock arm | cut -d= -f2),$(vcgencmd get_throttled | cut -d= -f2)"; sleep 1; done ) > "$OUT/thermal_$1.csv" & SAMPLER=$!; }
sample_stop() { kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null; }
bench() {   # bench <tag> <command...>
  local tag=$1; shift
  cool; echo performance > $GOV; sleep 2
  echo "=== $tag start $(temp) C"; sample_start "$tag"
  "$@"; local rc=$?
  sample_stop; echo "=== $tag end $(temp) C rc=$rc"
}
{ uname -a; cat /proc/device-tree/model; echo; vcgencmd get_throttled; vcgencmd measure_temp;
  $PY -c "import onnxruntime,numpy,cv2;print(onnxruntime.__version__,numpy.__version__,cv2.__version__)"; } > "$OUT/env.txt" 2>&1

cd ~/bundle
# control: deployed fp32 forward pass, to confirm the first run's clean numbers
bench control_fwd_deploy_fp32_t4 $PY bench_auto.py --model models/deploy_fp32.onnx --window 2 --threads 4 --runs 300 --out $OUT/fwd_deploy_fp32_t4_control.json
# thermal soak from a cool start (150 s, processing pipeline, cv2)
bench soak $PY bench_pipeline.py --model models/deploy_fp32.onnx --frames-dir frames --backend cv2 --soak-seconds 150 --out $OUT/soak_deploy_fp32_cv2.json
# reconstructions (forward pass) and per-operator profiles
bench replica_full $PY bench_auto.py --model models/replica_full.onnx --window 5 --threads 4 --runs 30 --out $OUT/replica_full_t4.json
bench replica_half $PY bench_auto.py --model models/replica_half.onnx --window 5 --threads 4 --runs 30 --out $OUT/replica_half_t4.json
bench profile_replica_full sh -c "$PY bench_profile.py --model models/replica_full.onnx --window 5 --threads 4 --runs 10 > $OUT/profile_replica_full.txt"
bench profile_replica_half sh -c "$PY bench_profile.py --model models/replica_half.onnx --window 5 --threads 4 --runs 10 > $OUT/profile_replica_half.txt"
bench profile_deploy sh -c "$PY bench_profile.py --model models/deploy_fp32.onnx --window 2 --threads 4 --runs 100 > $OUT/profile_deploy_fp32.txt"
# every remaining trained variant (window sweep + ablations)
cd ~/bundle_extra
for f in models/*.onnx; do
  tag=$(basename "$f" .onnx); W=$($PY -c "import json;print(json.load(open('models/models_info.json'))['$tag']['window'])")
  runs=300; [[ $W -ge 8 || $tag == abl_pretrained ]] && runs=100
  bench "fwd_$tag" $PY bench_auto.py --model "$f" --window "$W" --threads 4 --runs $runs --out $OUT/fwd_${tag}_t4.json
done
echo ondemand > $GOV
echo "ALL DONE (run2)"
