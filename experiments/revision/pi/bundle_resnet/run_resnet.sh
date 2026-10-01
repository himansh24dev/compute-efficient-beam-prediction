#!/usr/bin/env bash
# ResNet-50 x 5 frames cross-check, same thermal control as run_pi2.sh (cool to <= 50 C,
# performance governor during the run, 1 s temperature/clock/throttle log).
set -uo pipefail
PY=${PY:-python3}; GOV=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; OUT=~/results3; mkdir -p "$OUT"
temp() { vcgencmd measure_temp | sed -E "s/temp=([0-9.]+).*/\1/"; }
echo ondemand > $GOV; t0=$SECONDS
while awk -v t="$(temp)" -v m=50 'BEGIN{exit !(t>m)}'; do sleep 10; (( SECONDS - t0 > 1800 )) && break; done
echo "cooled to $(temp) C in $((SECONDS - t0)) s"; { uname -a; vcgencmd get_throttled; } > "$OUT/env.txt"
echo performance > $GOV; sleep 2
( while true; do echo "$(date +%s),$(temp),$(vcgencmd measure_clock arm | cut -d= -f2),$(vcgencmd get_throttled | cut -d= -f2)"; sleep 1; done ) > "$OUT/thermal_resnet50.csv" & S=$!
echo "=== resnet50 start $(temp) C"
$PY bench_auto.py --model resnet50_5frames.onnx --window 5 --threads 4 --runs 30 --out "$OUT/resnet50_5frames_t4.json"
kill $S; echo "=== resnet50 end $(temp) C"; echo ondemand > $GOV; echo "ALL DONE (resnet)"
