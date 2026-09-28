# Pi run 1 (2026-09-28, ~16:14-16:36) — validity

Board: Raspberry Pi 4 Model B Rev 1.1, Raspberry Pi OS (Debian 13), Python 3.13.5, onnxruntime 1.28.0,
performance governor, passive cooling.

VALID (no throttling; the soak's first sample at t=2.3 s shows throttled=0x0, i.e. nothing had
throttled before the soak):
  fwd_deploy_fp32_t1..t4, fwd_deploy_int8_t4, fwd_deploy_int8_selective_t4, fwd_gps_fp32_t4,
  fwd_cam_fp32_t4, fwd_core_{transformer,gru,lstm,mlp}_t4,
  pipe_deploy_fp32_cv2, pipe_deploy_fp32_pil, pipe_deploy_int8_cv2,
  pipe_deploy_int8_selective_cv2, pipe_gps_fp32

INVALID (thermally throttled; DO NOT USE, kept for transparency):
  soak_deploy_fp32_cv2.json   started at 72 C (performance governor at idle, 120 s cooldown was
                              insufficient); soft limit at 45 s (80.8 C), ARM capping by 90 s,
                              peak 85.7 C, p50 46.4 -> 54.2 ms.
  replica_full_t4.json        measured at 81-83 C after the soak (3024 ms).
  (replica_half, the profiles and the extra batch were aborted.)

All invalid items were re-measured in run 2 (../run2/) with a cool start (<=50 C) before every
benchmark and a 1 s thermal/clock/throttle log per benchmark.
