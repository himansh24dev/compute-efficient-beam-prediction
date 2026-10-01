# Exported ONNX models of the deployed configuration (revision)
Camera+GPS selective-SSM, W=2, validation-selected checkpoint A_ssm_2M_i1337 (opset 17; verified against
PyTorch, max abs error < 5e-6):
- `beam_ssm_fp32.onnx` — fp32 (23.2 ms forward pass on a Raspberry Pi 4, four threads)
- `beam_ssm_int8.onnx` — static INT8, whole model (12.6 ms)
- `beam_ssm_int8_selective.onnx` — static INT8, convolutions only (18.6 ms)
- `beam_gps_fp32.onnx` — GPS-only model A_ssm_gps_i1337 (3.4 ms)
Inputs: `gps` (batch, 2, 3) standardized north/east offset in metres and speed; `camera` (batch, 2, 3, 224, 224)
ImageNet-normalized. Output: 64 beam logits. Benchmark harness: experiments/revision/pi/.
