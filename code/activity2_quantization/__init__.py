"""
activity2_quantization
======================
Post-training quantization (PTQ) pipeline for LoraForge.

Applies three quantization formats to the Activity 1 FP16 merged model:
  - FP8 (E4M3):       ~8 GB VRAM, near-lossless, best for latency-critical deployments
  - AWQ INT4 (W4A16): ~4–5 GB VRAM, best VRAM compression, small accuracy cost
  - SmoothQuant INT8: ~8 GB VRAM, best raw throughput on W8A8 integer hardware

Pipeline steps:
  1. CalibrationDatasetBuilder  → build calibration dataset (512 samples, domain-matched)
  2. QuantizationProfiler       → apply PTQ via nvidia-modelopt, save quantized checkpoints
  3. QuantizationBenchmarker    → benchmark each format (TTFT + throughput) vs FP16 baseline
  4. FormatSelector             → score formats, recommend best fit for each deployment goal
  5. QuantizationReporter       → generate comparison report (quant_results.json + .md + .html)

Run from repo root:
  python -m activity2_quantization.main --mode all
  python -m activity2_quantization.main --mode quantize --format fp8
  python -m activity2_quantization.main --mode benchmark --format awq_int4
  python -m activity2_quantization.main --mode select
  python -m activity2_quantization.main --mode report
"""

__version__ = "0.1.0"
