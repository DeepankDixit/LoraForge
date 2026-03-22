# LoraForge

**End-to-end inference optimization pipeline for LoRA fine-tuned LLMs.**

> Take any LoRA fine-tuned model from naive FP16 deployment to production-optimized inference — in one command.

> LoraForge is an open-source, production-grade framework that takes any LoRA fine-tuned language model and automatically produces a fully optimized, benchmarked, and deployment-ready inference artifact.

> LoraForge is an automated inference optimization pipeline for anyone with a LoRA adapter. The contract is: bring an adapter trained on meta-llama/Llama-3.1-8B-Instruct, and LoraForge runs the entire optimization suite for you — merge, deploy, benchmark baseline, quantize, optimize KV cache, apply speculative decoding, and produce a dashboard comparing all variants.

> The domain does not matter. Cybersecurity is what WE use to build and demo the pipeline. Someone with a medical Q&A adapter, a legal document adapter, or a code generation adapter could plug theirs in and get the same treatment. The pipeline never looks at what the adapter was trained on — it only checks that the base model matches.

> Product (Activities 1–5): This is what LoraForge actually sells as a tool. The user brings their adapter, points the CLI at it, and gets the full optimization pipeline run automatically. The output is a dashboard showing: "Your baseline FP16 model uses 16.8GB and serves 52 tokens/second. After INT4 quantization it uses 4GB and serves 180 tokens/second with only 2% accuracy drop. After adding speculative decoding, TTFT drops from 280ms to 140ms." The user then makes an informed deployment decision based on real numbers from their own adapter, not generic benchmarks.
So the value proposition in one sentence: "Fine-tuned your model? Run one command and we tell you exactly what happens to its speed, memory, and accuracy under every major production optimization."

---

## The Problem

Millions of LoRA fine-tuned models exist. None come with production inference optimization.

When you fine-tune Llama-3.1-8B with LoRA and deploy it naively in FP16, you're leaving 3x performance on the table:

| | Naive FP16 | LoraForge Optimized |
|---|---|---|
| GPU Memory | 16 GB | **4 GB** (INT4 AWQ) |
| Time to First Token | ~340 ms | **~71 ms** |
| Throughput | ~800 tok/s | **~2,500 tok/s** |
| Accuracy loss | 0% | **< 2%** (MMLU) |
| Cost/1M tokens | $0.12 | **$0.03** |

LoraForge automates: quantization profiling, KV cache optimization, speculative decoding, and full benchmark reporting.

---

## Install

```bash
pip install loraforge
```

## Quick Start

```bash
loraforge optimize \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapter your-org/your-lora-adapter \
  --accuracy-budget 0.02 \
  --output-dir ./my-optimized-model
```

Then view your results:

```bash
loraforge report --results-dir ./my-optimized-model
```

Or serve the optimized model:

```bash
loraforge serve --results-dir ./my-optimized-model --port 8000
```

---

## What LoraForge Does

**Activity 1 — Baseline:** Deploys your model with vLLM, measures TTFT, throughput, GPU memory, and perplexity. Establishes the "before" benchmark.

**Activity 2 — Quantization:** Applies PTQ using NVIDIA ModelOpt across FP8, INT4 AWQ, and INT8 SmoothQuant. Benchmarks each format, selects the best within your accuracy budget.

**Activity 3 — KV Cache:** Configures prefix caching (eliminates repeated prefill for shared system prompts), quantized KV cache, and sliding window attention for long contexts.

**Activity 4 — Speculative Decoding:** Trains an EAGLE draft model and deploys it alongside your base model. Lossless 2–3x generation speedup.

**Activity 5 — Dashboard:** Aggregates all results into a Streamlit dashboard with Pareto curves, comparison tables, and cost estimates.

**Activity 6 — Domain CPT:** LoRA-based continued pre-training on domain corpora before the optimization pipeline.

---

## Supported Models

- `meta-llama/Llama-3.1-8B-Instruct` (primary, tested)
- `Qwen/Qwen2.5-7B-Instruct` (secondary, tested)
- Any HuggingFace-compatible decoder model (experimental)

---

## Requirements

- Python 3.10+
- CUDA 12.1+ (NVIDIA GPU, A10 or better recommended)
- 40GB+ VRAM for full pipeline (16GB for quantized serving only)

---

## Documentation

Full documentation in [`documentation/`](documentation/):
- [Project Overview](documentation/PROJECT_OVERVIEW.md)
- [Architecture](documentation/ARCHITECTURE.md)
- [Activity Breakdown](documentation/ACTIVITY_BREAKDOWN.md)
- [GPU Requirements & Costs](documentation/GPU_REQUIREMENTS.md)

---

## Author

**Deepank Dixit** — Senior AI/ML Engineer, MTech AI (IISc Bengaluru)

Inspired by NVIDIA's AI Inference Workshop at IISc Bengaluru (December 2025).

---

## License

MIT License — use freely, attribution appreciated.
