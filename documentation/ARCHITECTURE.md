# LoraForge — System Architecture

## Data Flow: End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LORAFORGE PIPELINE                          │
└─────────────────────────────────────────────────────────────────────┘

  INPUT
  ──────
  HuggingFace Base Model (e.g., Llama-3.1-8B-Instruct)
  +
  LoRA Adapter (PEFT checkpoint) OR raw SFT model
  +
  Config YAML (task type, quantization budget, serving mode)

       │
       ▼
┌─────────────┐
│  ACTIVITY 6 │  [OPTIONAL FIRST STEP]
│  Domain CPT │  LoRA-based continued pre-training on domain corpus
│             │  → Outputs: domain-adapted LoRA adapter
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ACTIVITY 1 │
│   Baseline  │  Merge base + LoRA adapter → FP16 model
│   Serving   │  Deploy via vLLM → collect TTFT, throughput, memory, perplexity
└──────┬──────┘
       │ baseline_metrics.json
       ▼
┌─────────────┐
│  ACTIVITY 2 │
│ Quantization│  Profile model → run PTQ (FP8, INT4-AWQ, INT8-SQ)
│  Pipeline   │  Calibration dataset → ModelOpt → quantized checkpoint
│             │  → Benchmark each format → select best
└──────┬──────┘
       │ best_quant_config.json + quantized_checkpoint/
       ▼
┌─────────────┐
│  ACTIVITY 3 │
│  KV Cache   │  Configure vLLM: prefix caching + quantized KV + window attn
│ Optimization│  Simulate multi-turn workload → measure TTFT reduction
└──────┬──────┘
       │ kv_cache_metrics.json
       ▼
┌─────────────┐
│  ACTIVITY 4 │
│ Speculative │  Train EAGLE draft model (small auxiliary decoder)
│  Decoding   │  Deploy base + draft together → measure acceptance rate + speedup
└──────┬──────┘
       │ speculative_metrics.json
       ▼
┌─────────────┐
│  ACTIVITY 5 │
│  Benchmark  │  Aggregate all metrics → generate dashboard
│  Dashboard  │  Pareto curves, comparison tables, cost estimates
│             │  → CLI: `loraforge optimize --model ... --adapter ...`
└─────────────┘

  OUTPUT
  ──────
  - Optimized model checkpoint (quantized + serving-ready)
  - vLLM deployment config
  - TRT-LLM export (optional)
  - benchmark_report.html (shareable dashboard)
  - optimization_summary.json (machine-readable results)
```

---

## Module Design

### shared/utils/
- `gpu_profiler.py` — VRAM usage, GPU utilization, memory bandwidth tracking
- `logger.py` — Structured logging (JSON + human-readable)
- `config_parser.py` — Load and validate YAML configs
- `checkpoint_manager.py` — Save/load model checkpoints, manage versions

### shared/metrics/
- `latency_metrics.py` — TTFT (P50/P90/P99), time-per-output-token (TPOT), end-to-end latency
- `throughput_metrics.py` — tokens/sec at varying batch sizes and sequence lengths
- `accuracy_metrics.py` — Perplexity on held-out set, MMLU subset evaluation, task-specific accuracy
- `memory_metrics.py` — Peak VRAM, KV cache footprint, model weight footprint
- `cost_estimator.py` — Estimates cost/1M tokens using Lambda Cloud/RunPod GPU pricing

### shared/models/
- `model_loader.py` — Load base model + LoRA adapter, merge weights, push to HuggingFace Hub
- `model_exporter.py` — Export to TRT-LLM checkpoint format, ONNX export
- `lora_utils.py` — LoRA adapter inspection (rank, alpha, target modules), adapter merging utilities

### shared/configs/
- `base_config.yaml` — Default pipeline configuration
- `llama31_8b.yaml` — Llama-3.1-8B specific settings
- `qwen25_7b.yaml` — Qwen-2.5-7B specific settings
- `quantization_recipes.yaml` — PTQ recipes: FP8, INT4-AWQ, INT8-SQ configs
- `serving_config.yaml` — vLLM serving parameters (max_model_len, gpu_memory_utilization, etc.)

---

## Key Design Principles

### 1. Modularity
Each activity is a standalone Python package with its own `__init__.py`, `main.py`, and `config.yaml`. Activities share code only through the `shared/` package. You can run Activity 2 independently without running Activity 1 first (it accepts a model path directly).

### 2. Config-Driven
All hyperparameters, model paths, quantization recipes, and serving configs are in YAML files. No hardcoded values in code. This makes the pipeline reproducible across different models and environments.

### 3. Reproducibility
Every run generates a timestamped experiment directory with: the config used, all metrics in JSON, plots as PNG, and a summary markdown file. Results are deterministic given the same config and random seed.

### 4. Progressive Enhancement
The pipeline is additive. Each activity takes the output of the previous one but can also be run standalone with a raw model path. This means you can benchmark quantization alone without speculative decoding, or apply KV cache optimization to a model someone else quantized.

### 5. HuggingFace-first
The pipeline works with any model on HuggingFace Hub. Model IDs and adapter paths follow the standard HuggingFace format. This maximizes compatibility with the community's models.

---

## Serving Architecture (Activity 5 Output)

```
User Request
     │
     ▼
loraforge serve --model <model-path> --port 8000
     │
     ├── Loads: quantized checkpoint (INT4 AWQ or FP8)
     ├── Configures: prefix caching, quantized KV cache
     ├── Provisions: EAGLE draft model (if available)
     └── Starts: vLLM OpenAI-compatible API server
     │
     ▼
OpenAI-compatible API (POST /v1/chat/completions)
     │
     ▼
Response + Benchmark Metrics (X-Latency-Ms, X-Tokens-Per-Sec headers)
```

---

## Inference Phase Understanding (Core Concepts)

### Prefill Phase (Context Processing)
- Processes the entire input prompt in parallel
- Computes and stores KV tensors for every input token
- Compute-bound (matrix multiplications dominate)
- TTFT (Time to First Token) is determined here
- Optimization levers: prefix caching, chunked prefill, flash attention

### Decode Phase (Token Generation)
- Generates output tokens one at a time (autoregressive)
- Reads cached KV tensors for each new token
- Memory bandwidth-bound (reading KV cache dominates, not compute)
- Throughput (tokens/sec) is determined here
- Optimization levers: quantized KV cache, speculative decoding, continuous batching

### Why This Matters for LoraForge
- A multi-turn enterprise system (like the AI Tutor) has long system prompts → prefix caching saves repeated prefill
- A high-traffic serving system needs high decode throughput → quantization + speculative decoding help
- LoraForge benchmarks BOTH phases separately so you know exactly where time is spent

---

## Quantization Decision Tree

```
Model + Task
     │
     ├── Accuracy-critical (MMLU loss < 1%)? ──Yes──► FP8 (safe choice)
     │
     ├── Memory-constrained (fit 8B on 16GB GPU)? ──Yes──► INT4 AWQ (W4A16)
     │
     ├── Balanced (good accuracy + memory)? ──Yes──► FP8 or INT8 SmoothQuant
     │
     └── Ultra-low latency on Blackwell? ──Yes──► W4A8 (INT4 weights, FP8 activations)

Default recommendation for Llama-3.1-8B on A100 40GB:
→ INT4 AWQ for memory-constrained deployment (fits in 8GB)
→ FP8 for accuracy-sensitive deployment (near-zero accuracy loss)
```
