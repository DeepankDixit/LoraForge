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

## Activity 1 Deployment Architecture (What Actually Ran on Lambda)

The Activity 1 deployment is a single-machine setup on a Lambda Cloud A10G VM. There is no Kubernetes, no load balancer, no container registry. Understanding this stack precisely is important because it is where the real benchmark numbers were measured.

```
Lambda Cloud A10G VM (Ubuntu 22.04, 24GB HBM)
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  SSH Connection (your laptop)                                        │
│                                                                      │
│  tmux session (keeps processes alive after SSH disconnect)           │
│  ├── pane 0: python -m activity1_baseline.vllm_server                │
│  │     └── subprocess: vllm.entrypoints.openai.api_server            │
│  │           ├── FastAPI HTTP server  ← port 8000                    │
│  │           ├── vLLM PagedAttention scheduler                       │
│  │           └── NVIDIA A10G GPU (16.8GB weights + 6.7GB KV cache)   │
│  │                                                                   │
│  └── pane 1: python -m activity1_baseline.benchmark.client           │
│        └── httpx AsyncClient → POST http://localhost:8000/v1/chat/completions
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Key details:**
- `vllm_server.py` uses `subprocess.Popen()` to launch vLLM as a child process — not imported as a library. This keeps CUDA context isolated and allows clean teardown.
- Both server and benchmark client run on the same machine. Network latency is loopback (~0.1ms) and does not affect benchmark numbers.
- vLLM's server is built on FastAPI internally. You do not write any FastAPI code — you launch vLLM's built-in OpenAI-compatible server via its CLI entrypoint.
- Port 8000 is only accessible from within the VM (not exposed externally by default on Lambda). For the Activity 1 benchmark this is fine — client and server are co-located.

**Endpoints exposed by vLLM:**
- `GET /health` — readiness check (200 OK when fully loaded)
- `GET /v1/models` — lists loaded models with metadata
- `POST /v1/chat/completions` — main inference endpoint (OpenAI-compatible, streaming supported)
- `POST /v1/completions` — legacy text completion (not used in LoraForge)

---

## Activity 2.5: Production Serving Layer (Planned)

This activity sits between Activity 2 (quantization complete) and Activity 3 (KV cache optimization). It wraps the vLLM subprocess in a production-grade serving layer that stays up, handles real external traffic, and provides observability. This is what makes the LoraForge output actually deployable, not just benchmarkable.

### What it adds over Activity 1's raw vLLM

| Component | Activity 1 | Activity 2.5 |
|---|---|---|
| API auth | None | API key validation (X-API-Key header) |
| Rate limiting | None | Per-key token bucket (requests/min) |
| Request logging | Print to stdout | Structured JSON logs with latency + model metadata |
| Health/readiness | vLLM /health | FastAPI wrapper with dependency checks |
| Process persistence | Manual tmux | systemd service (survives SSH disconnect + auto-restarts) |
| Reverse proxy | None | Nginx (TLS termination, connection pooling, static file serving) |
| Metrics | JSON file only | Prometheus metrics endpoint + Grafana dashboard (optional) |

### Architecture diagram (Activity 2.5)

```
External Client
     │
     ▼ HTTPS :443
┌──────────────────┐
│  Nginx           │  TLS termination, request buffering, rate-limit headers
│  reverse proxy   │
└─────────┬────────┘
          │ HTTP :8001 (internal)
          ▼
┌──────────────────────────────────────────────────────┐
│  FastAPI wrapper (activity25_serving/api_server.py)  │
│  ├── POST /v1/chat/completions  (proxies to vLLM)    │
│  ├── GET  /health               (vLLM + GPU check)   │
│  ├── GET  /metrics              (Prometheus format)  │
│  └── Middleware:                                     │
│       ├── API key auth (X-API-Key header)            │
│       ├── Rate limiting (token bucket per key)       │
│       └── Request/response logging (JSON)            │
└──────────────────┬───────────────────────────────────┘
                   │ HTTP :8000 (loopback)
                   ▼
┌──────────────────────────────────────────────────────┐
│  vLLM OpenAI-compatible server (unchanged from A1)   │
│  POST /v1/chat/completions  ← proxied                │
│  GET  /health                                        │
│  GET  /v1/models                                     │
│                                                      │
│  NVIDIA A10G GPU                                     │
│  ├── Model weights: INT4 AWQ or FP8 (from Activity 2)│
│  └── KV Cache pages: PagedAttention                  │
└──────────────────────────────────────────────────────┘
```

### Where it runs (deployment options)

Activity 1-4 use Lambda Cloud for hourly benchmarking runs (spend what you need, terminate). Activity 2.5 needs persistent hosting — a server that stays up while you test. Options:

| Option | Cost | Cold start | Best for |
|---|---|---|---|
| Keep Lambda A10G running | ~$18/day ($0.75/hr × 24) | None | Short demos |
| Modal serverless GPU endpoint | Pay-per-request, ~$0.001/req | ~10s | Low-traffic demo |
| RunPod persistent pod | ~$0.44/hr (A10, reserved) | None | Multi-day demo |
| HuggingFace Inference Endpoints | ~$1.70/hr (A10G) | None | Public-facing API |

**Recommendation:** Modal for demos (zero idle cost), Lambda reserved instance for sustained testing.

### Implementation scope (Activity 2.5 deliverables)

```
activity25_serving/
├── config.yaml              # Auth keys, rate limits, vLLM path, ports
├── api_server.py            # FastAPI wrapper with middleware
├── middleware/
│   ├── auth.py              # X-API-Key validation
│   ├── rate_limiter.py      # Token bucket per key
│   └── request_logger.py   # Structured JSON logging
├── systemd/
│   └── loraforge.service    # systemd unit file for auto-restart
├── nginx/
│   └── loraforge.conf       # Nginx reverse proxy config with TLS
└── deploy/
    ├── modal_deploy.py      # Modal serverless deployment
    └── setup_lambda.sh      # Lambda instance setup script
```

**Estimated scope:** ~400 lines of Python, ~50 lines of Nginx config, ~30 lines of systemd config. One half-day of coding after Activity 2 results are in hand.

---

## Activity 5 CLI Serving Architecture (Final Output)

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
