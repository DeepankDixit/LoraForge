# LoraForge: End-to-End Inference Optimization Pipeline for Fine-Tuned LLMs

## Project Vision

LoraForge is an open-source, production-grade framework that takes any LoRA fine-tuned language model and automatically produces a fully optimized, benchmarked, and deployment-ready inference artifact.

- LoraForge is an automated inference optimization pipeline for anyone with a LoRA adapter. The contract is: bring an adapter trained on meta-llama/Llama-3.1-8B-Instruct, and LoraForge runs the entire optimization suite for you — merge, deploy, benchmark baseline, quantize, optimize KV cache, apply speculative decoding, and produce a dashboard comparing all variants.
- The domain does not matter. Cybersecurity is what WE use to build and demo the pipeline. Someone with a medical Q&A adapter, a legal document adapter, or a code generation adapter could plug theirs in and get the same treatment. The pipeline never looks at what the adapter was trained on — it only checks that the base model matches.

The core thesis: **Millions of LoRA fine-tuned models exist. None come with production inference optimization. LoraForge fills that gap.**

---

## The Problem This Solves

When a team fine-tunes a model with LoRA (e.g., Llama-3.1-8B for domain-specific QA), they get good quality but poor inference performance because:

1. Deployment defaults to naive FP16 — 2x–4x more memory than necessary
2. No automated way to find the right quantization format (FP8 vs INT4 AWQ vs INT8 SmoothQuant)
3. KV cache is not configured for the specific workload (e.g., shared system prompts in multi-turn systems)
4. Speculative decoding — a lossless 2–4x speedup — is never applied because setup is manual
5. There is no reproducible benchmark to prove optimization worked

LoraForge automates all of this in a single pipeline with one entry point.

---

## Target Audience

- ML engineers deploying fine-tuned LLMs in production
- Research teams that need reproducible inference benchmarks
- Startups and enterprises running domain-adapted models on rented GPUs
- The broader HuggingFace community (any compatible model works)

---

## Core Innovation

LoraForge is the first open-source tool to treat LoRA-fine-tuned model inference optimization as a *pipeline*, not a collection of disconnected scripts. It covers:

- **Quantization profiling** — automatically selects the best quantization format for your model + task
- **KV cache optimization** — configures prefix caching, quantized KV, and window attention for your workload
- **Speculative decoding** — auto-provisions a draft model and benchmarks acceptance rates
- **Domain-adaptive CPT** — LoRA-based continued pre-training before the inference pipeline
- **Benchmark dashboard** — reproducible Pareto curves anyone can run and verify

---

## Technical Stack

| Component | Technology |
|---|---|
| Base models | Llama-3.1-8B, Qwen-2.5-7B (HuggingFace) |
| Fine-tuning | LoRA / QLoRA via HuggingFace PEFT |
| Quantization | NVIDIA ModelOpt (nvidia-modelopt), AutoAWQ, AutoGPTQ |
| Serving | vLLM (primary), TensorRT-LLM (export target) |
| Speculative Decoding | EAGLE (EAGLE-2/EAGLE-3) |
| CPT Framework | HuggingFace Trainer + NeMo (optional) |
| Benchmarking | Custom harness + lm-evaluation-harness |
| Dashboard | Streamlit + Plotly |
| Packaging | PyPI (pip install loraforge) |
| CI/CD | GitHub Actions |

---

## Repository Structure

```
LoraForge/
├── documentation/
│   ├── PROJECT_OVERVIEW.md          # This file
│   ├── ARCHITECTURE.md              # System design and data flow
│   ├── ACTIVITY_BREAKDOWN.md        # Detailed breakdown of all 6 activities
│   ├── GPU_REQUIREMENTS.md          # Compute requirements and cost estimates
│   └── LINKEDIN_NARRATIVE.md        # How to present this project publicly
├── code/
│   ├── CLAUDE.md                    # Master instructions for Claude Code
│   ├── shared/
│   │   ├── utils/                   # Logging, config parsing, GPU profiling
│   │   ├── configs/                 # YAML configs for all activities
│   │   ├── models/                  # Model loading, merging, export utilities
│   │   └── metrics/                 # Perplexity, latency, throughput, MMLU
│   ├── activity1_baseline/          # Baseline vLLM serving stack
│   ├── activity2_quantization/      # PTQ with ModelOpt (FP8, INT4 AWQ, INT8-SQ)
│   ├── activity3_kv_cache/          # Prefix caching, quantized KV, StreamingLLM
│   ├── activity4_speculative_decoding/ # EAGLE draft model training + benchmarking
│   ├── activity5_benchmark_dashboard/  # CLI tool + Streamlit dashboard
│   └── activity6_domain_cpt/        # LoRA-based continued pre-training
└── README.md                        # Public-facing GitHub README
```

---

## Primary Model Targets

### Llama-3.1-8B (Primary)
- **Why:** Real production experience — Deepank fine-tuned this model at Optum using LoRA (q_proj/v_proj across 32 transformer blocks, <0.1% trainable params). Can speak to every architectural detail.
- **Public adapter:** Multiple available on HuggingFace (OpenHermes, medical QA, security reasoning)
- **Architecture:** 32 transformer blocks, GQA (8 KV heads, 32 query heads), 128K context window

### Qwen-2.5-7B (Secondary)
- **Why:** Shows pipeline generalization beyond one model family. Strong on multilingual + code tasks.
- **Architectural difference:** Different GQA config, different RoPE implementation — good test of pipeline robustness

---

## Key Metrics Tracked

| Metric | Description | Target |
|---|---|---|
| TTFT (P50/P90/P99) | Time to First Token | Reduce by 50–70% |
| Throughput | Tokens/second at batch 1/4/8/16 | Increase by 2–4x |
| GPU Memory | Peak VRAM usage | Reduce by 40–60% |
| Perplexity delta | vs FP16 baseline | < 0.5 |
| MMLU accuracy delta | vs FP16 baseline | < 2% |
| Speculative acceptance rate | Draft token acceptance % | > 60% target |
| Cost per 1M tokens | Lambda Cloud pricing | Track reduction |

---

## Phased Timeline (Side Project, ~9 Months)

| Month | Activity | Milestone |
|---|---|---|
| 1 | Activity 1: Baseline | Baseline metrics established, vLLM serving running ✅ |
| 2–3 | Activity 2: Quantization | INT4 AWQ + FP8 benchmarks, comparison table complete |
| 3 | Activity 2.5: Production Serving | Live HTTPS endpoint with auth + logging + systemd + Modal deploy |
| 3–4 | Activity 3: KV Cache | Prefix caching + quantized KV benchmarks complete |
| 4–5 | Activity 4: Speculative Decoding | EAGLE draft trained, acceptance rate measured |
| 5–6 | Activity 5: Dashboard | CLI tool + PyPI package + dashboard live |
| 6–7 | Activity 6: CPT | Domain-adapted model run through full pipeline |
| 7–9 | Polish + Share | GitHub README, LinkedIn article, meetup talk |

---

## Open Source Strategy

1. **GitHub repo:** `deepank-dixit/loraforge` — MIT license
2. **PyPI package:** `pip install loraforge` — single CLI entry point
3. **HuggingFace Spaces:** Host interactive demo of benchmark dashboard
4. **LinkedIn article:** "How I optimized a LoRA fine-tuned LLM for production inference" — deep technical post with benchmark numbers
5. **Meetup talk:** Present at Bengaluru AI/ML meetup with slides derived from this project
6. **Blog post:** Optional cross-post to Towards Data Science / Medium

---

## Recruiter Signal This Project Sends

Completing this project demonstrates, in one artifact:
- Deep understanding of LLM inference: prefill/decode phases, KV cache, batching
- Hands-on quantization: PTQ calibration, accuracy/latency tradeoffs, format selection
- Speculative decoding: draft model training, acceptance rate, lossless speedup
- Production engineering: CLI tooling, PyPI packaging, reproducible benchmarks
- End-to-end ML thinking: from training (CPT) through serving (optimized deployment)

This is the **complete inference engineering lifecycle** — exactly what NVIDIA, Google DeepMind, Meta, Anthropic, Cohere, and Mistral hire for in senior/staff inference engineer roles.
