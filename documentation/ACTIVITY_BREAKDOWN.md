# LoraForge — Detailed Activity Breakdown

---

## Activity 0: SFT — Create the Cybersecurity LoRA Adapter

### Objective
Fine-tune Llama-3.1-8B-Instruct on cybersecurity instruction-response data to create our own domain-specific LoRA adapter. This is the prerequisite for all other activities — Activities 1-5 optimize and benchmark this adapter.

### Why It Exists
No public LoRA adapter exists for Llama-3.1-8B-Instruct on cybersecurity. We could use a full fine-tune (like `fdtn-ai/Foundation-Sec-8B`) but that would give us a model, not an adapter — it's a different object and skips the most educational part: understanding what an adapter actually is and how it's trained. Building the adapter ourselves creates a more compelling project narrative: "I identified the gap in public resources, curated the dataset, ran the training pipeline, and then optimized the result."

### The Technical Setup: QLoRA
We use QLoRA (4-bit NF4 base model + LoRA adapters) to fit the 16GB model into ~7GB VRAM during training. This makes training feasible on a single A10G (24GB GPU) at reasonable cost (~$5-8 on Lambda Cloud for the full run).

### What You Build
- **`dataset_builder.py`**: Downloads Trendyol + CVE datasets, formats to ChatML, applies chat template, splits train/eval
- **`qlora_trainer.py`**: Full QLoRA pipeline using BitsAndBytes + PEFT + trl's SFTTrainer. Saves adapter + metrics.
- **`eval.py`**: Qualitative evaluation (response comparison) + perplexity measurement on held-out set

### Inputs
- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- Primary dataset: `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset` (53K examples)
- Secondary dataset: `AlicanKiraz0/All-CVE-Records-Training-Dataset` (30K sampled from 300K)

### Outputs
- `outputs/cybersec_analyst_lora/adapter_config.json` — compatibility contract
- `outputs/cybersec_analyst_lora/adapter_model.safetensors` — trained A and B matrices (~150MB)
- `outputs/activity0_training_metrics.json` — loss curve, eval perplexity, GPU hours

### Key Metrics to Capture
| Metric | Expected Value |
|---|---|
| Training duration | 6-8 hours on A10G |
| Estimated cost | $5-8 on Lambda Cloud |
| Adapter size | ~150 MB |
| Base model perplexity on cybersec QA | 15-25 |
| Fine-tuned perplexity on cybersec QA | 8-15 |
| Perplexity reduction | 20-40% |

### Concepts Learned
- ChatML format and why the template matters for instruction-following models
- The difference between CLM objective (CPT) and masked-loss SFT
- Why B is initialized to zero (adapter starts from base model behavior)
- Catastrophic forgetting and why LoRA helps preserve general capabilities
- QLoRA: NF4 quantization, double quantization, paged optimizers
- Gradient checkpointing: the compute/memory tradeoff

### Key Files
```
activity0_sft/
├── config.yaml               # All hyperparameters — no hardcoded values in code
├── data/
│   ├── __init__.py
│   └── dataset_builder.py    # Download, format (ChatML), cache, split
├── training/
│   ├── __init__.py
│   └── qlora_trainer.py      # QLoRA training pipeline
└── evaluation/
    ├── __init__.py
    └── eval.py               # Perplexity + qualitative response comparison
```

### Run Commands
```bash
# Verify setup (5 steps, ~5 min, no output saved)
python -m activity0_sft.training.qlora_trainer --dry-run

# Full training run (6-8 hours on A10G)
python -m activity0_sft.training.qlora_trainer

# Evaluate trained adapter
python -m activity0_sft.evaluation.eval --adapter-path ./outputs/cybersec_analyst_lora

# Compare adapter vs base model (needs extra VRAM)
python -m activity0_sft.evaluation.eval --adapter-path ./outputs/cybersec_analyst_lora --compare-base
```

---

## Activity 1: Baseline Deployment — Merge, Serve, Benchmark, Report

### Concept to Read First
`CONCEPT_01_MERGE_AND_DEPLOYMENT.md` — the merge equation, PagedAttention internals,
TTFT vs throughput distinction, and vLLM startup sequence.

### Objective
Merge the Activity 0 LoRA adapter into the base model weights, deploy the FP16 merged model
via vLLM, run a structured TTFT + throughput benchmark, evaluate accuracy via perplexity,
and produce `baseline_results.json`. This is the performance floor every later activity beats.

### Why It Exists
You cannot claim you optimized something without a measured starting point. This activity
establishes the baseline — every optimization in Activities 2–4 is expressed as a delta
from these numbers. It also forces deep familiarity with vLLM internals: PagedAttention,
KV block allocation, continuous batching, and the prefill/decode phase distinction.

### What You Build
- **`merge/merge.py`**: Loads base model + LoRA adapter on CPU, calls `peft.merge_and_unload()` (W_merged = W_base + (alpha/r) × B × A), saves as a standard HuggingFace checkpoint (~16GB)
- **`vllm_server.py`**: Launches vLLM as subprocess with OpenAI-compatible API, polls /health until ready, reports GPU block count
- **`benchmark/client.py`**: Async streaming requests at multiple concurrency levels; captures TTFT at first-chunk receive, measures aggregate throughput
- **`evaluation/perplexity.py`**: Computes perplexity on held-out cybersec test set (assistant tokens only), runs 6 qualitative prompts for side-by-side base vs merged comparison
- **`reporting/report_generator.py`**: Reads JSON results → `baseline_results.json` + `baseline_report.md` + `baseline_report.html`
- **`main.py`**: Orchestrates all 5 steps; stops vLLM server before eval step to free VRAM

### Inputs
- LoRA adapter: `./outputs/cybersec_analyst_lora` (output from Activity 0)
- Base model: `meta-llama/Llama-3.1-8B-Instruct` (downloaded from HuggingFace)
- Test data: Trendyol held-out 5% split (500 examples for perplexity)

### Outputs
```
./outputs/cybersec_analyst_merged/     ← merged FP16 model (~16GB)
./results/baseline_results.json        ← TTFT P50/P95, throughput by concurrency
./results/accuracy_eval_results.json   ← perplexity base vs merged, qualitative responses
./results/baseline_report.md           ← human-readable summary with tables
./results/baseline_report.html         ← HTML version
```

### Key Metrics to Capture
| Metric | Concurrency 1 | Concurrency 4 | Concurrency 8 | Concurrency 16 |
|---|---|---|---|---|
| TTFT P50 (ms) | fill after run | fill after run | fill after run | fill after run |
| TTFT P95 (ms) | fill after run | fill after run | fill after run | fill after run |
| Throughput (tok/s) | fill after run | fill after run | fill after run | fill after run |
| GPU VRAM | ~16.8GB | ~16.8GB | ~16.8GB | ~16.8GB |
| Base model perplexity | ~18–25 | — | — | — |
| Merged model perplexity | ~9–15 | — | — | — |

### Concepts Learned
- The merge equation: W_merged = W_base + (alpha/r) × B × A — one-time matrix addition per layer
- Why merge on CPU: merge is pure matrix math, no forward pass needed
- PagedAttention: fixed-size KV pages + page table = no fragmentation, near-100% VRAM utilization
- Continuous batching: inserting new requests mid-generation keeps GPU utilization near 100%
- TTFT = driven by prefill compute (input prompt through all 32 layers)
- Throughput = driven by memory bandwidth (loading 16GB weights for each decode step)
- The TTFT/throughput tradeoff: larger batches increase throughput but increase TTFT
- GPU blocks in vLLM logs: "GPU blocks: N" tells you exactly how much KV cache space you have

### Key Files (all implemented)
```
activity1_baseline/
├── config.yaml              # All settings: merge, server, benchmark, eval, reporting
├── main.py                  # Pipeline orchestrator: merge → serve → bench → eval → report
├── vllm_server.py           # Subprocess-based vLLM launch with health-check polling
├── merge/
│   ├── __init__.py
│   └── merge.py             # CPU merge, validation, progress logging
├── benchmark/
│   ├── __init__.py
│   └── client.py            # Async streaming client, TTFT measurement, throughput stats
├── evaluation/
│   ├── __init__.py
│   └── perplexity.py        # Perplexity on cybersec test set (masked loss on assistant tokens)
└── reporting/
    ├── __init__.py
    └── report_generator.py  # baseline_results.json + markdown + HTML
```

### Run Commands
```bash
# Full pipeline (recommended first time):
python -m activity1_baseline.main

# Skip merge if merged model already exists:
python -m activity1_baseline.main --skip-merge

# Quick test (reduced sweep, 5 requests per config):
python -m activity1_baseline.main --quick

# Individual steps (for debugging or re-running):
python -m activity1_baseline.merge.merge
python -m activity1_baseline.vllm_server                # Ctrl+C to stop
python -m activity1_baseline.benchmark.client --quick
python -m activity1_baseline.evaluation.perplexity
python -m activity1_baseline.reporting.report_generator
```

### Wall Time Estimate (Lambda Cloud A10G)
```
Step 1 Merge:      8–12 minutes  (load 16GB weights, save merged model)
Step 2 Server:     2–4 minutes   (vLLM startup + CUDA graph compilation)
Step 3 Benchmark:  30–60 minutes (full sweep: 5 concurrencies × 3 prompts × 2 outputs × 20 requests)
Step 4 Eval:       20–40 minutes (perplexity on 500 examples + qualitative responses)
Step 5 Report:     <1 minute
TOTAL:             ~60–120 minutes
```

---

## Activity 2: Quantization Pipeline

### Objective
Apply Post-Training Quantization (PTQ) to the LoRA-fine-tuned model using NVIDIA ModelOpt. Profile three quantization formats, benchmark each against the FP16 baseline, and automatically select the best format for a given accuracy-memory budget.

### Why It Exists
Quantization is the single highest-ROI optimization in LLM inference. Moving from FP16 to INT4 AWQ halves memory footprint and increases memory bandwidth utilization, directly improving throughput. The challenge is that different formats suit different accuracy tolerances and hardware targets — no single format is universally best. This activity systematizes that decision.

### The Real-World Problem It Solves
A team deploys Llama-3.1-8B fine-tuned for healthcare QA. In FP16 it needs 16GB VRAM — only fits on a single A100 40GB, expensive. INT4 AWQ fits in 4GB — runs on much cheaper hardware. But nobody has measured how much accuracy is lost on their specific domain task. LoraForge measures it automatically.

### What You Build
- **`QuantizationProfiler`**: Runs all three PTQ formats (FP8, INT4-AWQ, INT8-SQ) using ModelOpt calibration API
- **`CalibrationDatasetBuilder`**: Constructs a 512-sample calibration dataloader from the domain corpus
- **`QuantizationBenchmarker`**: For each quantized model, runs the same benchmark suite as Activity 1
- **`FormatSelector`**: Given accuracy budget (e.g., max 2% MMLU loss), returns the most memory-efficient format
- **`QuantizationReporter`**: Produces the comparison table (identical to NVIDIA's Llama benchmark slide structure)

### Quantization Formats Implemented
1. **FP8 (E4M3)** — Per-tensor FP8 weights and activations. Near-zero accuracy loss. Hardware: Ada, Hopper, Blackwell
2. **INT4 AWQ (W4A16)** — Blockwise INT4 weights, FP16 activations. Default algorithm: AWQ. Best memory reduction
3. **INT8 SmoothQuant (W8A8)** — Per-channel INT8 weights, per-tensor INT8 activations. Balanced trade-off

### Inputs
- Merged FP16 model (from Activity 1 output) OR direct model path
- Calibration dataset (512 samples from domain corpus)
- Accuracy budget config (default: max 2% MMLU loss)

### Outputs
- Three quantized checkpoints: `checkpoints/fp8/`, `checkpoints/int4_awq/`, `checkpoints/int8_sq/`
- `quantization_comparison.json` — metrics for all three formats
- `best_quant_config.json` — selected format + reasoning
- `quantization_report.md` — comparison table with all metrics

### Expected Results (Llama-3.1-8B)
| Format | VRAM | MMLU Loss | TTFT speedup | Throughput gain |
|---|---|---|---|---|
| FP16 (baseline) | ~16GB | 0% | 1x | 1x |
| FP8 | ~8GB | ~0.5% | 1.4x | 1.8x |
| INT4 AWQ | ~4GB | ~2–4% | 1.8x | 2.5x |
| INT8 SQ | ~8GB | ~1–2% | 1.3x | 1.5x |

### Key ModelOpt Code Pattern
```python
import modelopt.torch.quantization as mtq

# Calibration dataloader (512 samples is sufficient for PTQ)
calib_dataloader = build_calib_dataloader(dataset, num_samples=512)

def forward_loop(model):
    for data in calib_dataloader:
        model(data)

# Apply INT4 AWQ
quantized_model = mtq.quantize(model, mtq.INT4_AWQ_CFG, forward_loop)

# Export for TRT-LLM deployment
from modelopt.torch.export import export_tensorrt_llm_checkpoint
export_tensorrt_llm_checkpoint(quantized_model, output_dir="checkpoints/int4_awq/")
```

### Concepts Learned
- PTQ vs QAT: why PTQ is sufficient for most deployments
- AWQ (Activation-aware Weight Quantization): why it outperforms naive rounding
- SmoothQuant: migrating quantization difficulty from activations to weights
- Calibration datasets: why 512 samples is enough and what makes a good calibration set
- Hardware mapping: which formats need Hopper/Ada vs which work on older GPUs

### Key Files
```
activity2_quantization/
├── main.py
├── config.yaml
├── quantizer/
│   ├── modelopt_quantizer.py    # QuantizationProfiler: wraps ModelOpt API
│   ├── calibration.py           # CalibrationDatasetBuilder
│   └── format_selector.py       # FormatSelector: accuracy-budget-aware selection
├── benchmark/
│   └── quant_benchmarker.py     # QuantizationBenchmarker
└── reporting/
    └── comparison_reporter.py   # Produces comparison table + best_quant_config.json
```

---

## Activity 3: KV Cache Optimization

### Objective
Configure KV cache management in vLLM specifically for multi-turn workloads with shared system prompts. Implement prefix caching, quantized KV cache (INT8/FP8), and evaluate sliding window attention for long-context conversations.

### Why It Exists
The KV cache is the memory bottleneck for long-context and high-concurrency inference. For a system like the AI Tutor (a multi-turn system where all users share the same large SOP system prompt), prefix caching alone can eliminate 70–80% of the prefill computation across conversations. Nobody configures this by default.

### The Real-World Problem It Solves
An enterprise chatbot serves 10,000 employees. Every conversation starts with a 2,000-token system prompt. Without prefix caching, that 2,000-token prefill runs for EVERY conversation. With prefix caching, it runs ONCE and the KV tensors are reused. This directly reduces TTFT for every user and halves GPU compute cost at scale.

### What You Build
- **`PrefixCacheEvaluator`**: Simulates a multi-turn workload where N% of tokens are shared system prompt. Measures TTFT reduction vs no-cache baseline.
- **`KVQuantizationEvaluator`**: Enables INT8/FP8 KV cache in vLLM, measures memory savings and throughput impact
- **`StreamingLLMEvaluator`**: Tests sliding window attention (StreamingLLM) for conversations exceeding context window
- **`MultiTurnSimulator`**: Generates realistic multi-turn conversation traces for benchmarking

### Three Optimizations Implemented

**3a. Prefix Caching (Radix Cache / vLLM `enable_prefix_caching=True`)**
- System prompt KV tensors computed once, cached in GPU memory
- All subsequent conversations reuse cached tensors → zero prefill cost for system prompt
- Measure: TTFT as a function of shared prefix percentage

**3b. Quantized KV Cache (FP8 or INT8)**
- KV tensors stored in lower precision → more concurrent conversations fit in GPU memory
- vLLM config: `kv_cache_dtype="fp8"` or `"int8"`
- Measure: Max concurrent users supported, VRAM freed, any accuracy impact

**3c. StreamingLLM / Sliding Window Attention**
- For conversations that grow beyond context window
- Keeps initial tokens (attention sink) + sliding window of recent tokens
- Avoids OOM on very long conversations
- Measure: TTFT at 4K, 8K, 16K, 32K, 64K context lengths

### Inputs
- Quantized model from Activity 2 (or raw FP16 model)
- Multi-turn conversation traces (synthetic, generated by MultiTurnSimulator)
- System prompt corpus (SOP-style text, 500–3000 tokens)

### Outputs
- `kv_cache_metrics.json` — all three optimization metrics
- `prefix_cache_speedup_curve.png` — TTFT vs shared prefix % curve
- `kv_memory_comparison.png` — VRAM usage with/without quantized KV
- `kv_cache_report.md` — recommendations for this specific workload

### Concepts Learned
- KV cache block management (PagedAttention's physical/virtual block mapping)
- Prefix sharing and radix tree cache organization (SGLang's RadixAttention concept)
- KV quantization: why KV tensors can tolerate more aggressive quantization than weights
- Attention sink: why the first few tokens must be kept even in sliding window approaches
- TTFT vs throughput as separate optimization targets

### Key Files
```
activity3_kv_cache/
├── main.py
├── config.yaml
├── prefix_cache/
│   ├── prefix_evaluator.py      # PrefixCacheEvaluator
│   └── multi_turn_simulator.py  # MultiTurnSimulator: generates conversation traces
├── kv_quantization/
│   └── kv_quant_evaluator.py    # KVQuantizationEvaluator
├── streaming_llm/
│   └── streaming_evaluator.py   # StreamingLLMEvaluator
└── reporting/
    └── kv_reporter.py
```

---

## Activity 4: Speculative Decoding

### Objective
Train an EAGLE-style draft model for the fine-tuned base model. Deploy base + draft together in vLLM. Benchmark the speculative decoding speedup and draft acceptance rate. Demonstrate lossless generation acceleration.

### Why It Exists
Speculative decoding is mathematically lossless — the output distribution is identical to standard autoregressive decoding, but generation is 2–4x faster. It works because the decode phase is memory-bandwidth-bound, not compute-bound. The GPU has spare compute capacity while it waits for KV cache reads. A small draft model uses that spare compute to guess ahead, and the large model verifies multiple tokens in one forward pass.

### The Real-World Problem It Solves
The AI Tutor serves employees who need fast, interactive responses. With speculative decoding, what would take 2 seconds to generate takes under 1 second — without any change to output quality. This is not an approximation. It is mathematically equivalent.

### What You Build
- **`EAGLEDraftTrainer`**: Trains the EAGLE draft model using the fine-tuned base model's hidden states
- **`SpeculativeDecodingServer`**: Deploys base + draft together using vLLM's speculative decoding mode
- **`AcceptanceRateBenchmarker`**: Measures draft token acceptance rate at varying temperatures and batch sizes
- **`SpeedupAnalyzer`**: Benchmarks generation speedup vs vanilla autoregressive at batch 1, 4, 8, 16

### How EAGLE Works (What You Learn)
EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) trains a small auxiliary decoder that:
1. Takes the last hidden state of the base model as input (not just the token embedding)
2. Predicts the next K tokens (K=4 by default)
3. Base model verifies all K predictions in one forward pass (parallel, not sequential)
4. All accepted tokens are added to the output at once
5. Net effect: ~K tokens generated per forward pass instead of 1

Training requires ~70K conversation samples, ~1–2 days on 4x A100s. For a side project, you can use a pre-existing EAGLE draft for Llama-3.1-8B from the EAGLE GitHub repo and fine-tune it for your domain.

### Inputs
- Quantized model from Activity 2
- Training corpus: ~50K–70K conversations (public dataset: OpenHermes-2.5, ShareGPT, or domain-specific)
- Pre-trained EAGLE draft (optional: start from existing Llama-3.1-8B EAGLE draft on GitHub)

### Outputs
- `eagle_draft_model/` — trained draft model checkpoint
- `speculative_metrics.json` — acceptance rate, speedup at each batch size
- `acceptance_rate_curve.png` — acceptance rate vs temperature
- `speedup_comparison.png` — speculative vs standard at batch 1/4/8/16
- `speculative_report.md` — configuration recommendations

### Expected Results
| Batch Size | Acceptance Rate | Speedup |
|---|---|---|
| 1 | ~75% | 2.8–3.5x |
| 4 | ~72% | 2.5–3.0x |
| 8 | ~70% | 2.2–2.7x |
| 16 | ~68% | 1.8–2.2x |

### Key vLLM Config (Speculative Decoding)
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="path/to/quantized_base_model",
    speculative_model="path/to/eagle_draft_model",
    num_speculative_tokens=4,          # Draft ahead 4 tokens
    speculative_draft_tensor_parallel_size=1,
    use_v2_block_manager=True,
)
```

### Concepts Learned
- Memory bandwidth bottleneck in autoregressive decode
- Draft model architecture: why hidden state input is better than token embedding input
- Acceptance rate and what affects it (temperature, domain match, draft model size)
- Lossless property: mathematical proof that speculative decoding preserves output distribution
- Batch size interaction: why speculative decoding remains valuable at high batch sizes

### Key Files
```
activity4_speculative_decoding/
├── main.py
├── config.yaml
├── eagle/
│   ├── draft_trainer.py         # EAGLEDraftTrainer: fine-tune or train from scratch
│   ├── draft_model.py           # EAGLE model architecture definition
│   └── training_data.py         # Build training corpus from conversations
├── serving/
│   └── speculative_server.py    # SpeculativeDecodingServer: vLLM with draft
├── benchmark/
│   ├── acceptance_benchmarker.py  # AcceptanceRateBenchmarker
│   └── speedup_analyzer.py        # SpeedupAnalyzer
└── reporting/
    └── speculative_reporter.py
```

---

## Activity 5: Benchmark Dashboard & CLI Tool

### Objective
Package the entire LoraForge pipeline as a production-grade CLI tool (`loraforge`) and Streamlit dashboard. Aggregate all metrics from Activities 1–4, generate Pareto curves and comparison tables, and publish as a PyPI package.

### Why It Exists
This is the difference between a learning exercise and an open-source tool. The GitHub README becomes: "Install with `pip install loraforge`. Run with `loraforge optimize --model meta-llama/Llama-3.1-8B-Instruct --adapter your-lora-adapter`. Get a full optimization report in 20 minutes." That is a tool, not a notebook.

### What You Build
- **`loraforge` CLI**: Typer-based CLI with commands: `optimize`, `benchmark`, `serve`, `report`
- **`OptimizationOrchestrator`**: Runs Activities 1–4 in sequence, collects all metrics
- **`ReportAggregator`**: Merges all metric JSONs into unified `benchmark_report.json`
- **`DashboardApp`**: Streamlit app with: Pareto curves (accuracy vs memory, accuracy vs latency), comparison tables, cost estimator, configuration recommender
- **PyPI package**: `setup.py` + `pyproject.toml`, published to PyPI

### Dashboard Panels
1. **Summary Panel**: Model name, adapter name, best configuration, key improvements (memory saved %, speedup achieved, accuracy preserved)
2. **Quantization Comparison**: Table + bar charts for all three formats
3. **KV Cache Impact**: TTFT curves across prefix sharing percentages and context lengths
4. **Speculative Decoding**: Acceptance rate curve, speedup vs batch size
5. **Pareto Curves**: Accuracy loss vs memory reduction, accuracy loss vs throughput gain
6. **Cost Estimator**: Cost/1M tokens on Lambda Cloud (A10, A100, H100) for each configuration

### CLI Design
```bash
# Full pipeline: profile, quantize, optimize KV, apply speculative decoding
loraforge optimize \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapter your-org/your-lora-adapter \
  --task medical-qa \
  --accuracy-budget 0.02 \          # Max 2% accuracy loss
  --output-dir ./loraforge-results

# View results dashboard
loraforge report --results-dir ./loraforge-results

# Serve the optimized model
loraforge serve --results-dir ./loraforge-results --port 8000
```

### Key Files
```
activity5_benchmark_dashboard/
├── cli/
│   ├── main.py                  # Typer CLI entry point
│   ├── optimize.py              # `loraforge optimize` command
│   ├── benchmark.py             # `loraforge benchmark` command
│   ├── serve.py                 # `loraforge serve` command
│   └── report.py                # `loraforge report` command
├── orchestrator/
│   └── pipeline_orchestrator.py # OptimizationOrchestrator
├── reporting/
│   ├── aggregator.py            # ReportAggregator
│   └── dashboard/
│       ├── app.py               # Streamlit app: DashboardApp
│       ├── panels/              # One file per dashboard panel
│       └── cost_estimator.py    # Cost/1M tokens calculation
├── setup.py
├── pyproject.toml
└── README.md                    # PyPI README
```

---

## Activity 6: Domain-Adaptive Continued Pre-Training

### Objective
Implement LoRA-based continued pre-training (CPT) on a domain corpus before fine-tuning. Demonstrate that domain adaptation before SFT improves downstream accuracy, and run the CPT-adapted model through the full LoraForge optimization pipeline.

### Why It Exists
Large pre-trained models lack deep domain knowledge in specialized fields (medical, legal, cybersecurity). Continued pre-training on domain text before SFT teaches the model the vocabulary, reasoning patterns, and factual knowledge of the domain. This improves the quality ceiling that SFT can reach.

The second purpose: it adds "continuous pre-training" to your skill set — a topic you identified as a gap — without requiring expensive full-parameter training.

### The Real-World Problem It Solves
You want to deploy an LLM for cybersecurity threat analysis. The base Llama-3.1-8B doesn't know recent CVEs, threat actor TTPs, or security ontologies. Full CPT costs millions of dollars. LoRA-based CPT on 1–5B domain tokens costs a few hundred dollars and captures most of the domain knowledge gain.

### What You Build
- **`DomainCorpusBuilder`**: Downloads and preprocesses domain corpus (options: CyberSecEval, PubMed abstracts, Nemotron-Personas-India)
- **`LoRACPTTrainer`**: Runs LoRA-based CPT using HuggingFace Trainer with causal language modeling objective
- **`ForgettingEvaluator`**: Measures base-knowledge retention before and after CPT (MMLU, general reasoning)
- **`DomainGainEvaluator`**: Measures domain-specific accuracy improvement on held-out domain test set
- **`CPTPipeline`**: Chains CPT → SFT → LoraForge optimization pipeline end-to-end

### Why LoRA for CPT (Not Full Fine-Tuning)
- Full CPT on 8B model: requires 8x A100 for weeks, costs thousands of dollars
- LoRA CPT: runs on single A100 40GB, completes in 12–24 hours for a small domain corpus (1B tokens)
- Catastrophic forgetting: LoRA CPT preserves base model knowledge better than full parameter updates
- Practical: as a senior engineer with a full-time job, LoRA CPT is the only feasible approach

### Recommended Domain Corpus Options (Pick One)
1. **Cybersecurity** (most relevant to your Cisco background): CyberSecEval + public CVE descriptions + MITRE ATT&CK documentation (~500M tokens available)
2. **Healthcare**: PubMed abstracts (~2B tokens, well-structured) — relevant to your Optum work
3. **India-specific sovereign AI**: NVIDIA Nemotron-Personas-India dataset (7.7B tokens, CC BY 4.0 license) — ties directly to the NVIDIA workshop, strong LinkedIn narrative

### Inputs
- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- Domain corpus: Selected from above options
- CPT LoRA config: rank=16, alpha=32, target modules=all linear, learning rate=2e-5

### Outputs
- `cpt_lora_adapter/` — CPT LoRA adapter checkpoint
- `cpt_evaluation.json` — base knowledge retention + domain gain metrics
- CPT-adapted model fed into Activities 1–5 for full optimization pipeline
- `cpt_report.md` — what was learned, what was forgotten, net domain gain

### Concepts Learned
- Continued pre-training vs supervised fine-tuning: different objectives, different data
- Catastrophic forgetting: why models lose base knowledge during domain adaptation
- LoRA for CPT: how low-rank updates preserve base weights while learning domain patterns
- Replay buffers: simple technique to mix domain data with general data to prevent forgetting
- Two-stage training: CPT (causal LM objective) → SFT (instruction following objective)

### Key Files
```
activity6_domain_cpt/
├── main.py
├── config.yaml
├── corpus/
│   ├── corpus_builder.py        # DomainCorpusBuilder: download + preprocess
│   └── data_mixer.py            # Mix domain data with replay buffer (general data)
├── training/
│   ├── cpt_trainer.py           # LoRACPTTrainer: HF Trainer with CLM objective
│   └── cpt_config.py            # LoRA config, training args, scheduler
├── evaluation/
│   ├── forgetting_evaluator.py  # ForgettingEvaluator: MMLU before/after
│   └── domain_evaluator.py      # DomainGainEvaluator: domain test set accuracy
└── reporting/
    └── cpt_reporter.py
```

### Optional Extension: Full-Parameter Distributed CPT (Multi-GPU)

The plan above uses LoRA CPT on a single GPU — which is fine-tuning, not pre-training. If you want to demonstrate genuine distributed deep learning at the foundational model level, Activity 6 can be upgraded to run **full-parameter CPT across multiple GPUs** using PyTorch FSDP (Fully Sharded Data Parallel) via HuggingFace Accelerate.

**What this changes:**

| | LoRA CPT (default plan) | FSDP Full-Parameter CPT (optional) |
|---|---|---|
| What gets updated | LoRA A/B matrices only (~0.5% of params) | All 8B parameters |
| Parallelism | Single GPU | Multi-GPU (FSDP shards params + gradients + optimizer state) |
| Memory per GPU | ~18 GB (with 4-bit base + LoRA) | ~24 GB (BF16 full model, sharded across 4× A10G) |
| Lambda instance | 1× A10G at $0.75/hr | 4× A10G at ~$3.00/hr |
| Est. cost for 2K-step run | ~$1–2 | ~$12–18 |
| Resume claim | "LoRA domain adaptation" | "Distributed continued pre-training with FSDP across multiple GPUs" |

**What FSDP does:** Each GPU holds only 1/N of the model parameters (N = number of GPUs). Before each forward and backward pass, FSDP gathers the full layer, computes, then immediately releases it. Gradients and optimizer states are similarly sharded. No GPU sees more than a fraction of the model at any time.

**You do not need to run for weeks on a trillion tokens.** Running FSDP across 4 GPUs for 2,000 steps on a curated cybersecurity corpus produces real, runnable code with real per-rank training dynamics, real loss curves, and a real checkpoint consolidation step. The fact that you stop at 2K steps rather than 1T tokens does not make the distributed training experience any less authentic.

**Skills demonstrated:** FSDP configuration, per-rank data loading, mixed-precision BF16 across ranks, gradient norm clipping in distributed context, checkpoint consolidation with `FSDP.save_state_dict`, and the `accelerate config` / `accelerate launch` workflow.

**Decision:** Keep LoRA CPT as the primary implementation (it is practical and ships the project). Revisit the FSDP upgrade when all 5 core activities are complete. If done, this becomes the single strongest technical talking point in the entire project.
