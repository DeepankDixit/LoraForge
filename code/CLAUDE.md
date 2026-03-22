# CLAUDE.md — LoraForge Master Instructions for Claude Code

## Project Identity

**Name:** LoraForge
**Purpose:** End-to-end inference optimization pipeline for LoRA fine-tuned LLMs
**Owner:** Deepank Dixit (Senior AI/ML Engineer, IISc MTech AI)
**Primary Model:** Llama-3.1-8B-Instruct + LoRA adapters
**Secondary Model:** Qwen-2.5-7B-Instruct

---

## Non-Negotiable Code Standards

Every file you write MUST follow these standards without exception:

1. **Type hints everywhere.** Every function parameter and return value must have type annotations.
2. **Docstrings on every class and public method.** Google-style docstrings.
3. **Structured logging.** Use `shared/utils/logger.py` — never use `print()` in production code.
4. **Config-driven.** No hardcoded model paths, hyperparameters, or magic numbers. Everything in YAML configs loaded via `shared/utils/config_parser.py`.
5. **Error handling.** Wrap GPU operations in try/except with meaningful error messages. Handle OOM gracefully.
6. **Reproducibility.** Set seeds everywhere: `torch.manual_seed(config.seed)`, `random.seed(config.seed)`, `np.random.seed(config.seed)`.
7. **Checkpointing.** All training loops must save checkpoints. All benchmark runs must save results to timestamped directories.
8. **One responsibility per class.** Classes do one thing. No God classes.
9. **Tests.** Every module must have a corresponding test file in `tests/`. Use pytest.
10. **Requirements pinned.** All dependencies in `requirements.txt` must be pinned to exact versions.

---

## Repository Layout

```
code/
├── CLAUDE.md                    ← You are reading this
├── pyproject.toml               ← Package metadata + build config
├── setup.py                     ← PyPI packaging
├── requirements.txt             ← Pinned dependencies
├── requirements-dev.txt         ← Dev dependencies (pytest, black, mypy)
├── .env.example                 ← Template for environment variables
├── Makefile                     ← Common commands: make test, make lint, make benchmark
├── tests/                       ← pytest test suite (mirrors code structure)
│
├── shared/                      ← Shared utilities used by all activities
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py            ← Structured logging (JSON + human-readable)
│   │   ├── config_parser.py     ← YAML config loader with validation
│   │   ├── gpu_profiler.py      ← VRAM, utilization, bandwidth tracking
│   │   └── checkpoint_manager.py ← Save/load/list checkpoints
│   ├── configs/
│   │   ├── base_config.yaml     ← Default pipeline configuration
│   │   ├── llama31_8b.yaml      ← Llama-3.1-8B specific overrides
│   │   ├── qwen25_7b.yaml       ← Qwen-2.5-7B specific overrides
│   │   ├── quantization_recipes.yaml ← PTQ recipes for all formats
│   │   └── serving_config.yaml  ← vLLM serving parameters
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py      ← Load base + LoRA, merge, push to Hub
│   │   ├── model_exporter.py    ← Export to TRT-LLM / ONNX
│   │   └── lora_utils.py        ← LoRA inspection, adapter merging
│   └── metrics/
│       ├── __init__.py
│       ├── latency_metrics.py   ← TTFT P50/P90/P99, TPOT
│       ├── throughput_metrics.py ← tokens/sec at varying batch sizes
│       ├── accuracy_metrics.py  ← Perplexity, MMLU subset, task accuracy
│       ├── memory_metrics.py    ← Peak VRAM, model weight footprint
│       └── cost_estimator.py    ← Cost/1M tokens on Lambda/RunPod
│
├── activity0_sft/               ← NEW: Create our own cybersecurity LoRA adapter
│   ├── config.yaml              ← QLoRA training config (rank=16, alpha=32)
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset_builder.py   ← Download Trendyol + CVE datasets, format to ChatML
│   ├── training/
│   │   ├── __init__.py
│   │   └── qlora_trainer.py     ← QLoRA training pipeline (BitsAndBytes + PEFT + trl)
│   └── evaluation/
│       ├── __init__.py
│       └── eval.py              ← Perplexity + qualitative response comparison
│
├── activity1_baseline/              ← Deploy merged FP16 model; measure TTFT + throughput baseline
│   ├── config.yaml                  ← Merge, server, benchmark, eval, reporting config
│   ├── main.py                      ← Full pipeline orchestrator (merge→serve→benchmark→eval→report)
│   ├── vllm_server.py               ← Launch vLLM with OpenAI-compatible API
│   ├── merge/
│   │   ├── __init__.py
│   │   └── merge.py                 ← peft.merge_and_unload() → save FP16 merged model
│   ├── benchmark/
│   │   ├── __init__.py
│   │   └── client.py                ← Async TTFT + throughput sweep (concurrency × prompt_len)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── perplexity.py            ← Perplexity on cybersec test set + qualitative eval
│   └── reporting/
│       ├── __init__.py
│       └── report_generator.py      ← baseline_results.json + baseline_report.md + HTML
│
├── activity2_quantization/
├── activity3_kv_cache/
├── activity4_speculative_decoding/
├── activity5_benchmark_dashboard/
└── activity6_domain_cpt/
```

---

## Shared Modules — Build These First

### shared/utils/logger.py
Build a logger that:
- Outputs JSON to file (for programmatic parsing of benchmark results)
- Outputs human-readable colored text to console
- Automatically includes: timestamp, activity name, function name, GPU memory at time of log
- Usage: `from shared.utils.logger import get_logger; logger = get_logger(__name__)`

### shared/utils/config_parser.py
Build a config parser that:
- Loads YAML files using `pydantic` BaseModel for validation
- Supports hierarchical config merging (base_config.yaml overridden by model-specific yaml)
- Raises clear errors for missing required fields
- Exposes a `Config` dataclass with typed fields

### shared/utils/gpu_profiler.py
Build a GPU profiler that:
- Uses `pynvml` or `torch.cuda.memory_stats()` to track VRAM
- Provides a context manager: `with GPUProfiler() as prof: ...` → returns `prof.peak_vram_gb`
- Tracks GPU utilization % over time (poll every second in background thread)
- Provides `get_memory_breakdown()`: model weights vs KV cache vs activations

### shared/metrics/latency_metrics.py
Build a latency tracker that:
- Measures Time to First Token (TTFT): time from request sent to first token received
- Measures Time Per Output Token (TPOT): average time between subsequent tokens
- Measures end-to-end latency
- Computes P50, P90, P99 across N runs
- Returns `LatencyStats` dataclass with all percentiles

### shared/models/model_loader.py
Build a model loader that:
- Loads base model in specified dtype (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
- Loads and merges LoRA adapter using PEFT's `merge_and_unload()`
- Saves merged model to disk in HuggingFace format
- Optionally pushes to HuggingFace Hub
- Handles both local paths and HuggingFace Hub IDs

---

## Activity 0: SFT — Create the Cybersecurity LoRA Adapter

### Objective
Fine-tune Llama-3.1-8B-Instruct on cybersecurity instruction-response data using QLoRA.
Produce the `cybersec_analyst_lora/` adapter that feeds into all subsequent activities.
No public LoRA adapter exists for Llama-3.1-8B-Instruct + cybersecurity — we build our own.

### Entry Point
```bash
python -m activity0_sft.training.qlora_trainer --config activity0_sft/config.yaml
python -m activity0_sft.training.qlora_trainer --config activity0_sft/config.yaml --dry-run
python -m activity0_sft.evaluation.eval --adapter-path ./outputs/cybersec_analyst_lora
```

### Datasets
- **Primary:** `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset` — 53K examples, 200+ security domains
- **Secondary:** `AlicanKiraz0/All-CVE-Records-Training-Dataset` — sample 30K of 300K CVE chat records
- **Total training examples:** ~83K after filtering
- **Evaluation only:** `walledai/CyberSecEval` (do NOT train on this)

### Training Configuration (activity0_sft/config.yaml)
```yaml
model:
  base_model_id: "meta-llama/Llama-3.1-8B-Instruct"
  max_seq_length: 2048
  use_flash_attention: true

quantization:
  load_in_4bit: true              # NF4 quantization for base model (~4GB vs 16GB)
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true

lora:
  r: 16
  lora_alpha: 32                  # scaling = alpha/r = 2.0
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  task_type: "CAUSAL_LM"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8  # effective batch size = 32
  learning_rate: 2.0e-4
  optim: "paged_adamw_8bit"
  bf16: true
  gradient_checkpointing: true
  packing: true                   # pack short sequences into 2048-token windows
```

### Key Files (already implemented)
- `activity0_sft/config.yaml` — complete training config
- `activity0_sft/data/dataset_builder.py` — download, format (ChatML), cache, split
- `activity0_sft/training/qlora_trainer.py` — full training pipeline with metrics
- `activity0_sft/evaluation/eval.py` — perplexity + qualitative comparison

### Expected Output
```
./outputs/cybersec_analyst_lora/
  adapter_config.json          ← "base_model: Llama-3.1-8B-Instruct, r=16, alpha=32"
  adapter_model.safetensors    ← A and B matrices (~150MB)
./outputs/activity0_training_metrics.json   ← loss, perplexity, GPU hours
./outputs/tensorboard_logs/                 ← training curves
```

### Expected Results
- Training duration: 6-8 hours on Lambda A10G (24GB)
- Cost: ~$5-8 on Lambda Cloud
- Final eval perplexity: 8-15 (vs 15-25 for base model on cybersecurity QA)
- Adapter size: ~150MB (vs 16GB full model)

### Dependencies to Install
```bash
pip install transformers>=4.44.0 peft>=0.12.0 trl>=0.12.0
pip install bitsandbytes>=0.43.0 accelerate>=0.33.0
pip install datasets omegaconf typer
pip install flash-attn --no-build-isolation  # Optional but recommended
```

---

## Activity 1: Baseline Deployment — Merge, Serve, Benchmark, Report

### Objective
Merge the Activity 0 LoRA adapter into the base model, deploy the FP16 merged model via vLLM,
run a TTFT + throughput benchmark sweep, evaluate perplexity, and produce `baseline_results.json`.
This is the performance floor that Activities 2–5 must beat.

### Concept to Read First
`documentation/CONCEPT_01_MERGE_AND_DEPLOYMENT.md` — explains the merge operation,
PagedAttention, TTFT vs throughput distinction, and vLLM startup sequence.

### Entry Points
```bash
# Full pipeline (recommended):
python -m activity1_baseline.main

# Individual steps:
python -m activity1_baseline.merge.merge                     # Step 1: Merge adapter
python -m activity1_baseline.vllm_server                     # Step 2: Launch server
python -m activity1_baseline.benchmark.client                # Step 3: Run benchmark
python -m activity1_baseline.evaluation.perplexity           # Step 4: Eval perplexity
python -m activity1_baseline.reporting.report_generator      # Step 5: Generate report

# Flags:
python -m activity1_baseline.main --skip-merge               # Reuse existing merged model
python -m activity1_baseline.main --skip-eval                # Skip perplexity (saves 20–40 min)
python -m activity1_baseline.main --quick                    # Reduced sweep for testing
python -m activity1_baseline.main --report-only              # Regenerate report from JSON
```

### Key Files (all implemented)
- `activity1_baseline/config.yaml` — all settings (merge, server, benchmark, eval, reporting)
- `activity1_baseline/merge/merge.py` — `peft.merge_and_unload()` on CPU, saves FP16 checkpoint
- `activity1_baseline/vllm_server.py` — subprocess-based vLLM launch with health-check polling
- `activity1_baseline/benchmark/client.py` — asyncio concurrent requests, streaming TTFT measurement
- `activity1_baseline/evaluation/perplexity.py` — perplexity on cybersec test set, qualitative eval
- `activity1_baseline/reporting/report_generator.py` — baseline_results.json + markdown + HTML report
- `activity1_baseline/main.py` — orchestrates all 5 steps with proper server lifecycle management

### Expected Outputs
```
./outputs/cybersec_analyst_merged/     ← merged FP16 model (~16GB, standard HuggingFace format)
./results/baseline_results.json        ← TTFT P50/P95, throughput by concurrency/prompt length
./results/accuracy_eval_results.json   ← perplexity (base vs merged), qualitative responses
./results/baseline_report.md           ← human-readable summary with interpretation
./results/baseline_report.html         ← HTML version (open in browser)
```

### Expected Baseline Numbers (A10G 24GB)
```
VRAM:                   ~16.8GB model + ~4.8GB KV cache
TTFT P50 (c=1, 512t):   ~150–300ms
TTFT P95 (c=1, 512t):   ~300–600ms
Throughput (c=8):        ~350–550 tok/s
Perplexity (merged):     ~9–15 (vs ~18–25 for base model on cybersec QA)
```

### Dependencies to Install
```bash
pip install vllm>=0.5.0
pip install httpx openai
pip install requests pyyaml
```

### Run Command Reference
```bash
make baseline-merge    # Step 1 only
make baseline-serve    # Step 2 only (foreground, Ctrl+C to stop)
make baseline-bench    # Step 3 only (requires server running)
make baseline-eval     # Step 4 only
make baseline-report   # Step 5 only
make baseline          # All steps via main.py
```

---

## Activity 2: Quantization Pipeline

### Objective
Apply PTQ (FP8, INT4 AWQ, INT8 SmoothQuant) using NVIDIA ModelOpt. Benchmark each. Select best format for given accuracy budget. Output: quantized checkpoint + comparison table.

### Entry Point
`python activity2_quantization/main.py --config activity2_quantization/config.yaml`

### Key Dependency
`pip install nvidia-modelopt[torch]` — NVIDIA ModelOpt library

### Files to Create

**activity2_quantization/config.yaml**
```yaml
model:
  base_model_id: "meta-llama/Llama-3.1-8B-Instruct"
  lora_adapter_id: null  # Set if starting from LoRA adapter
  merged_model_dir: "./checkpoints/llama31_8b_merged_fp16"  # Use Activity 1 output

quantization:
  formats: ["fp8", "int4_awq", "int8_smoothquant"]
  calibration_dataset: "wikitext"
  calibration_num_samples: 512
  calibration_batch_size: 4
  accuracy_budget: 0.02  # Max 2% MMLU accuracy loss tolerated

output:
  checkpoints_dir: "./checkpoints"
  results_dir: "./results/activity2_quantization"
  export_tensorrt_llm: true  # Also export to TRT-LLM format
```

**activity2_quantization/quantizer/modelopt_quantizer.py**
```python
# Class: QuantizationProfiler
# Purpose: Apply PTQ using NVIDIA ModelOpt for multiple quantization formats
#
# Methods:
#   __init__(self, config: QuantizationConfig)
#
#   build_calibration_dataloader(self, dataset_name: str, num_samples: int) → DataLoader
#     - Load dataset, tokenize, return DataLoader for calibration
#     - Use 512 samples as default (sufficient for PTQ)
#
#   quantize_fp8(self, model: nn.Module, calib_dataloader: DataLoader) → nn.Module
#     - Apply FP8 PTQ using modelopt.torch.quantization
#     - Config: per-tensor FP8 weights and activations
#     - Return quantized model
#
#   quantize_int4_awq(self, model: nn.Module, calib_dataloader: DataLoader) → nn.Module
#     - Apply INT4 AWQ using modelopt.torch.quantization with INT4_AWQ_CFG
#     - Config: blockwise INT4 weights, FP16 activations
#     - Return quantized model
#
#   quantize_int8_smoothquant(self, model: nn.Module, calib_dataloader: DataLoader) → nn.Module
#     - Apply INT8 SmoothQuant using modelopt
#     - Config: per-channel INT8 weights, per-tensor INT8 activations
#     - Return quantized model
#
#   save_quantized_checkpoint(self, model: nn.Module, format: str, output_dir: str) → None
#     - Save quantized model to output_dir/{format}/
#     - Also export TensorRT-LLM checkpoint if config.export_tensorrt_llm
#
#   profile_all_formats(self) → Dict[str, QuantizedModel]
#     - Run all three quantization formats
#     - Return dict: {"fp8": model, "int4_awq": model, "int8_sq": model}

# IMPORTANT: Use the following ModelOpt pattern exactly:
#
# import modelopt.torch.quantization as mtq
# from modelopt.torch.export import export_tensorrt_llm_checkpoint
#
# def forward_loop(model):
#     for data in calib_dataloader:
#         model(**data)
#
# quantized_model = mtq.quantize(model, mtq.INT4_AWQ_CFG, forward_loop)
# export_tensorrt_llm_checkpoint(quantized_model, output_dir)
```

**activity2_quantization/quantizer/format_selector.py**
```python
# Class: FormatSelector
# Purpose: Given benchmark results for all formats, select the best one
#
# Methods:
#   __init__(self, accuracy_budget: float, memory_priority: bool = True)
#     - accuracy_budget: max acceptable MMLU accuracy loss (e.g., 0.02 for 2%)
#     - memory_priority: if True, minimize VRAM; if False, maximize throughput
#
#   select(self, benchmark_results: Dict[str, BenchmarkResult]) → SelectionResult
#     - Filter formats where accuracy_loss <= accuracy_budget
#     - Among passing formats, select: lowest VRAM if memory_priority, else highest throughput
#     - Return SelectionResult with: chosen_format, reasoning, tradeoff_summary
#
#   generate_recommendation_text(self, result: SelectionResult) → str
#     - Return human-readable explanation of why this format was chosen
```

---

## Activity 3: KV Cache Optimization

### Objective
Configure prefix caching, quantized KV cache, and StreamingLLM for multi-turn workloads. Measure TTFT reduction and memory savings.

### Entry Point
`python activity3_kv_cache/main.py --config activity3_kv_cache/config.yaml`

**activity3_kv_cache/config.yaml**
```yaml
model:
  model_dir: "./checkpoints/int4_awq"  # Use best quantized model from Activity 2

workload:
  system_prompt_length: 1500  # tokens (simulating SOP system prompt)
  conversation_turns: 10
  num_users: 50
  shared_prefix_percentages: [0, 25, 50, 75, 100]  # % of prompt that is shared

kv_cache:
  enable_prefix_caching: true
  kv_cache_dtype: "fp8"  # Options: "auto" (fp16), "fp8", "int8"
  streaming_llm:
    enabled: true
    attention_sink_size: 4
    window_size: 1024

benchmark:
  context_lengths: [1024, 4096, 8192, 16384]
  num_rounds: 10
  seed: 42

output:
  results_dir: "./results/activity3_kv_cache"
```

**activity3_kv_cache/prefix_cache/prefix_evaluator.py**
```python
# Class: PrefixCacheEvaluator
# Purpose: Measure TTFT reduction from prefix KV cache reuse
#
# Methods:
#   __init__(self, config: KVCacheConfig)
#
#   simulate_workload(self, shared_prefix_pct: float, num_users: int) → List[ConversationTrace]
#     - Generate conversation traces where shared_prefix_pct% of tokens are system prompt
#     - Return list of ConversationTrace objects
#
#   benchmark_with_prefix_cache(self, traces: List[ConversationTrace]) → PrefixCacheResult
#     - Start vLLM with enable_prefix_caching=True
#     - Send traces, measure TTFT for each turn
#     - Critical: measure TTFT on FIRST message vs SUBSEQUENT messages (should differ after warmup)
#     - Return PrefixCacheResult with: per-turn TTFT, cache hit rate
#
#   benchmark_without_prefix_cache(self, traces: List[ConversationTrace]) → PrefixCacheResult
#     - Same as above but with enable_prefix_caching=False
#     - Return baseline for comparison
#
#   compute_speedup(self, with_cache: PrefixCacheResult, without_cache: PrefixCacheResult) → float
#     - Compute TTFT speedup ratio for cached turns
```

**activity3_kv_cache/prefix_cache/multi_turn_simulator.py**
```python
# Class: MultiTurnSimulator
# Purpose: Generate realistic multi-turn conversation traces for benchmarking
#
# Methods:
#   __init__(self, config: WorkloadConfig)
#
#   generate_system_prompt(self, num_tokens: int) → str
#     - Generate a realistic system prompt of approximately num_tokens tokens
#     - Use: SOP-style text about company policies, procedures, guidelines
#     - Source: Use LLM to generate synthetic SOP text, or use public SOP templates
#
#   generate_conversation(self, system_prompt: str, num_turns: int) → ConversationTrace
#     - Generate a realistic multi-turn conversation with the system prompt
#     - Each turn: user question + assistant response
#     - Return ConversationTrace with full message history
#
#   generate_batch(self, num_users: int, shared_prefix_pct: float) → List[ConversationTrace]
#     - Generate num_users conversation traces
#     - shared_prefix_pct% of each trace's prompt tokens are identical (system prompt)
```

---

## Activity 4: Speculative Decoding

### Objective
Train an EAGLE-style draft model. Deploy with vLLM speculative decoding. Benchmark acceptance rate and generation speedup.

### Entry Point
`python activity4_speculative_decoding/main.py --config activity4_speculative_decoding/config.yaml`

### Key Dependency
- EAGLE GitHub: https://github.com/SafeAILab/EAGLE
- vLLM speculative decoding: built-in, no additional dependency

**activity4_speculative_decoding/config.yaml**
```yaml
base_model:
  model_dir: "./checkpoints/int4_awq"

draft_model:
  architecture: "eagle"  # Options: "eagle", "medusa"
  hidden_size: 4096  # Match base model hidden size
  num_draft_tokens: 4  # Number of tokens to draft ahead
  pretrained_draft: null  # Set to HuggingFace path if using pretrained EAGLE draft

training:
  dataset: "ShareGPT"  # or "OpenHermes-2.5" or path to custom dataset
  num_samples: 70000
  max_seq_len: 2048
  batch_size: 4
  learning_rate: 3.0e-5
  num_epochs: 3
  warmup_ratio: 0.05
  save_steps: 500
  output_dir: "./checkpoints/eagle_draft"
  seed: 42

benchmark:
  batch_sizes: [1, 4, 8, 16]
  temperatures: [0.0, 0.6, 1.0]
  num_requests: 200
  prompt_length: 512
  output_length: 256

output:
  results_dir: "./results/activity4_speculative_decoding"
```

**activity4_speculative_decoding/eagle/draft_model.py**
```python
# Class: EAGLEDraftModel (nn.Module)
# Purpose: EAGLE auxiliary decoder that predicts next tokens using base model hidden states
#
# Architecture:
#   - Input: last hidden state from base model (shape: [batch, seq_len, hidden_size])
#             + token embeddings (shape: [batch, seq_len, hidden_size])
#   - Body: Single transformer decoder layer (LlamaDecoderLayer config)
#   - Head: Linear layer projecting to vocab size (weight-tied with base model lm_head)
#
# Key EAGLE insight: Using hidden states (not just token embeddings) as input allows
# the draft model to "see" what the base model is thinking, dramatically improving
# acceptance rates vs embedding-only approaches like Medusa.
#
# Methods:
#   __init__(self, base_model_config: LlamaConfig)
#     - Build decoder layer matching base model's hidden_size, num_heads, etc.
#     - Initialize embedding projection if needed
#
#   forward(self, hidden_states: Tensor, input_ids: Tensor) → Tensor
#     - Concatenate hidden_states with input embeddings
#     - Pass through decoder layer
#     - Project to vocab → return logits [batch, seq_len, vocab_size]
```

**activity4_speculative_decoding/eagle/draft_trainer.py**
```python
# Class: EAGLEDraftTrainer
# Purpose: Train EAGLE draft model using base model hidden states as supervision signal
#
# Training procedure:
#   1. Load base model (frozen — do NOT update base model weights)
#   2. For each batch: run base model forward pass, extract hidden states at each layer
#   3. Feed hidden states + input_ids to EAGLEDraftModel
#   4. Compute cross-entropy loss: draft_model predictions vs ground truth next tokens
#   5. Update only EAGLEDraftModel parameters
#
# Methods:
#   __init__(self, config: TrainingConfig)
#     - Load base model (frozen)
#     - Initialize EAGLEDraftModel
#     - Setup optimizer (AdamW), scheduler (cosine with warmup)
#
#   prepare_dataset(self) → Dataset
#     - Load ShareGPT or specified dataset
#     - Tokenize and chunk to max_seq_len
#     - Return HuggingFace Dataset
#
#   train_epoch(self, dataloader: DataLoader) → float
#     - Run one training epoch
#     - Return average loss
#
#   train(self) → None
#     - Full training loop with checkpointing every save_steps
#     - Log: loss, learning rate, GPU memory every 100 steps
#
#   save_draft_model(self, output_dir: str) → None
#     - Save EAGLEDraftModel weights + config in HuggingFace format
#     - MUST save in a format compatible with vLLM's speculative decoding
```

**activity4_speculative_decoding/benchmark/acceptance_benchmarker.py**
```python
# Class: AcceptanceRateBenchmarker
# Purpose: Measure draft token acceptance rate and generation speedup
#
# Methods:
#   __init__(self, config: BenchmarkConfig)
#
#   benchmark_speculative(self, batch_size: int, temperature: float) → SpeculativeResult
#     - Start vLLM with speculative decoding enabled
#     - vLLM config: speculative_model=draft_model_path, num_speculative_tokens=4
#     - Send num_requests requests, collect: tokens_generated, wall_clock_time, num_draft_rounds
#     - Estimate acceptance rate: tokens_accepted / (num_draft_rounds * num_speculative_tokens)
#     - Return SpeculativeResult with: speedup_vs_standard, acceptance_rate, throughput
#
#   benchmark_standard(self, batch_size: int) → StandardResult
#     - Same benchmark without speculative decoding (control)
#     - Return StandardResult for comparison
#
#   compute_speedup(self, spec: SpeculativeResult, std: StandardResult) → float
#     - Return throughput ratio: spec.throughput / std.throughput
```

---

## Activity 5: Benchmark Dashboard & CLI

### Objective
Package everything as a CLI tool (`loraforge`) and Streamlit dashboard. Publish to PyPI.

### Entry Point (after installation)
```bash
pip install loraforge
loraforge optimize --model meta-llama/Llama-3.1-8B-Instruct --adapter org/adapter-name
loraforge report --results-dir ./loraforge-results
loraforge serve --results-dir ./loraforge-results --port 8000
```

**Key CLI Framework:** Typer (`pip install typer[all]`)
**Dashboard:** Streamlit + Plotly
**Package:** setuptools + pyproject.toml

**activity5_benchmark_dashboard/cli/main.py**
```python
# CLI entry point using Typer
# Commands:
#
# `loraforge optimize` — run full pipeline (Activities 1→4)
#   --model: HuggingFace model ID or local path (required)
#   --adapter: HuggingFace adapter ID or local path (optional)
#   --task: task name for evaluation (default: "general")
#   --accuracy-budget: max acceptable accuracy loss (default: 0.02)
#   --skip-speculative: skip EAGLE training (flag, default: False)
#   --skip-cpt: skip continued pre-training step (flag, default: True)
#   --output-dir: where to save results (default: ./loraforge-results)
#   --gpu: GPU index to use (default: 0)
#
# `loraforge benchmark` — run only benchmarking on existing model
#   --model-dir: path to model directory (required)
#   --output-dir: results output directory
#
# `loraforge serve` — serve optimized model
#   --results-dir: path to loraforge results directory
#   --port: API port (default: 8000)
#
# `loraforge report` — open dashboard for results
#   --results-dir: path to loraforge results directory
#   --port: dashboard port (default: 8501)
```

**activity5_benchmark_dashboard/reporting/dashboard/app.py**
```python
# Streamlit Dashboard Application
# Pages / Sections:
#
# 1. SUMMARY (top of page)
#    - Model name, adapter name, run timestamp
#    - Key improvements (3 metric cards): Memory saved %, TTFT speedup, Accuracy preserved
#    - Best configuration recommendation with reasoning
#
# 2. QUANTIZATION COMPARISON
#    - Table: Format | VRAM | MMLU Loss | TTFT (P50) | Throughput | Recommended
#    - Bar chart: VRAM usage by format (horizontal bars)
#    - Bar chart: Throughput by format
#    - Highlight recommended format in green
#
# 3. KV CACHE IMPACT
#    - Line chart: TTFT vs Shared Prefix % (with/without prefix caching)
#    - Line chart: TTFT vs Context Length (standard vs StreamingLLM)
#    - Metric: "X% TTFT reduction for conversations with shared system prompt"
#
# 4. SPECULATIVE DECODING
#    - Line chart: Acceptance Rate vs Temperature
#    - Bar chart: Throughput Speedup vs Batch Size
#    - Metric: "Xs generation speedup at batch size 1 with Y% acceptance rate"
#
# 5. PARETO CURVES
#    - Scatter plot: Accuracy Loss vs Memory Reduction (each point = one format)
#    - Scatter plot: Accuracy Loss vs Throughput Gain
#    - Pareto frontier highlighted
#
# 6. COST ESTIMATOR
#    - Dropdown: Select GPU type (A10, A100 40GB, A100 80GB, H100)
#    - Slider: Expected requests per day
#    - Table: Cost/1M tokens for each quantization format on selected GPU
#    - Metric: "Estimated monthly cost: $X vs $Y for unoptimized model"
```

---

## Activity 6: Domain-Adaptive Continued Pre-Training

### Objective
LoRA-based CPT on a domain corpus. Measure domain gain vs catastrophic forgetting. Run CPT-adapted model through full LoraForge pipeline.

### Entry Point
`python activity6_domain_cpt/main.py --config activity6_domain_cpt/config.yaml`

**activity6_domain_cpt/config.yaml**
```yaml
model:
  base_model_id: "meta-llama/Llama-3.1-8B-Instruct"

domain:
  name: "cybersecurity"  # Options: cybersecurity, medical, india-sovereign
  corpus_sources:
    - type: "huggingface"
      dataset_id: "sentence-transformers/cybersecurity-corpus"  # placeholder
    - type: "local"
      path: null
  max_tokens: 1_000_000_000  # 1B tokens max for side project
  replay_ratio: 0.1  # Mix 10% general data (WikiText) to prevent forgetting

lora_cpt:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  lora_dropout: 0.05
  bias: "none"

training:
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.05
  max_seq_length: 2048
  save_steps: 500
  logging_steps: 50
  seed: 42
  output_dir: "./checkpoints/cpt_lora_adapter"
  bf16: true  # Use bfloat16 for training

evaluation:
  mmlu_num_samples: 200  # For forgetting measurement (general knowledge)
  domain_test_dataset: null  # Path to domain-specific test set
  domain_test_num_samples: 200

output:
  results_dir: "./results/activity6_domain_cpt"
```

**activity6_domain_cpt/training/cpt_trainer.py**
```python
# Class: LoRACPTTrainer
# Purpose: Run LoRA-based continued pre-training on domain corpus
#
# Training objective: Causal Language Modeling (CLM) — predict next token
# This is DIFFERENT from SFT (which uses instruction-response pairs)
# CPT trains on raw domain text (articles, documentation, papers)
#
# Methods:
#   __init__(self, config: CPTConfig)
#     - Load base model with LoRA config (using PEFT)
#     - Freeze base model weights, only LoRA parameters are trainable
#     - Setup HuggingFace Trainer with CLM data collator
#
#   prepare_data(self) -> Dataset
#     - Download domain corpus
#     - Mix with replay buffer (replay_ratio * |domain_data| samples from WikiText)
#     - Tokenize with stride (for long documents)
#     - Return combined Dataset
#
#   train(self) -> CPTResult
#     - Run HuggingFace Trainer
#     - Log: perplexity on domain text every 500 steps (should decrease)
#     - Save checkpoint every save_steps
#     - Return: CPTResult with training loss curve, final perplexity
#
#   save_adapter(self, output_dir: str) -> None
#     - Save LoRA adapter (NOT the full model — just the adapter weights)
#     - Use: model.save_pretrained(output_dir)  (PEFT handles adapter-only saving)
#
# IMPORTANT: After CPT, the adapter should be used on top of the ORIGINAL base model
# for evaluation. Do NOT merge CPT adapter before running SFT — keep them separate
# and stack: base model + CPT adapter + SFT adapter (PEFT supports multi-adapter stacking)
```

**activity6_domain_cpt/evaluation/forgetting_evaluator.py**
```python
# Class: ForgettingEvaluator
# Purpose: Measure how much base-model general knowledge is retained after CPT
#
# This is critical for the project narrative: showing that LoRA-based CPT
# preserves general knowledge better than full-parameter CPT
#
# Methods:
#   __init__(self, base_model_id: str, cpt_adapter_dir: str)
#
#   evaluate_base_model(self) -> EvalResult
#     - Run MMLU subset on base model (no adapter)
#     - Return: EvalResult(accuracy=X, subject_breakdown={...})
#
#   evaluate_cpt_model(self) -> EvalResult
#     - Run MMLU subset on base model + CPT adapter
#     - Return: EvalResult(accuracy=X, subject_breakdown={...})
#
#   compute_forgetting(self, base: EvalResult, cpt: EvalResult) -> ForgettingReport
#     - Compute: forgetting_rate = (base.accuracy - cpt.accuracy) / base.accuracy
#     - Show per-subject breakdown: which subjects were forgotten, which retained
#     - Return ForgettingReport
#     - Expected: LoRA CPT should show < 2% forgetting (full CPT often shows 5–15%)
```

---

## Testing Requirements

Create tests for every module. Minimum coverage:

```
tests/
├── test_shared/
│   ├── test_logger.py           # Test JSON output format, log levels
│   ├── test_config_parser.py    # Test YAML loading, validation, merging
│   ├── test_gpu_profiler.py     # Test context manager, VRAM tracking
│   └── test_metrics.py          # Test TTFT calculation, percentiles
├── test_activity1/
│   ├── test_vllm_server.py      # Mock vLLM, test server lifecycle
│   └── test_benchmark_client.py # Test request formatting, result parsing
├── test_activity2/
│   └── test_format_selector.py  # Test selection logic with mock results
├── test_activity3/
│   └── test_multi_turn_sim.py   # Test trace generation
├── test_activity4/
│   └── test_eagle_model.py      # Test EAGLE forward pass shapes
├── test_activity5/
│   └── test_cli.py              # Test CLI commands with Typer test runner
└── test_activity6/
    └── test_cpt_trainer.py      # Test training loop with tiny model
```

For GPU-dependent tests, use `@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")`.
For slow tests (model loading), use `@pytest.mark.slow` and exclude from default test run.

---

## Makefile Commands

```makefile
.PHONY: test lint format typecheck install benchmark-activity1

install:
    pip install -e ".[dev]"

test:
    pytest tests/ -v --ignore=tests/slow -x

test-all:
    pytest tests/ -v

lint:
    ruff check .

format:
    black . && isort .

typecheck:
    mypy . --ignore-missing-imports

sft-train:
    python -m activity0_sft.training.qlora_trainer --config activity0_sft/config.yaml

sft-dry-run:
    python -m activity0_sft.training.qlora_trainer --config activity0_sft/config.yaml --dry-run

sft-eval:
    python -m activity0_sft.evaluation.eval --adapter-path ./outputs/cybersec_analyst_lora

baseline:
    python -m activity1_baseline.main

baseline-merge:
    python -m activity1_baseline.merge.merge

baseline-serve:
    python -m activity1_baseline.vllm_server

baseline-bench:
    python -m activity1_baseline.benchmark.client

baseline-eval:
    python -m activity1_baseline.evaluation.perplexity

baseline-report:
    python -m activity1_baseline.reporting.report_generator

benchmark-activity1:
    python -m activity1_baseline.main

benchmark-activity2:
    python activity2_quantization/main.py --config activity2_quantization/config.yaml

benchmark-activity3:
    python activity3_kv_cache/main.py --config activity3_kv_cache/config.yaml

train-eagle:
    python activity4_speculative_decoding/main.py --config activity4_speculative_decoding/config.yaml

benchmark-activity4:
    python activity4_speculative_decoding/main.py --config activity4_speculative_decoding/config.yaml --benchmark-only

dashboard:
    streamlit run activity5_benchmark_dashboard/reporting/dashboard/app.py

cpt-train:
    python activity6_domain_cpt/main.py --config activity6_domain_cpt/config.yaml
```

---

## Environment Variables (.env.example)

```bash
# HuggingFace
HF_TOKEN=hf_xxxxxxxxxxxx              # Required for gated models (Llama-3.1)
HF_HOME=/path/to/hf/cache             # Local HuggingFace cache directory

# NVIDIA ModelOpt
NGC_API_KEY=xxxxxxxxxxxx               # Required for some ModelOpt features

# Logging
LOG_LEVEL=INFO                         # DEBUG, INFO, WARNING, ERROR
LOG_DIR=./logs                         # Directory for log files

# Results
RESULTS_BASE_DIR=./results             # Base directory for all benchmark results
CHECKPOINTS_DIR=./checkpoints          # Base directory for model checkpoints

# GPU
CUDA_VISIBLE_DEVICES=0                 # Which GPU(s) to use
```

---

## Critical Implementation Notes

1. **HuggingFace Token:** Llama-3.1-8B-Instruct is a gated model. Set `HF_TOKEN` in `.env`. Use `huggingface-cli login` before running.

2. **LoRA Merging:** When merging LoRA weights for Activity 1, use `model = PeftModel.from_pretrained(base_model, adapter_path); model = model.merge_and_unload()`. Save the merged model to disk — quantization (Activity 2) needs a merged checkpoint.

3. **vLLM LoRA Mode:** vLLM can serve LoRA adapters WITHOUT merging (dynamic LoRA). Activity 1 benchmarks the merged model for simplicity. Activity 3 can optionally test dynamic LoRA serving.

4. **ModelOpt + HuggingFace:** ModelOpt works with standard HuggingFace `AutoModelForCausalLM` loaded in `torch.float16`. Do NOT use `device_map="auto"` with ModelOpt — use explicit `.to("cuda")`.

5. **EAGLE Compatible Checkpoint:** The EAGLE draft model must be saved in a format vLLM recognizes for speculative decoding. vLLM expects the draft model to have the same tokenizer as the base model. Test with: `loraforge serve --draft-model ./checkpoints/eagle_draft`.

6. **CPT + SFT Stacking:** PEFT supports loading multiple adapters. After CPT training, test: `model.load_adapter(cpt_adapter_path, adapter_name="cpt"); model.load_adapter(sft_adapter_path, adapter_name="sft"); model.set_adapter(["cpt", "sft"])`.

7. **Benchmark Fairness:** Always warm up the server with 10 dummy requests before benchmarking. Record first-request latency separately (cold start) from steady-state latency.

---

## When to Ask for Help

If you encounter:
- CUDA OOM during quantization → Reduce `calibration_batch_size` to 1
- vLLM startup errors → Check `max_model_len` fits in GPU memory
- ModelOpt import errors → Ensure `nvidia-modelopt[torch]` is installed with matching CUDA version
- EAGLE training loss not decreasing → Reduce learning rate to 1e-5, check hidden state extraction

Always log the full error with GPU memory state before asking for help.
