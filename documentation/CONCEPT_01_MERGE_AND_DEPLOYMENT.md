# Concept 01: The Merge Operation, vLLM, PagedAttention, and Measuring Baseline Performance

## From Adapter to Deployed Model — Everything That Happens Between Activity 0 and Activity 1

**Activity mapping:** CONCEPT_01 → **Activity 1** (baseline deployment). Read this before the merge script, vLLM server setup, and benchmark client.

---

## Part 1: What "Merging" Actually Means

At the end of Activity 0, you have two separate things on disk:

```
meta-llama/Llama-3.1-8B-Instruct (base model, ~16GB in FP16, on HuggingFace)
cybersec_analyst_lora/
    adapter_config.json         (~2KB — describes the LoRA config)
    adapter_model.safetensors   (~150MB — just the A and B matrices)
```

The adapter is TINY compared to the base model. It contains only the learned deltas for the layers you targeted (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj across 32 transformer blocks).

### The merge equation

Recall from Concept 00 — Part 2: at inference time, LoRA adds the adapter delta to the base weights:

```
output = W × input + (alpha/r) × (B × A) × input
       = [W + (alpha/r) × (B × A)] × input
```

The term `(alpha/r) × (B × A)` is computed at every forward pass when running with a separate adapter. Merging bakes this into W permanently:

```
Before merge: W_base (frozen), A, B (adapter)
After merge:  W_merged = W_base + (alpha/r) × (B × A)
```

This is a one-time matrix addition per targeted layer. After merging, A and B matrices are discarded. The result is a single, self-contained model that behaves exactly like the adapter-equipped model — but with zero runtime overhead from LoRA computation.

### Why merge for inference?

Two reasons:

**1. Serving simplicity:** vLLM, TGI, and other inference engines work best with standard HuggingFace model checkpoints. Loading a base model + adapter dynamically adds complexity and has limited support for some optimizations.

**2. Inference speed:** The adapter delta computation (`(alpha/r) × B × A × input`) adds FLOPs at every forward pass, every layer, every token. After merging, this overhead is zero — the fused weight matrix is used directly.

The downside: once merged, you lose the ability to swap adapters dynamically (vLLM's `--enable-lora` mode keeps them separate for multi-adapter serving). For our purposes — one adapter, one deployment — merging is the right call.

### The merge code (what Activity 1 does)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Step 1: Load base model in FP16
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu"           # merge on CPU — no GPU memory needed
)

# Step 2: Load adapter on top of base
peft_model = PeftModel.from_pretrained(base_model, "./outputs/cybersec_analyst_lora")

# Step 3: Merge and unload — W_merged = W_base + (alpha/r) × B × A
merged_model = peft_model.merge_and_unload()

# Step 4: Save as a standard HuggingFace checkpoint
merged_model.save_pretrained("./outputs/cybersec_analyst_merged")
tokenizer.save_pretrained("./outputs/cybersec_analyst_merged")
```

After this, `./outputs/cybersec_analyst_merged/` contains a complete, standalone model — indistinguishable from any other Llama checkpoint. vLLM loads it with no knowledge that LoRA was involved.

---

## Part 2: Why FP16 for the Baseline

Activity 1 deploys the merged model in **FP16** (16-bit floating point). This is the unoptimized baseline — we want to measure what performance looks like before any inference-time optimization.

### The FP16 memory math

```
Llama-3.1-8B merged model in FP16:
  8.03 billion parameters × 2 bytes = 16.06 GB VRAM

A10G GPU has 24 GB VRAM.
Remaining for KV cache, activations: ~7.9 GB
```

This is why we need a GPU with ≥24 GB VRAM for the baseline. The model barely fits, with enough room left for serving a few concurrent requests.

Compare this to the quantized variants (Activity 2):
```
FP8:  ~8 GB (2× reduction)
INT8: ~8 GB (2× reduction)
INT4: ~4 GB (4× reduction)
```

The baseline (FP16) gives us the accuracy ceiling and the performance floor. Activities 2–5 trade memory and sometimes accuracy for speed.

---

## Part 3: How vLLM Works — The Engine We're Deploying Into

vLLM is the de facto standard for high-throughput LLM inference. Before we benchmark against it, you need to understand what it does differently from naive HuggingFace `model.generate()`.

### The naive approach (why HuggingFace generation is slow in production)

When you call `model.generate()` in HuggingFace for multiple concurrent requests:
1. Each request gets its own full GPU memory allocation
2. The KV cache (keys and values for each attention head, each layer) is pre-allocated for the **maximum sequence length** — even if the actual sequence is much shorter
3. Memory is allocated in contiguous blocks — if request A finishes and frees memory, that memory can't be given to request B (fragmentation)
4. Batching is static — you batch requests at the start, wait for the longest one to finish, then batch again

This wastes 60–80% of GPU memory on padding and pre-allocation, and kills throughput for concurrent requests.

### PagedAttention — the core vLLM innovation

PagedAttention (Kwon et al., 2023) solves the KV cache memory problem. The insight comes from virtual memory in operating systems.

**The problem with contiguous KV cache allocation:**
```
Request A: 512 tokens → needs 512 × (32 layers × 2 × 128 heads × head_dim) = fixed block
Request B: 1024 tokens → needs 2× as much

If A finishes, its block is 512 tokens of free memory.
B can't use that fragmented space — it needs 1024 contiguous tokens.
Result: memory fragmentation, cannot fully utilize GPU.
```

**PagedAttention's solution:**

Divide KV cache into fixed-size pages (e.g., 16 tokens each). Maintain a page table — just like a CPU's virtual memory system — that maps logical token positions to physical pages.

```
Logical view (request A, 512 tokens):
  [Block 0: tokens 0-15] → [Block 1: tokens 16-31] → ... → [Block 31: tokens 496-511]

Physical memory (interleaved with other requests):
  Physical page 4  ← holds tokens 0-15 of request A
  Physical page 11 ← holds tokens 0-15 of request B
  Physical page 4  ← page 4 is shared by both? NO → each page holds one request's tokens

But freed pages from finished requests CAN be immediately reused for new requests.
No fragmentation. Near-100% memory utilization.
```

The key operations:
- **Append:** When a new token is generated, append its KV to the current page. When the page is full, allocate a new page (any free physical page works — no contiguity needed)
- **Attention computation:** The attention kernel is modified to handle non-contiguous KV storage via the page table lookup
- **Eviction:** When memory is tight, pages can be swapped to CPU RAM (like virtual memory paging) — vLLM supports this but it's slow

### Continuous batching

Traditional batching: batch 8 requests → run until all 8 finish → batch the next 8.

vLLM's continuous batching: as soon as one request in the current batch finishes, immediately insert a waiting request into the batch. The GPU is never idle between requests.

```
Time → →  →  →  →  →  →  →
Slot 1: [Request A: 45 tokens] [Request E: starts] ...
Slot 2: [Request B: 23 tokens][Request D starts here] ...
Slot 3: [Request C: 67 tokens ......................] [Request F starts] ...
```

This is why vLLM's throughput (tokens/second across concurrent requests) is much higher than naïve serving — GPU utilization stays near 100%.

---

## Part 4: The Two Performance Metrics — TTFT and Throughput

These two metrics measure completely different things. Conflating them is the most common mistake in inference engineering.

### TTFT — Time To First Token

**Definition:** The time between sending the request and receiving the FIRST generated token.

**What drives TTFT:**
1. **Prefill computation:** Before generating any output, the model must process the entire input prompt through all 32 transformer layers. For a 512-token prompt: 512 × 32 × (attention + MLP) operations. This is pure compute.
2. **Queue wait:** If other requests are being processed, your request waits in a queue.
3. **Network RTT:** If the server is remote, add round-trip latency.

**Why TTFT matters:**
- User-perceived responsiveness. A chatbot that takes 3 seconds to show ANY output feels broken, even if it then streams fast.
- For real-time applications (voice, code completion), TTFT is THE metric.

**Baseline TTFT for Llama-3.1-8B FP16 on A10G:**
- Short prompt (128 tokens): ~100–200ms
- Medium prompt (512 tokens): ~300–500ms
- Long prompt (2048 tokens): ~1–2 seconds

**What reduces TTFT:**
- KV cache prefix sharing (Activity 3): if a system prompt is shared across requests, its KV can be pre-computed once
- Speculative decoding (Activity 4): the draft model prefills faster
- Quantization (Activity 2): fewer bytes to move through memory → faster prefill

### Throughput — Tokens Per Second

**Definition:** Total output tokens generated across all requests per second (aggregate, not per-request).

**What drives throughput:**
1. **Memory bandwidth:** Once the KV cache is built (prefill done), decoding is memory-bound — the bottleneck is loading the 16GB model weights from HBM to compute cores for each token step
2. **Batch size:** Larger batches amortize the weight-loading cost across more tokens simultaneously
3. **Model size:** Larger models → more bytes to move → lower throughput per token

**Why throughput matters:**
- API serving cost. You pay for GPU-hours. Higher throughput = more requests served per dollar.
- Batch processing (summarizing 10,000 documents) — TTFT is irrelevant, throughput is everything.

**Baseline throughput for Llama-3.1-8B FP16 on A10G:**
- Single request: ~40–60 tokens/second
- Batch of 16 requests: ~400–600 tokens/second total (25–40 per request, but 10× more done simultaneously)

**The TTFT vs Throughput tradeoff:**
- Larger batches → higher throughput, higher TTFT (requests wait longer to start)
- Smaller batches → lower TTFT, lower throughput
- vLLM's continuous batching finds a good operating point automatically

### The third metric: Memory Usage

Peak VRAM usage constrains everything else:
- Higher VRAM usage → fewer KV cache pages → fewer concurrent requests → lower throughput
- The baseline (FP16) leaves only ~8GB for KV cache on a 24GB A10G
- Activity 2 (quantization to FP8/INT4) frees ~8–12GB more → doubles or triples concurrent request capacity

---

## Part 5: The Activity 1 Benchmark Architecture

Here is exactly what Activity 1 builds and measures:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ACTIVITY 1 ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  STEP 1: MERGE                                                      │
│  merge.py: base_model + LoRA adapter → merged_model (FP16)         │
│  Output: ./outputs/cybersec_analyst_merged/ (~16GB)                 │
│                                                                     │
│  STEP 2: SERVE                                                      │
│  vllm_server.py launches:                                           │
│    python -m vllm.entrypoints.openai.api_server                    │
│      --model ./outputs/cybersec_analyst_merged                      │
│      --dtype float16                                                │
│      --max-model-len 4096                                           │
│      --port 8000                                                    │
│  Exposes: OpenAI-compatible API at localhost:8000                   │
│                                                                     │
│  STEP 3: BENCHMARK                                                  │
│  benchmark/client.py sends concurrent requests:                     │
│    - Concurrency levels: [1, 4, 8, 16, 32]                        │
│    - Prompt lengths: [128, 256, 512, 1024, 2048] tokens            │
│    - Output lengths: [64, 128, 256] tokens                         │
│    - Measures: TTFT per request, throughput per concurrency level   │
│                                                                     │
│  STEP 4: EVALUATE ACCURACY                                          │
│  evaluation/perplexity.py:                                          │
│    - Computes perplexity on held-out cybersec test set              │
│    - Runs 6 qualitative prompts from eval.py                        │
│    - Baseline: base model (no adapter) vs merged model              │
│                                                                     │
│  STEP 5: REPORT                                                     │
│  reporting/report_generator.py:                                     │
│    - Reads benchmark JSON → generates markdown + HTML report        │
│    - Baseline numbers stored in baseline_results.json               │
│    - All later activities compare against these numbers             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why OpenAI-compatible API?

vLLM exposes an OpenAI-compatible REST API (chat completions endpoint). This means:
1. The benchmark client uses the same code as any OpenAI client — trivially portable
2. You can test with curl, Python's `openai` library, or any LLM framework
3. Future activities (2–5) use the same API endpoint — only the server config changes

```python
# The benchmark client is just an OpenAI client pointed at localhost
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="./outputs/cybersec_analyst_merged",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=256,
    temperature=0.1,
)
```

### Measuring TTFT with streaming

To accurately measure TTFT, you need streaming — otherwise the client waits for the full response before getting any timing:

```python
import time

start = time.perf_counter()
stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    max_tokens=256,
    stream=True,    # ← enables streaming
)

first_token_time = None
full_response = ""

for chunk in stream:
    if first_token_time is None and chunk.choices[0].delta.content:
        first_token_time = time.perf_counter()   # ← TTFT captured here
        ttft = first_token_time - start
    full_response += chunk.choices[0].delta.content or ""

end = time.perf_counter()
total_tokens = len(full_response.split())
throughput = total_tokens / (end - first_token_time)   # tokens/sec, decode phase only
```

---

## Part 6: What the Baseline Numbers Tell You

After Activity 1, you will have a JSON file that looks like this:

```json
{
  "model": "cybersec_analyst_merged_fp16",
  "gpu": "A10G 24GB",
  "vram_used_gb": 16.8,
  "vram_available_for_kv_cache_gb": 7.2,
  "results": {
    "concurrency_1": {
      "ttft_p50_ms": 180,
      "ttft_p95_ms": 340,
      "throughput_tokens_per_sec": 52
    },
    "concurrency_8": {
      "ttft_p50_ms": 620,
      "ttft_p95_ms": 1100,
      "throughput_tokens_per_sec": 390
    },
    "concurrency_16": {
      "ttft_p50_ms": 1400,
      "ttft_p95_ms": 2600,
      "throughput_tokens_per_sec": 520
    }
  },
  "accuracy": {
    "perplexity_base_model": 18.4,
    "perplexity_merged_model": 9.7,
    "perplexity_reduction_pct": 47.3
  }
}
```

This file becomes the **baseline** that every subsequent activity compares against. Activity 2 will produce the same JSON format with `model: "cybersec_analyst_fp8"` and you'll see throughput go up and VRAM go down. Activity 4 (speculative decoding) will show TTFT drop dramatically.

The story your GitHub README tells: *"I took an FP16 baseline and systematically applied each optimization, measuring the real-world impact at each step."* The baseline_results.json is proof that you started from real numbers, not hand-waved estimates.

---

## Part 7: The vLLM Startup Sequence — What Happens When You Launch the Server

Understanding this prevents confusion when reading server logs:

```
1. MODEL LOADING (~30-60 seconds)
   vLLM loads the model weights from disk into GPU HBM
   FP16: moves 16GB from NVMe → RAM → GPU PCIe → HBM
   Progress bar shows weight loading %

2. GRAPH COMPILATION (30-90 seconds, first-time only)
   vLLM uses CUDA graphs for fast inference
   It pre-compiles computation graphs for different batch sizes and sequence lengths
   Cached after first run — subsequent starts skip this

3. KV CACHE INITIALIZATION
   vLLM computes how many PagedAttention pages it can fit in remaining VRAM
   Formula: (total_vram - model_vram - overhead) / page_size
   With FP16 on A10G: (24GB - 16.8GB - 0.5GB) / (block_size × 2 bytes × n_heads × n_layers)
   Logs: "GPU blocks: 412, CPU blocks: 0"
   Each GPU block = 16 tokens of KV cache for all layers

4. READY
   "Application startup complete."
   API listening at 0.0.0.0:8000

5. REQUEST PROCESSING (continuous)
   Each request: tokenize → scheduler picks batch → prefill → decode → detokenize → respond
```

When you see "GPU blocks: 412" in the logs, you know exactly how many concurrent 16-token pages are available for KV cache. Activity 3 (prefix caching) increases the effective number of blocks by reusing pages for shared prefixes. Activity 2 (quantization) reduces model VRAM, freeing more blocks.

---

## Summary: The Activity 1 Flow

```
WHAT WE START WITH:
  cybersec_analyst_lora/ adapter from Activity 0 (~150MB)
  meta-llama/Llama-3.1-8B-Instruct base model on HuggingFace

WHAT WE DO:
  1. Merge:    peft.merge_and_unload() → save merged FP16 model (~16GB)
  2. Serve:    Launch vLLM with OpenAI-compatible API
  3. Benchmark: Sweep concurrency levels and prompt lengths → TTFT + throughput matrix
  4. Evaluate: Perplexity on held-out cybersec test set + qualitative responses
  5. Report:   Generate baseline_results.json + human-readable report

WHAT WE MEASURE:
  TTFT:       How long until the first token? (user experience metric)
  Throughput: How many tokens/sec across all concurrent requests? (cost metric)
  VRAM:       How much GPU memory does the model occupy? (capacity metric)
  Perplexity: Did the merge preserve the fine-tuning quality? (accuracy metric)

WHAT WE GET:
  outputs/cybersec_analyst_merged/     ← the merged FP16 model
  results/baseline_results.json        ← the numbers every later activity beats
  results/baseline_report.md           ← human-readable summary

THE STORY:
  "FP16 baseline: 16.8GB VRAM, 52 tok/s at concurrency 1, 180ms P50 TTFT.
   This is the starting point. Activities 2-5 will systematically improve it."
```

---

## The Next Concept

Concept 05 will cover quantization in depth: what FP8, INT8, and INT4 mean at the hardware level, how vLLM's built-in quantization (bitsandbytes, GPTQ, AWQ) works, and the accuracy-memory-speed tradeoff you navigate in Activity 2.
