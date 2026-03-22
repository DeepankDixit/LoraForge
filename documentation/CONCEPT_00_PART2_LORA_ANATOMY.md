# Concept 00 — Part 2 of 3: LoRA Adapter Anatomy, Compatibility, and What "Merging" Actually Means

**Activity mapping:** CONCEPT_00 (Parts 1–3) → **Activity 0** (SFT). This is Part 2 of 3.

---

## What Does a LoRA Adapter Actually Look Like on Disk?

When you train a LoRA fine-tuned model and save it, you get a small directory — not a
copy of the full model. This is the key insight. The adapter is just the *delta*, not
the whole model. Here's what you find:

```
my_cybersec_lora_adapter/
├── adapter_config.json          ← The "contract" — what base model this was trained on
├── adapter_model.safetensors    ← The actual trained weights (A and B matrices)
├── tokenizer.json               ← Copy of the base model's tokenizer
├── tokenizer_config.json        ← Tokenizer configuration
└── special_tokens_map.json      ← Special token definitions
```

That's it. For a Llama-3.1-8B model fine-tuned with LoRA rank=16 on 7 target modules,
the `adapter_model.safetensors` file is typically **80–200 MB** versus the full model's **16 GB**.
This is why LoRA is so practical — you can share it, version it, and swap it quickly.

---

## The adapter_config.json: The Compatibility Contract

This file is what determines whether an adapter works with a given base model.
Here is exactly what it contains after a typical SFT run:

```json
{
  "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 16,
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ],
  "task_type": "CAUSAL_LM"
}
```

**Every field matters:**

| Field | What it controls | Compatibility impact |
|---|---|---|
| `base_model_name_or_path` | What model this was built for | CRITICAL — must match |
| `r` (rank) | Size of the LoRA matrices | No impact on loading |
| `lora_alpha` | Scaling factor (alpha/r = effective LR scale) | No impact on loading |
| `target_modules` | Which weight matrices have adapters | CRITICAL — these layer names must exist in base model |
| `task_type` | CAUSAL_LM for decoder models | No impact on loading |

---

## The Math: Why r and alpha Matter

This is the conceptual core of LoRA. Do not skip this.

A transformer attention layer has a weight matrix W with shape [d_model × d_model].
For Llama-3.1-8B: d_model = 4096, so W is 4096 × 4096 = 16.7M parameters. Per layer.
With 32 layers and 7 target matrices, full fine-tuning updates:
```
32 layers × 7 matrices × 4096 × 4096 = 3.8 billion parameters
```
That's half the entire model, just for the matrices we care about.

LoRA instead trains two small matrices:
```
A: shape [r × d_model]  = [16 × 4096]  = 65,536 parameters
B: shape [d_model × r]  = [4096 × 16]  = 65,536 parameters

A × B gives a [d_model × d_model] update, but parameterized by only 131K values.
That's a 127× compression per matrix.
```

At inference time, the effective weight is:
```
W_effective = W_original + (lora_alpha / r) × (B × A)
            = W_original + (32 / 16)   × (B × A)
            = W_original + 2.0         × (B × A)
```

The ratio `lora_alpha / r` is the scaling factor. Setting alpha = 2 × r (e.g., r=16, alpha=32)
is a common default that means "apply the full adapter update without additional scaling."

**What "rank" controls:** Higher rank = more capacity to learn = more parameters.
Typical values and when to use them:
- `r=4` or `r=8` — tiny adapters, minimal forgetting, for simple behavioral changes
- `r=16` — good default for most SFT tasks (what Deepank used at Optum)
- `r=32` or `r=64` — larger capacity, better for complex domain knowledge
- `r=128` — large adapters, approach full fine-tune quality, heavier compute

---

## Compatibility: When Can You Use Someone Else's Adapter?

This is your critical question. Here are the rules, in order of importance:

### Rule 1: Base model must be the same architecture AND version

```
adapter trained on: meta-llama/Llama-3.1-8B-Instruct
base model you load: meta-llama/Llama-3.1-8B-Instruct   ✓ Works perfectly
base model you load: meta-llama/Llama-3.1-70B-Instruct  ✗ Wrong size — different d_model
base model you load: meta-llama/Llama-3.2-8B-Instruct   ✗ Different version — architecture changed
base model you load: meta-llama/Llama-3-8B-Instruct     ✗ Llama 3 ≠ Llama 3.1
base model you load: mistralai/Mistral-7B-Instruct-v0.3  ✗ Completely different model
```

PEFT reads `base_model_name_or_path` from `adapter_config.json` and will raise a warning
(or error) if it doesn't match. You can override this with `is_trainable=False` and
`ignore_mismatched_sizes=True`, but the results will be garbage — the weight matrices
will be injected into the wrong layers.

### Rule 2: Target modules must exist in the base model

If an adapter was trained targeting `q_proj, k_proj, v_proj` but you load it on a model
that uses `query_key_value` (like some Falcon models), PEFT will error out because those
layer names don't exist.

For all Llama-3.x models, the target module names are standardized:
`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
So any Llama-3.x LoRA adapter targeting these names will work on any Llama-3.x model
*of the same size*.

### Rule 3: Quantization base vs full precision base

If an adapter was trained on a QLoRA setup (4-bit quantized base + LoRA), you can still
load it on a full-precision (FP16) base model. The A and B matrices themselves are always
stored in FP32 or BF16 regardless. This is fine and actually recommended for inference
(full precision base + LoRA adapter = better quality than QLoRA quantized base at inference time).

---

## What "Merging" Means: The Pre-computation Trick

At inference time, you have two options:

### Option A: Keep them separate (dynamic LoRA)
```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(model, "path/to/adapter")
# At every forward pass: compute W_original + (alpha/r) × (B × A) on the fly
```
This adds a small compute overhead per forward pass but lets you:
- Swap adapters at runtime (serve multiple fine-tunes with one base model)
- Hot-reload adapters without restarting the server

vLLM supports this with `--enable-lora` flag.

### Option B: Merge into a single model (what Activity 1 does)
```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(model, "path/to/adapter")
model = model.merge_and_unload()  # W_merged = W_original + (alpha/r) × (B × A)
model.save_pretrained("path/to/merged_model")
# Now it's just a regular model — no PEFT overhead at inference time
```

After merging, the resulting file IS the LoRA-adapted model baked into a single set of weights.
The LoRA structure is gone — it's just a regular Llama-3.1-8B with modified weight values.
This is what we serve in Activity 1, and what we quantize in Activity 2.

**Why merge for Activity 1?** Simplicity. ModelOpt (Activity 2) works on standard
HuggingFace models, not PEFT models. Merging first makes quantization straightforward.

---

## The Key Gap: No Public Llama-3.1-8B-Instruct Cybersecurity LoRA Adapter Exists

After searching HuggingFace, here is the honest situation:

There is **one** notable cybersecurity fine-tuned model based on Llama-3.1-8B:
`fdtn-ai/Foundation-Sec-8B` — trained by Foundation AI.

However it is a **full fine-tune** (all weights updated, no LoRA structure).
You cannot use it as a LoRA adapter. You'd have to use it as-is as a merged model.

For all other cybersecurity fine-tunes on HuggingFace:
- Most are trained on Llama-2, Mistral, or older model versions
- None are specific LoRA adapters for Llama-3.1-8B-Instruct + cybersecurity

**This is exactly why Activity 0 (our own SFT) is the right call.**

Since no public LoRA adapter exists for our specific setup (Llama-3.1-8B-Instruct +
cybersecurity), we must create one ourselves. This is not a limitation — it's actually
the most authentic and interesting part of the story:

*"I identified the gap, curated the dataset, ran the SFT myself, and then built the
entire inference optimization pipeline on top of my own model."*

That is a significantly more compelling story than "I downloaded someone else's adapter."

---

## Summary: The Flow We Will Follow

```
Activity 0: SFT (NEW — we create this)
  Input: meta-llama/Llama-3.1-8B-Instruct (base model, 16GB)
  Data:  Trendyol cybersecurity dataset (53K examples) +
         CVE chat dataset (sample 30K from 300K)
  Train: QLoRA — 4-bit quantized base + LoRA rank=16
  Output: cybersec_analyst_lora/
          ├── adapter_config.json  ← "trained on Llama-3.1-8B-Instruct"
          └── adapter_model.safetensors  ← our delta weights (~150MB)

Activity 1: Baseline
  Load: Llama-3.1-8B-Instruct (FP16) + our adapter
  Merge: model.merge_and_unload() → single 16GB model
  Serve: vLLM with merged FP16 model
  Measure: TTFT, throughput, VRAM, perplexity → baseline_metrics.json

Activity 2: Quantization
  Input: merged FP16 model from Activity 1
  Apply: FP8 / INT4 AWQ / INT8-SQ via ModelOpt
  Output: quantized checkpoint (4-8GB instead of 16GB)

Activity 3: KV Cache → Activity 4: Speculative Decoding → Activity 5: Dashboard

Activity 6: CPT (optional, upstream of Activity 0)
  Would come BEFORE Activity 0 if we wanted to improve the base model first
  Using: cybersecurity raw text corpus (CVE descriptions, MITRE ATT&CK docs)
```
