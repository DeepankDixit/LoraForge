# Concept 00 — Part 3 of 3: The SFT Training Objective, Catastrophic Forgetting, and What QLoRA Actually Does

**Activity mapping:** CONCEPT_00 (Parts 1–3) → **Activity 0** (SFT). This is Part 3 of 3 — the direct theory behind the training code.

---

## Part 1: The Causal Language Modeling (CLM) Objective

This is the single most important concept to understand before training any language model.

All GPT-style models (including Llama) are trained using **next-token prediction**. Given a sequence of tokens, predict the next one. That's it. The entire intelligence of the model emerges from doing this billions of times.

### What "next-token prediction" actually means mathematically

Given a sequence of tokens `[t₁, t₂, t₃, ..., tₙ]`, the model learns to maximize:

```
P(t₁, t₂, t₃, ..., tₙ) = P(t₁) × P(t₂|t₁) × P(t₃|t₁,t₂) × ... × P(tₙ|t₁,...,tₙ₋₁)
```

At training time, this becomes minimizing cross-entropy loss over each token:

```python
# Pseudocode for what the loss function computes
loss = 0
for position in range(1, len(tokens)):
    # Given everything before position, predict token at position
    logits = model(tokens[:position])
    loss += cross_entropy(logits, tokens[position])
loss = loss / len(tokens)
```

The model sees ALL tokens in the sequence simultaneously (that's what the attention mask is for), and the loss is computed over EVERY token position. This is the CLM (Causal Language Modeling) objective — "causal" because each token can only attend to previous tokens, not future ones.

---

## Part 2: CPT vs SFT — The Single Critical Difference

Both CPT and SFT use the CLM objective underneath. The difference is **which tokens you compute loss on**.

### CPT (Continued Pre-Training): Loss on every token

```
Training sample: "The TLS handshake begins when the client sends a ClientHello message..."

Tokens:  [The] [TLS] [hand] [shake] [begins] [when] [the] [client] [sends] [a] [Client] [Hello]
Loss:     ✓     ✓     ✓       ✓       ✓        ✓      ✓     ✓        ✓      ✓    ✓         ✓

Every single token contributes to the gradient update.
The model is learning: "given this domain text, what comes next?"
```

### SFT (Supervised Fine-Tuning): Loss only on assistant tokens

```
Training sample (ChatML format):
<|im_start|>system
You are a cybersecurity analyst.<|im_end|>
<|im_start|>user
What is a CVE?<|im_end|>
<|im_start|>assistant
A CVE (Common Vulnerabilities and Exposures) is a standardized identifier...<|im_end|>

Tokens:  [<|im_start|>] [system] [You] [are] ... [?] [<|im_end|>] [<|im_start|>] [assistant] [A] [CVE] [is]...
Loss:         ✗              ✗      ✗     ✗   ...  ✗      ✗              ✗              ✗         ✓    ✓    ✓

Only the assistant's RESPONSE tokens contribute to the loss.
The model is learning: "given this instruction, produce THIS specific response."
```

In HuggingFace's `trl` library, this is controlled by the `DataCollatorForCompletionOnlyLM` class which masks out the prompt tokens from the loss computation. In the SFT Trainer, you set this up via the `response_template` argument.

**This single change (masked loss) is what transforms a "text predictor" into an "instruction follower."**

---

## Part 3: The Data Format — ChatML

Llama-3.1-8B-Instruct was instruction-tuned using the **ChatML format**. When we fine-tune, we must use the same format, otherwise the model's internal "where does the user message end and my response begin?" understanding breaks.

Here is exactly what ChatML looks like for a cybersecurity QA example:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert cybersecurity analyst with deep knowledge of CVEs, MITRE ATT&CK, and network security.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the difference between a vulnerability and an exploit?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

A vulnerability is a weakness in software or hardware that could be exploited. An exploit is the actual mechanism or code that takes advantage of that vulnerability to cause unintended behavior...<|eot_id|>
```

The special tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`) are defined in the tokenizer's `special_tokens_map.json`. Llama-3.x uses different special tokens than Llama-2 (`<s>`, `[INST]`) — this is one reason cross-version adapters don't work.

In Python, the tokenizer applies this format automatically via `apply_chat_template()`:

```python
messages = [
    {"role": "system", "content": "You are a cybersecurity analyst."},
    {"role": "user", "content": "What is a CVE?"},
    {"role": "assistant", "content": "A CVE (Common Vulnerabilities and Exposures)..."}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
# Returns the full ChatML string above
```

---

## Part 4: Catastrophic Forgetting — The Central Problem of Fine-Tuning

Here is the core problem: **when you fine-tune a model on a narrow dataset, it tends to forget everything else it knew.**

### Why this happens

The base model (Llama-3.1-8B-Instruct) was trained on trillions of tokens of diverse text. Its weights encode an enormous amount of general knowledge — math, coding, language understanding, common sense.

When you fine-tune on, say, 53K cybersecurity QA pairs and update all 8B parameters:
- The gradient updates push ALL weights toward producing cybersecurity answers
- Weights that encoded "how to write a poem" or "basic arithmetic" get overwritten
- After fine-tuning, the model might fail at tasks it previously handled well

This is **catastrophic forgetting** (also called catastrophic interference). It was first studied in the 1990s for neural networks, but it's acutely relevant for LLMs.

### Practical evidence

You can observe catastrophic forgetting by asking a fully fine-tuned cybersecurity model to:
1. Write a simple Python function (unrelated to security) — performance degrades
2. Answer a basic geography question — performance degrades
3. Follow multi-turn conversation naturally — often becomes stilted

### Why LoRA largely solves this

LoRA doesn't update the original weights W at all. It adds a small **parallel branch** (B × A) that produces a delta:

```
Original forward pass (base model):
  output = W × input                    # W is frozen, never changes

LoRA forward pass:
  output = W × input + (alpha/r) × (B × A) × input
         = (base knowledge) + (fine-tuning delta)
```

The A and B matrices start near zero (A is initialized with random Gaussian, B is initialized to zero, so B × A = 0 at start). The model begins training from the base model's behavior and learns small adjustments.

**The base knowledge is preserved in W. Only the small delta (B × A) is trained.** This is why LoRA fine-tuned models tend to retain general capabilities much better than full fine-tuning.

The tradeoff: LoRA cannot learn as dramatic a behavioral change as full fine-tuning. But for domain adaptation and instruction style, rank=16 is more than sufficient.

---

## Part 5: QLoRA — What It Actually Adds on Top of LoRA

LoRA solves the forgetting problem. QLoRA solves the memory problem.

### The memory math for fine-tuning

To fine-tune Llama-3.1-8B with standard LoRA on a single GPU:

```
Base model weights (FP16):  8B params × 2 bytes = 16 GB
Gradients (FP32):           8B params × 4 bytes = 32 GB  (but LoRA only trains A,B)
LoRA A,B matrices:          ~30M params × 4 bytes = 120 MB
Optimizer states (AdamW):   2 × gradient size for momentum + variance
Activations (forward pass): depends on batch size and sequence length

Total for full LoRA training: ~22-25 GB VRAM minimum
```

A single A10G (24 GB) would work, but barely. An A100 (40 GB) is more comfortable.

### What QLoRA does

QLoRA was introduced by Dettmers et al. (2023) in the paper "QLoRA: Efficient Finetuning of Quantized LLMs". The core idea:

**Load the base model in 4-bit NF4 (NormalFloat4) quantization, then train LoRA adapters on top.**

```
QLoRA memory breakdown:
Base model weights (NF4, 4-bit):  8B params × 0.5 bytes = 4 GB  ← 4× reduction
LoRA A,B matrices (BF16):         ~30M params × 2 bytes  = 60 MB
Optimizer states (AdamW on A,B):  ~360 MB
Activations:                       ~2-4 GB depending on batch

Total for QLoRA: ~7-9 GB VRAM
```

This means you can fine-tune Llama-3.1-8B on a **single consumer GPU with 16 GB VRAM** (RTX 3090, RTX 4090). On a Lambda Cloud A10G (24 GB), you have significant headroom for longer sequences.

### The technical details that matter

**NF4 (NormalFloat4):** A 4-bit quantization format specifically designed for normally-distributed neural network weights. Unlike INT4 which just rounds to 16 discrete values, NF4 places quantization bins at quantiles of a standard normal distribution — so it wastes fewer bits on the tails.

**Double quantization:** QLoRA also quantizes the quantization constants themselves (the scale factors needed to dequantize). This saves ~0.5 GB more.

**Paged optimizers:** NVidia's `bitsandbytes` library uses CPU memory for optimizer states that overflow GPU VRAM, with automatic paging. This prevents OOM crashes during training.

**During the forward pass:** The 4-bit weights are dequantized to BF16 on-the-fly before computation. The computation itself happens in BF16. Only the storage is 4-bit. This means QLoRA is slightly slower than FP16 LoRA (dequantization overhead) but dramatically more memory-efficient.

### The key insight: A and B matrices are NOT quantized

The LoRA A and B matrices themselves are stored and trained in BF16. Only the frozen base model weights W are quantized. This is why QLoRA adapters are compatible with full-precision base models at inference time — the adapter was never quantized.

---

## Part 6: What Gradient Checkpointing Does

You'll see `gradient_checkpointing=True` in the training config. This trades compute for memory.

Normally during backpropagation:
- Forward pass stores ALL intermediate activations (needed for gradient computation)
- For a 32-layer transformer with batch_size=4, seq_len=2048, this is ~8 GB of activations

With gradient checkpointing:
- Forward pass stores only a few "checkpoints" (typically one per transformer block)
- During backward pass, re-compute the activations from the last checkpoint
- This adds ~30% compute time but reduces activation memory by ~70%

In code: `model.gradient_checkpointing_enable()` before calling `trainer.train()`.

---

## Part 7: The Training Loop — What Happens Every Step

Here is the exact sequence of events during one training step in our QLoRA setup:

```
1. BATCH LOADING
   DataLoader pulls a batch of 4 tokenized sequences (batch_size=4)
   Each sequence is max_seq_length=2048 tokens

2. FORWARD PASS (what model.forward() does)
   For each transformer layer (32 layers in Llama-3.1-8B):
   a. Dequantize W_q from NF4 → BF16 (the query projection weights)
   b. Compute: attention_scores = (W_q × input) + (alpha/r × B_q × A_q × input)
                                   ↑ base model      ↑ LoRA delta (learned)
   c. Same for k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
   d. Forward through the full MLP block

3. LOSS COMPUTATION
   output_logits has shape [batch=4, seq_len=2048, vocab_size=128256]
   Apply the attention mask: zero out loss on system/user tokens
   CrossEntropyLoss only on the assistant response tokens

4. BACKWARD PASS
   Compute gradients: dLoss/dA, dLoss/dB for each LoRA matrix
   The frozen W weights have no gradients computed (requires_grad=False)

5. OPTIMIZER STEP (AdamW, paged version)
   Update A and B matrices using Adam momentum + variance estimates
   Typical learning rate: 2e-4

6. LOGGING
   Every 10 steps: log training loss, learning rate, GPU memory
   Every 50 steps: save a checkpoint (adapter_config + adapter_model)
```

After ~2-3 epochs over the 53K Trendyol dataset, A and B have learned the cybersecurity response style. The training stops, we save the final adapter. The base model weights were never modified.

---

## Part 8: Evaluating the Adapter

How do we know if the fine-tuning worked? Three approaches:

### 1. Training loss curve
Loss should decrease smoothly from ~2.5 (random) to ~0.8-1.2 (well-trained).
If it decreases too fast and plateaus near 0: overfitting (too many epochs, too small dataset).
If it doesn't decrease: learning rate too low, wrong data format, or configuration error.

### 2. Manual evaluation (qualitative)
Ask the model cybersecurity-specific questions before and after fine-tuning:
- "Explain CVE-2021-44228 (Log4Shell) and its attack vector"
- "What is the difference between MITRE ATT&CK T1059 and T1078?"
- "A user reports their credentials were compromised. What initial response steps do you recommend?"

Compare: does the fine-tuned model give more precise, domain-appropriate answers?

### 3. Perplexity on a held-out cybersecurity test set
Take 5% of the Trendyol dataset as a test split. Compute perplexity:
- `perplexity = exp(average cross-entropy loss)` on the test set
- Lower perplexity = model assigns higher probability to the correct tokens
- Compare base model perplexity vs fine-tuned model perplexity on this domain text
- A good SFT run typically reduces perplexity by 20-40% on domain data

Perplexity is our primary quantitative metric in Activity 0. In later activities, we use TTFT, throughput, and accuracy on benchmark tasks.

---

## Summary: The Activity 0 Training Flow

```
WHAT WE HAVE:
  meta-llama/Llama-3.1-8B-Instruct (frozen, 16GB in FP16, loaded as 4GB in NF4)
  Trendyol dataset: 53,000 instruction-response pairs (cybersecurity)
  CVE dataset: 30,000 sampled chat records (CVE analysis)
  Total: 83,000 training examples

WHAT HAPPENS:
  Format all data into ChatML format using tokenizer.apply_chat_template()
  Mask system + user tokens from loss (only train on assistant responses)
  Load base model in 4-bit NF4 with bitsandbytes
  Attach LoRA adapters to q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  Train for 3 epochs with AdamW (lr=2e-4), gradient checkpointing, bf16 compute
  Save adapter every 500 steps as checkpoint
  Final eval: perplexity on held-out 5% of Trendyol test split

WHAT WE GET:
  cybersec_analyst_lora/
    adapter_config.json      ← "trained on Llama-3.1-8B-Instruct, r=16, alpha=32"
    adapter_model.safetensors ← A and B matrices (~150MB)
  training_metrics.json      ← loss curve, eval perplexity, GPU hours used

WHAT IT ENABLES:
  Activity 1: Load this adapter + base model → merge → deploy with vLLM
  Activities 2-5: Optimize the merged model further
```

---

## The Next Concept

**Concept 01** will cover what happens *after* training: the merge operation in detail, how vLLM loads models, what PagedAttention is, and how we measure the baseline performance (TTFT, throughput, VRAM). That's the bridge from Activity 0 to Activity 1.
