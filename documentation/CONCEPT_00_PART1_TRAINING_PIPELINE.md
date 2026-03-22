# Concept 00 — Part 1 of 3: The LLM Training Pipeline
## Pre-training → Instruction Tuning → CPT → Inference Optimization

**Activity mapping:** CONCEPT_00 (Parts 1–3) → **Activity 0** (SFT). Read all three parts before writing Activity 0 code.

---

## The Big Picture: How an LLM Gets Made

Before we write a single line of inference optimization code, you need a clear mental model
of how a model like Llama-3.1-8B comes to exist and how it gets to production.
There are four distinct stages. Most engineers only ever touch stages 1 and 2.
LoraForge touches all of them — that is what makes it unusual.

**Important clarification before reading the table below:**
Stage 0B (Meta's instruction tuning) and Stage 2 (what we do in Activity 0) are the
**same technique** — Supervised Fine-Tuning on instruction-response pairs. The only
difference is scale and method. Meta does it with millions of examples + RLHF on thousands
of GPUs. We do it with 83K domain examples + LoRA on a single GPU. The learning experience
is identical. Activity 0 IS instruction tuning. We are not skipping Stage 0B.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE COMPLETE LLM LIFECYCLE                               │
├──────────┬──────────────────────────────────────────────────────────────────┤
│ Stage 0  │  PRE-TRAINING                                                    │
│          │  Who does it: Meta, Google, Mistral, Anthropic                   │
│          │  Data: Trillions of tokens (web, books, code, Wikipedia)         │
│          │  Objective: Predict the next token (autoregressive LM)           │
│          │  Cost: $10M – $100M+, months of training on thousands of GPUs    │
│          │  Output: Base model weights (e.g., Llama-3.1-8B-Base)            │
│          │  YOU DO NOT DO THIS (cost and scale are prohibitive)             │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ Stage 0B │  INSTRUCTION TUNING AT SCALE (RLHF / DPO / SFT)                │
│          │  Who does it: Meta (for Llama-Instruct), Mistral, etc.           │
│          │  Data: Millions of instruction-response pairs + human feedback   │
│          │  Purpose: Teach the model to follow instructions politely        │
│          │  Output: Llama-3.1-8B-INSTRUCT (what we start from)             │
│          │  YOU DO NOT DO THIS AT THIS SCALE — but the technique is         │
│          │  IDENTICAL to what Activity 0 does. Activity 0 IS instruction    │
│          │  tuning, just domain-focused and at practitioner scale.          │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ Stage 1  │  CONTINUED PRE-TRAINING (CPT) ← Activity 6 in LoraForge        │
│          │  Who does it: YOU (or domain experts)                            │
│          │  Data: Domain-specific RAW TEXT (papers, docs, code, manuals)    │
│          │  Objective: Same as pre-training — predict next token            │
│          │  Purpose: Teach the model your domain's vocabulary & facts       │
│          │  Cost: Manageable with LoRA — ~$50-200 on Lambda Cloud           │
│          │  Output: LoRA adapter with domain knowledge baked in             │
│          │  OPTIONAL — only needed for highly specialized domains           │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ Stage 2  │  SFT / INSTRUCTION TUNING ← Activity 0 in LoraForge            │
│          │  Who does it: YOU                                                 │
│          │  Data: 83K instruction-response pairs (Trendyol + CVE datasets)  │
│          │  Objective: Predict only the assistant's response tokens         │
│          │  Purpose: Teach the model HOW to respond in the cybersecurity    │
│          │           domain — response style, format, terminology           │
│          │  Cost: 6-8 hours on a single A10G GPU (~$7 on Lambda Cloud)      │
│          │  Output: cybersec_analyst_lora/ ← the LoRA adapter we build      │
│          │  THIS IS WHAT "LoRA ADAPTER" MEANS IN THIS PROJECT               │
│          │  THIS IS INSTRUCTION TUNING. WE DO NOT SKIP IT.                  │
├──────────┼──────────────────────────────────────────────────────────────────┤
│ Stage 3  │  INFERENCE OPTIMIZATION ← Activities 1–5 in LoraForge          │
│          │  Who does it: YOU (inference engineers)                          │
│          │  Input: Base model + LoRA adapter from Stage 2 (Activity 0)      │
│          │  Operations: Merge → Quantize → Configure KV Cache →             │
│          │              Speculative Decoding → Benchmark → Deploy           │
│          │  Purpose: Make the model fast, cheap, and scalable               │
│          │  Cost: ~$50-100 on Lambda Cloud for full benchmarking            │
│          │  Output: Optimized deployment artifact + benchmark report        │
│          │  THIS IS THE CORE OF LORAFORGE                                   │
└──────────┴──────────────────────────────────────────────────────────────────┘
```

---

## The Key Insight: Stage 0B and Activity 0 Are the Same Technique

This is the most important clarification in this document.

When Meta trains Llama-3.1-8B-**Instruct** from Llama-3.1-8B-**Base**, they do SFT:
- They collect millions of (instruction, response) pairs
- They train the model to predict the response tokens given the instruction
- They optionally run RLHF/DPO on top for alignment

When **we** run Activity 0, we do the exact same thing:
- We collect 83K (instruction, response) pairs from cybersecurity datasets
- We train the model to predict the response tokens given the instruction
- We use LoRA to do it efficiently on a single GPU

**The technique is identical. The scale is different. The learning is the same.**

The reason the pipeline table shows "Stage 0B" and "Stage 2" as separate rows is only to
distinguish Meta's industrial-scale operation (Instruct model creation) from your
practitioner-scale domain fine-tuning (Activity 0). Under the hood they are both:
```
Loss = CrossEntropy(model_output[assistant_tokens], true_assistant_tokens)
```

Activity 0 gives you genuine instruction tuning experience. You will:
- Choose and curate a domain-specific dataset (Trendyol + CVE)
- Format data into ChatML (instruction-response pairs)
- Configure LoRA hyperparameters (rank, alpha, target modules)
- Run QLoRA training and watch the loss curve converge
- Evaluate whether the model's cybersecurity responses improved
- Save a LoRA adapter that you own and can inspect

---

## The Question: "Do We Need CPT?"

**Short answer: No. LoraForge works perfectly without it.**

Activity 0 (SFT/instruction tuning) is already in scope and is the first activity.
Activities 1–5 are about inference optimization of the adapter we build in Activity 0.
Activity 6 (CPT) is an *elective* that demonstrates the full pipeline — it is different
from SFT because it trains on raw text (not instruction-response pairs) to inject
domain knowledge before fine-tuning.

**When would you actually need CPT?**

CPT is distinct from SFT (Activity 0). Activity 0 teaches the model *how to respond* in
cybersecurity (instruction-response style). CPT would teach the model *domain facts* by
training on raw text. Ask yourself: *Does the base model already understand my domain deeply
enough that SFT alone is sufficient?*

- Llama-3.1-8B was trained on a massive web corpus. It knows general programming,
  medicine, law, science, and everyday topics reasonably well.
- BUT: It does NOT know internal company SOPs, proprietary systems, niche security
  vulnerabilities (post-2023 CVEs), or highly specialized Indian regulatory frameworks.

**CPT matters when:** Your domain has vocabulary, facts, or reasoning patterns that
simply do not appear in the pre-training corpus. Examples:
  - A model for Indian GST compliance law (very specific, often in Hindi/regional languages)
  - A model for Cisco ISE syslog analysis (internal product, niche format)
  - A model for rare disease diagnostics (highly specialized medical literature)

**CPT does NOT matter when:** The base model already has sufficient domain knowledge
and you just need it to respond in a particular *style* or follow particular *instructions*.
That's what SFT (fine-tuning) is for.

**For LoraForge specifically:** We do CPT in Activity 6 because it demonstrates
the full lifecycle and makes the project story compelling — not because inference
optimization requires it.

---

## The Critical Distinction: CPT vs Fine-Tuning

This is the most commonly confused pair of concepts. Burn this into memory:

```
                    CPT                          SFT (Fine-tuning)
                    ───                          ─────────────────
Training data:  RAW TEXT                     INSTRUCTION + RESPONSE PAIRS
                "The TLS handshake           {"role": "user",
                 begins when the client       "content": "What is TLS?"}
                 sends a ClientHello..."      {"role": "assistant",
                                              "content": "TLS is..."}

Training        NEXT TOKEN PREDICTION        NEXT TOKEN PREDICTION
objective:      on ALL tokens                on RESPONSE TOKENS ONLY
                (assistant-only loss = OFF)  (assistant-only loss = ON)

What the        Domain knowledge:            Response behavior:
model learns:   vocabulary, facts,           how to answer, what format,
                writing style, patterns      what tone, what boundaries

Analogy:        Reading medical textbooks    Doing medical residency
                for years                    (supervised practice)

Output:         A model that KNOWS more      A model that RESPONDS better
                about the domain             in your domain

In LoraForge:   Activity 6 (optional)        What you did at Optum
                                             = the "LoRA adapter" we deploy
```

---

## What Is a LoRA Adapter, Exactly?

You used LoRA at Optum (q_proj/v_proj across 32 transformer blocks, <0.1% of params).
Here's what that actually means:

### The core problem LoRA solves

Full fine-tuning of Llama-3.1-8B means updating all 8 billion parameters.
That requires enormous memory and time. Worse, it often overfits small datasets.

### How LoRA works

LoRA says: instead of updating the full weight matrix W (8192 × 8192 for attention),
train two small matrices A and B such that the update ΔW = A × B.

```
Full fine-tuning:
  W_new = W_original + ΔW       (ΔW is 8192×8192 = 67M params, per layer)

LoRA fine-tuning:
  W_new = W_original + (A × B)  (A is 8192×16, B is 16×8192 = 262K params)
                                   ↑                    ↑
                                   rank=16 adapter      only 0.4% of original size
```

The LoRA adapter = just the A and B matrices for each targeted layer.
They are saved separately from the base model (~100–200MB vs ~16GB for the full model).

### At inference time (Activity 1)

```
Option A — Merged model (what we do in Activity 1):
  W_deployed = W_original + (A × B)
  → Single model file, no special serving needed
  → Slightly slower to set up but simpler to serve

Option B — Dynamic LoRA (vLLM supports this):
  Load base model once, load adapter on the fly per request
  → Useful when serving MULTIPLE fine-tuned adapters simultaneously
  → vLLM's --enable-lora flag activates this
```

For our baseline (Activity 1), we merge and deploy the merged model.
This is why the architecture diagram says "Merge base + LoRA adapter → FP16 model."

---

## The Dataset Question: What Do We Use for CPT?

For Activity 6, we need a domain corpus — raw text in our chosen domain.
Here are the realistic options and my recommendation:

### Option 1: Cybersecurity (RECOMMENDED for your profile)

Why: You have 4 years of Cisco security experience. You understand the domain.
You can evaluate whether the CPT actually improved the model's security knowledge.
The domain vocabulary (CVEs, TTPs, MITRE ATT&CK, syslog formats, TLS handshakes)
is genuinely underrepresented in general pre-training data.

**Datasets:**
- `mitre_attack_descriptions` — MITRE ATT&CK technique descriptions (public)
- CVE descriptions from NVD (National Vulnerability Database) — public, downloadable
- CyberSecEval prompts and completions (Meta, open source)
- Security Stack Exchange Q&A (scraped, freely available)
- USENIX Security / CCS paper abstracts (academic, public)
- Cisco security documentation (public documentation pages)

**Estimated corpus size:** 500M–1B tokens is achievable from public sources.
This is sufficient for meaningful domain adaptation.

### Option 2: Healthcare (relevant to Optum context)

Why: You work at Optum (UnitedHealth). You built an AI Tutor for healthcare employees.
Deep familiarity with the domain.

**Datasets:**
- PubMed abstracts (~35M abstracts, freely available via E-utilities API)
- MedC-I (medical instruction dataset)
- Clinical guidelines (NIH, WHO — public)
- MIMIC notes (requires DUA agreement, not for open-sourcing)

**Estimated corpus size:** Easily 2–5B tokens from PubMed alone.

### Option 3: India-grounded (the NVIDIA angle)

Why: NVIDIA explicitly showed the `Nemotron-Personas-India` dataset in the workshop
slides you attended. Using it directly ties your project to NVIDIA's current work.
Strong LinkedIn story: "I used the dataset NVIDIA presented at IISc."

**BUT — important clarification:** Nemotron-Personas-India is a **synthetic persona dataset**
(21M personas, grounded in census data). It is ABOUT Indian people, not domain text.
It is better suited for:
  - Teaching a model to generate culturally-grounded Indian personas
  - Alignment/RLHF applications
  - NOT for domain-specific CPT like security or medical

If you want the India angle, pair it with:
- `IndicCorp v2` — 20.9 billion words across 24 Indian languages (CC-BY)
- `Sangraha` — curated high-quality Indian web text (AI4Bharat, open-source)
- `AI4Bharat IndicHeadlines` — Indian news corpus

### My recommendation for LoraForge

**Use cybersecurity.** Here is why:
1. Your Cisco background gives you credibility to evaluate the results
2. Public datasets are available without sign-up or agreements
3. The domain is genuinely underrepresented in Llama's training data
4. It creates a coherent story: "I know security from Cisco, so I domain-adapted
   an LLM for security reasoning, then optimized its inference pipeline"
5. Cybersecurity LLMs are actively being hired for at companies like Palo Alto,
   CrowdStrike, SentinelOne — adjacent targets to your recruiter list

---

## Mental Model: The Two "Languages" of Training

Here is the simplest possible way to think about CPT vs SFT:

**CPT teaches the model a new language.**
After CPT on security text, the model "speaks" security fluently.
It knows what a CVE is, what lateral movement means, what a SIEM does.
It learned this the same way a human learns: by reading a lot about it.

**SFT teaches the model how to have a conversation.**
After SFT on instruction-response pairs, the model knows WHEN to use its knowledge.
It knows to answer concisely, follow instructions, stay in character.
It learned this by being supervised in practice conversations.

**You need both for a good assistant.** A model that speaks security fluently but
gives rambling, unfocused answers is not useful. A model that answers perfectly in
format but doesn't know what a CVE is cannot help a security analyst.

---

## Summary: What LoraForge Actually Does, Stage by Stage

```
INPUT: meta-llama/Llama-3.1-8B-Instruct (publicly available on HuggingFace)

ACTIVITY 0: "Build our own cybersecurity instruction-tuned adapter" ← INSTRUCTION TUNING
  SFT on 83K cybersecurity instruction-response pairs (Trendyol + CVE)
  QLoRA: 4-bit base + LoRA rank=16 on a single A10G GPU
  Output: cybersec_analyst_lora/ adapter (~150MB)
  This IS Stage 0B / instruction tuning — just at practitioner scale with LoRA

ACTIVITY 6 [OPTIONAL — comes BEFORE Activity 0 if done]:
  If we want to inject raw domain knowledge first → CPT on security corpus

ACTIVITY 1: "How bad is the naive deployment?"
  Merge base + Activity 0 adapter → Deploy FP16 in vLLM → Measure baseline

ACTIVITY 2: "Can we make it smaller without breaking it?"
  Apply quantization (FP8, INT4, INT8) → Compare accuracy vs memory tradeoffs

ACTIVITY 3: "Can we make it faster for multi-turn conversations?"
  Configure KV cache prefix sharing → Measure TTFT reduction

ACTIVITY 4: "Can we generate tokens faster?"
  Train EAGLE draft model → Deploy speculative decoding → Measure speedup

ACTIVITY 5: "Can we package this as a tool?"
  CLI + dashboard → pip install loraforge → full benchmark in 20 minutes

OUTPUT: Optimized model (4GB instead of 16GB, 3x faster generation, <2% accuracy loss)
        + Reproducible benchmark report
        + Open-source tool on PyPI and GitHub
```

---

## What to Do Next

Activity 0 (SFT / instruction tuning) is already fully specified and coded. The domain
choice is made: **cybersecurity**, using the Trendyol + CVE datasets.

The next steps are:
1. Read **Concept 00 — Part 2** — LoRA adapter anatomy, compatibility, and what merging means
2. Read **Concept 00 — Part 3** — the SFT training objective, catastrophic forgetting, and QLoRA
3. Spin up a Lambda Cloud A10G GPU and run:
   ```
   python -m activity0_sft.training.qlora_trainer --dry-run   # 5-step sanity check
   python -m activity0_sft.training.qlora_trainer             # full 6-8 hour run
   ```
4. Once the adapter is saved, move to Activity 1 (baseline deployment)

**If you want to also do CPT (Activity 6):** it would come *before* Activity 0 in the
pipeline — train on raw cybersecurity text first, then SFT on top. This is optional and
not required to complete Activities 1–5.
