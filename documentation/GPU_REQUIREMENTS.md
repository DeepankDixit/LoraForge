# LoraForge — GPU Requirements & Cost Estimates

## Per-Activity GPU Requirements

| Activity | Min GPU | Recommended | Estimated Hours | Estimated Cost (Lambda Cloud) |
|---|---|---|---|---|
| 1: Baseline | 1x A100 40GB | 1x A100 40GB | 4–6 hrs | ~$6–9 |
| 2: Quantization | 1x A100 40GB | 1x A100 40GB | 6–10 hrs | ~$9–15 |
| 3: KV Cache | 1x A100 40GB | 1x A100 40GB | 4–6 hrs | ~$6–9 |
| 4: Speculative Decoding (training) | 2x A100 40GB | 4x A100 40GB | 24–48 hrs | ~$72–$144 |
| 4: Speculative Decoding (benchmarking) | 1x A100 40GB | 1x A100 40GB | 4–6 hrs | ~$6–9 |
| 5: Dashboard | CPU / 1x T4 | CPU / 1x T4 | 2–4 hrs | ~$1–3 |
| 6: CPT (1B token corpus) | 1x A100 80GB | 2x A100 40GB | 12–24 hrs | ~$18–36 |
| **TOTAL** | | | **~60–100 hrs** | **~$120–220** |

*Lambda Cloud pricing as of early 2026: A100 40GB ~$1.50/hr, A100 80GB ~$2.50/hr, H100 ~$3.50/hr*

---

## Recommended Cloud Providers

### Lambda Cloud (Primary Recommendation)
- Best GPU availability for A100s in India timezone
- Simple pricing, no commitment
- Good for burst workloads (rent for a weekend)
- URL: lambda.ai

### RunPod (Backup)
- Spot instances cheaper but less reliable
- Community cloud with good A100 availability
- Good for Activity 6 (CPT) where interruptions can be handled with checkpointing

### Google Colab Pro+ (For Activities 1–3 Only)
- A100 available on Colab Pro+ (~$50/month)
- Limited to 24hr sessions — sufficient for Activities 1, 2, 3
- Not recommended for Activity 4 (draft model training too long) or Activity 6 (CPT)

### Azure ML (Activity 6 Only — Familiar from Optum)
- You already have Azure ML experience from the AI Tutor project
- Can leverage existing Azure credits or familiarity
- Good for Activity 6 CPT if you have Azure access

---

## Memory Requirements by Activity

### Llama-3.1-8B Model Memory Footprint
| Precision | Model Weights | KV Cache (4K ctx, BS=8) | Total |
|---|---|---|---|
| FP16 | ~16 GB | ~4 GB | ~20 GB |
| FP8 | ~8 GB | ~2 GB | ~10 GB |
| INT4 AWQ | ~4 GB | ~2 GB | ~6 GB |
| INT4 AWQ + FP8 KV | ~4 GB | ~1 GB | ~5 GB |

### Practical Implication
With INT4 AWQ + FP8 KV cache, Llama-3.1-8B fits on a **16GB consumer GPU** (RTX 4090, RTX 3090). This means the final optimized model from your pipeline could theoretically run on a gaming GPU — a concrete, impressive outcome.

---

## Cost Optimization Tips

1. **Checkpoint frequently** — Every major step saves checkpoints. If a session ends, resume from last checkpoint. Use `trainer.save_pretrained()` after each epoch in Activity 6.

2. **Start with smaller runs** — Before running the full 48hr Activity 4 training, do a 1hr test run with 1000 samples to verify the training loop works.

3. **Batch your work** — Rent a GPU, complete all benchmarking for one activity in a single session. GPU rental has no startup cost, just hourly billing.

4. **Use spot/preemptible instances for CPT** — Activity 6 training can be interrupted and resumed. Lambda Cloud spot instances are ~40% cheaper.

5. **RunPod community cloud for long runs** — RunPod's community cloud is cheapest for long jobs that can tolerate occasional interruption (e.g., 24hr CPT).

---

## Free Tier Options

- **Kaggle Notebooks**: 2x T4 GPUs free, 30hr/week. Sufficient for Activity 1 baseline with smaller model
- **Google Colab Free**: T4 GPU, ~12hr sessions. Very limited but useful for code testing
- **NVIDIA LaunchPad**: Free access to H100 clusters for researchers — worth applying if you have an IISc email

---

## Hardware for Local Development

For development and testing (not full training/benchmarking):
- **RTX 3090 / RTX 4090** — Runs quantized Llama-3.1-8B in INT4. Good for local iteration
- **Mac M1/M2/M3 Pro with 32GB** — Runs quantized 8B models via llama.cpp. Useful for testing pipeline logic
- **Your current laptop** — Sufficient for code development, testing with tiny models (Phi-3-mini, SmolLM)

The actual benchmarking must be done on cloud A100s for credible numbers (consumer GPUs have different memory bandwidth characteristics).
