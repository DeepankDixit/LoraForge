"""
activity1_baseline/evaluation/perplexity.py
=============================================
Evaluate the merged FP16 model on the held-out cybersecurity test set.

Two evaluations are run:

1. Perplexity evaluation (quantitative):
   - Loads the held-out 5% test split of the Trendyol dataset
   - Computes perplexity = exp(avg cross-entropy loss) on assistant tokens only
   - Compares base model perplexity vs merged model perplexity
   - Lower perplexity = model assigns higher probability to correct answers

2. Qualitative evaluation:
   - Runs 6 curated cybersecurity prompts (same as Activity 0 eval.py)
   - Generates responses from both base model and merged model
   - Saves side-by-side comparison to results/qualitative_eval.json

Why evaluate the MERGED model (not the adapter)?
  After merging, W_merged = W_base + (alpha/r) × B × A.
  The merge preserves fine-tuning quality — perplexity should be identical to
  the adapter+base combination. This evaluation verifies merge correctness.
  If perplexity is significantly worse than Activity 0's eval, the merge failed.

Expected results:
  Base model perplexity on cybersec test set: ~15–25
  Merged model perplexity: ~8–15 (40–50% reduction from Activity 0 fine-tuning)

Run:
  python -m activity1_baseline.evaluation.perplexity
  python -m activity1_baseline.evaluation.perplexity --base-only  # skip merged eval
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

CYBERSEC_SYSTEM_PROMPT = (
    "You are an expert cybersecurity analyst with deep knowledge of CVEs, "
    "MITRE ATT&CK, network security, and incident response."
)


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate merged model perplexity")
    parser.add_argument("--config", type=str, default="./activity1_baseline/config.yaml")
    parser.add_argument("--base-only", action="store_true",
                        help="Only evaluate base model (skip merged model load)")
    parser.add_argument("--merged-only", action="store_true",
                        help="Only evaluate merged model (skip base model)")
    parser.add_argument("--qualitative-only", action="store_true",
                        help="Only run qualitative evaluation (no perplexity computation)")
    return parser.parse_args()


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model_for_eval(model_path: str, device: str = "auto") -> tuple:
    """
    Load a model in FP16 for evaluation.

    Uses device_map="auto" to handle GPU memory automatically.
    If GPU has <18GB free, falls back to 4-bit quantization for evaluation
    (slight accuracy difference but acceptable for perplexity comparison).
    """
    logger.info(f"Loading model for evaluation: {model_path}")

    # Check available GPU memory
    use_4bit = False
    if torch.cuda.is_available():
        free_vram_gb = torch.cuda.mem_get_info()[0] / 1e9
        logger.info(f"Available GPU memory: {free_vram_gb:.1f}GB")
        if free_vram_gb < 18.0:
            logger.warning(f"Only {free_vram_gb:.1f}GB VRAM free — using 4-bit eval mode")
            logger.warning("Note: 4-bit eval perplexity will be slightly higher than FP16")
            use_4bit = True

    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": False,
    }

    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()   # Disable dropout for deterministic inference
    logger.info(f"Model loaded successfully")
    return model, tokenizer


# ── Perplexity Computation ────────────────────────────────────────────────────

def format_example_to_chatml(example: dict, tokenizer) -> str:
    """
    Format a dataset example to ChatML for perplexity computation.

    We apply the same formatting as Activity 0's training pipeline so that
    the perplexity comparison is apples-to-apples.
    """
    # Extract messages from the example
    messages_raw = example.get("messages", example.get("conversations", []))
    if not messages_raw:
        return None

    messages = []
    has_system = any(m.get("role") == "system" for m in messages_raw)
    if not has_system:
        messages.append({"role": "system", "content": CYBERSEC_SYSTEM_PROMPT})

    for msg in messages_raw:
        role = msg.get("role", msg.get("from", ""))
        role_map = {"human": "user", "gpt": "assistant", "bot": "assistant"}
        role = role_map.get(role, role)
        content = msg.get("content", msg.get("value", ""))
        if role in ("user", "assistant", "system") and content:
            messages.append({"role": role, "content": content})

    if not messages:
        return None

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    dataset_path: str,
    split: str = "test",
    max_samples: int = 500,
    max_seq_length: int = 2048,
    batch_size: int = 4,
) -> dict:
    """
    Compute perplexity on assistant tokens only (mirroring the SFT training objective).

    Returns a dict with perplexity value and metadata.

    Perplexity = exp(average cross-entropy loss)
    For SFT evaluation, we mask out system/user tokens and only compute loss
    on assistant response tokens — identical to how training loss was computed.
    """
    logger.info(f"Loading eval dataset: {dataset_path} (split={split}, max={max_samples})")

    try:
        dataset = load_dataset(dataset_path, split=split)
    except Exception:
        # Fall back to train split with 95/5 split if test doesn't exist
        logger.warning(f"Split '{split}' not found. Using last 5% of 'train' split.")
        full_dataset = load_dataset(dataset_path, split="train")
        eval_size = max(1, int(len(full_dataset) * 0.05))
        dataset = full_dataset.select(range(len(full_dataset) - eval_size, len(full_dataset)))

    # Sample if needed
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    logger.info(f"Evaluating on {len(dataset)} examples")

    # Find the assistant response start token ID for masking
    # In Llama-3.1 ChatML: assistant tokens start after <|start_header_id|>assistant<|end_header_id|>
    assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
    assistant_header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)

    total_loss = 0.0
    total_tokens = 0
    processed = 0
    errors = 0

    for batch_start in range(0, len(dataset), batch_size):
        batch = dataset.select(range(batch_start, min(batch_start + batch_size, len(dataset))))

        batch_texts = []
        for example in batch:
            text = format_example_to_chatml(dict(example), tokenizer)
            if text:
                batch_texts.append(text)

        if not batch_texts:
            continue

        # Tokenize batch
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
            padding=True,
            pad_to_multiple_of=8,
        )

        input_ids = encoding.input_ids.to(model.device)
        attention_mask = encoding.attention_mask.to(model.device)

        # Build labels: -100 for non-assistant tokens (excluded from loss)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100   # Mask padding tokens

        # Mask out non-assistant tokens (system + user turns)
        for seq_idx in range(labels.size(0)):
            seq = input_ids[seq_idx].tolist()
            in_assistant_turn = False
            for pos in range(len(seq) - len(assistant_header_ids)):
                if seq[pos:pos + len(assistant_header_ids)] == assistant_header_ids:
                    in_assistant_turn = True
                    # Mask the header itself
                    for j in range(len(assistant_header_ids)):
                        labels[seq_idx, pos + j] = -100
                elif in_assistant_turn:
                    # Check if we're entering a new turn (user or system)
                    eot_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
                    if seq[pos:pos + len(eot_id)] == eot_id:
                        in_assistant_turn = False  # End of assistant turn
                else:
                    labels[seq_idx, pos] = -100   # Non-assistant, mask it

        # Forward pass
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # outputs.loss is the mean cross-entropy over unmasked (assistant) tokens
        batch_loss = outputs.loss.item()

        # Count unmasked tokens for proper weighting
        unmasked_tokens = (labels != -100).sum().item()
        if unmasked_tokens > 0:
            total_loss += batch_loss * unmasked_tokens
            total_tokens += unmasked_tokens
            processed += len(batch_texts)

        if (batch_start // batch_size) % 10 == 0:
            if total_tokens > 0:
                current_ppl = math.exp(total_loss / total_tokens)
                logger.info(f"  Processed {processed}/{len(dataset)}, running PPL: {current_ppl:.2f}")

    if total_tokens == 0:
        logger.error("No valid tokens found for perplexity computation")
        return {"perplexity": None, "num_samples": 0, "error": "no valid tokens"}

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    logger.info(f"Perplexity: {perplexity:.2f} (avg_loss={avg_loss:.4f}, tokens={total_tokens})")
    return {
        "perplexity": perplexity,
        "avg_cross_entropy": avg_loss,
        "num_samples": processed,
        "num_tokens": total_tokens,
        "errors": errors,
    }


# ── Qualitative Evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    user_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """Generate a response for a single user prompt."""
    messages = [
        {"role": "system", "content": CYBERSEC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.cuda.amp.autocast(enabled=True):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (not the input prompt)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_qualitative_evaluation(
    model,
    tokenizer,
    prompts: list[str],
    model_label: str,
) -> list[dict]:
    """Run qualitative prompts and return responses."""
    logger.info(f"Running qualitative evaluation ({model_label}, {len(prompts)} prompts)...")
    results = []

    for i, prompt in enumerate(prompts):
        logger.info(f"  Prompt {i+1}/{len(prompts)}: {prompt[:80]}...")
        start = time.perf_counter()
        response = generate_response(model, tokenizer, prompt)
        elapsed = time.perf_counter() - start
        logger.info(f"  Generated {len(response.split())} words in {elapsed:.1f}s")
        results.append({
            "prompt": prompt,
            "response": response,
            "generation_time_seconds": elapsed,
        })

    return results


# ── Main Evaluation Orchestrator ──────────────────────────────────────────────

def run_evaluation(config: dict, args: argparse.Namespace) -> dict:
    """Run the full evaluation pipeline for Activity 1."""
    eval_cfg = config.get("evaluation", {})
    report_cfg = config.get("reporting", {})
    output_dir = Path(report_cfg.get("output_dir", "./results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    qualitative_prompts = eval_cfg.get("qualitative_prompts", [])
    results = {}

    # ── Base model evaluation ─────────────────────────────────────────────
    if not args.merged_only:
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING BASE MODEL")
        logger.info("=" * 60)
        base_model, base_tokenizer = load_model_for_eval(config["base_model_id"])

        if not args.qualitative_only:
            base_ppl = compute_perplexity(
                base_model, base_tokenizer,
                eval_cfg.get("test_dataset"),
                split=eval_cfg.get("test_split_name", "test"),
                max_samples=eval_cfg.get("max_eval_samples", 500),
                max_seq_length=eval_cfg.get("max_seq_length", 2048),
                batch_size=eval_cfg.get("batch_size", 4),
            )
            results["base_model_perplexity"] = base_ppl

        if qualitative_prompts:
            base_qual = run_qualitative_evaluation(
                base_model, base_tokenizer, qualitative_prompts, "base_model"
            )
            results["base_model_qualitative"] = base_qual

        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Merged model evaluation ───────────────────────────────────────────
    if not args.base_only:
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING MERGED FP16 MODEL")
        logger.info("=" * 60)
        merged_path = config.get("merged_model_path", "./outputs/cybersec_analyst_merged")
        merged_model, merged_tokenizer = load_model_for_eval(merged_path)

        if not args.qualitative_only:
            merged_ppl = compute_perplexity(
                merged_model, merged_tokenizer,
                eval_cfg.get("test_dataset"),
                split=eval_cfg.get("test_split_name", "test"),
                max_samples=eval_cfg.get("max_eval_samples", 500),
                max_seq_length=eval_cfg.get("max_seq_length", 2048),
                batch_size=eval_cfg.get("batch_size", 4),
            )
            results["merged_model_perplexity"] = merged_ppl

            # Compute perplexity improvement
            if "base_model_perplexity" in results:
                base_ppl_val = results["base_model_perplexity"].get("perplexity")
                merged_ppl_val = merged_ppl.get("perplexity")
                if base_ppl_val and merged_ppl_val:
                    improvement = (base_ppl_val - merged_ppl_val) / base_ppl_val * 100
                    results["perplexity_improvement_pct"] = improvement
                    logger.info(f"\nPerplexity reduction: {improvement:.1f}%")
                    logger.info(f"  Base: {base_ppl_val:.2f} → Merged: {merged_ppl_val:.2f}")

        if qualitative_prompts:
            merged_qual = run_qualitative_evaluation(
                merged_model, merged_tokenizer, qualitative_prompts, "merged_fp16"
            )
            results["merged_model_qualitative"] = merged_qual

    # ── Save results ──────────────────────────────────────────────────────
    eval_output_path = output_dir / "accuracy_eval_results.json"
    with open(eval_output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nEvaluation results saved to: {eval_output_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_evaluation(config, args)
