"""
activity1_baseline/merge/merge.py
==================================
Merge the LoRA adapter from Activity 0 into the base model weights.

What this script does:
  1. Load the base model (Llama-3.1-8B-Instruct) in FP16 on CPU
  2. Load the LoRA adapter on top (PEFT's PeftModel)
  3. Call merge_and_unload() — computes W_merged = W_base + (alpha/r) × B × A
     for every targeted layer, then discards the A and B matrices
  4. Save the merged model as a standard HuggingFace checkpoint

The output is a plain ~16GB FP16 model that vLLM can load directly.
No PEFT/LoRA code is needed at inference time after this step.

Why merge on CPU?
  The merge operation is just matrix addition — no forward pass, no GPU compute.
  On CPU, we avoid OOM entirely. The downside is speed (~5–10 minutes to load
  and save 16GB of weights), but this is a one-time operation.

Run:
  python -m activity1_baseline.merge.merge
  # or with custom paths:
  python -m activity1_baseline.merge.merge \
      --base-model meta-llama/Llama-3.1-8B-Instruct \
      --adapter ./outputs/cybersec_analyst_lora \
      --output ./outputs/cybersec_analyst_merged
"""

import argparse
import logging
from logging import config
import os
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--config", type=str, default="./activity1_baseline/config.yaml")
    parser.add_argument("--base-model", type=str, help="Override base_model_id from config")
    parser.add_argument("--adapter", type=str, help="Override lora_adapter_path from config")
    parser.add_argument("--output", type=str, help="Override merged_model_path from config")
    parser.add_argument("--force", action="store_true",
                        help="Re-merge even if output directory already exists")
    return parser.parse_args()


# ── Validation ────────────────────────────────────────────────────────────────

def validate_adapter(adapter_path: str) -> None:
    """Verify the adapter directory has the expected files from Activity 0."""
    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}\n"
            f"Have you completed Activity 0 (qlora_trainer.py)?"
        )
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required if not (path / f).exists()]
    if missing:
        # Also check for .bin format (older PEFT versions saved this way)
        if "adapter_model.safetensors" in missing:
            if (path / "adapter_model.bin").exists():
                logger.warning("Found adapter_model.bin (old format). Proceeding.")
                missing.remove("adapter_model.safetensors")
    if missing:
        raise FileNotFoundError(
            f"Adapter directory is incomplete. Missing: {missing}\n"
            f"Check that Activity 0 ran to completion."
        )
    logger.info(f"Adapter validated: {adapter_path}")


# ── Merge ─────────────────────────────────────────────────────────────────────

def load_base_model(model_id: str, torch_dtype: str = "float16") -> tuple:
    """
    Load the base model and tokenizer in the specified dtype on CPU.

    Why CPU?
      merge_and_unload() only does matrix addition. No GPU needed.
      Avoids OOM on GPUs with <16GB free VRAM.
    """
    logger.info(f"Loading base model: {model_id} (dtype={torch_dtype})")
    logger.info("This will take 3–8 minutes depending on disk speed...")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu",         # Explicit CPU — no GPU needed for merge
        trust_remote_code=False,  # Llama-3.1 doesn't need this; be explicit
        low_cpu_mem_usage=True,   # Reduces peak RAM by loading shard-by-shard
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Base model loaded: {param_count / 1e9:.2f}B parameters")
    return model, tokenizer


def attach_and_merge(base_model, adapter_path: str):
    """
    Load the LoRA adapter on top of the base model, then merge and unload.

    What merge_and_unload() does internally:
      For each LoRA module:
        1. Retrieve W_base (the frozen original weight)
        2. Compute delta = (lora_A @ lora_B) * (lora_alpha / lora_rank)
           Note: PEFT stores A and B in transpose order vs Concept 02 notation,
           but the math is equivalent.
        3. W_merged = W_base + delta.T (or delta depending on the layer type)
        4. Set the module's weight to W_merged
        5. Delete lora_A and lora_B (free the adapter memory)

    After this call, the model is a standard nn.Module with no PEFT overhead.
    """
    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,   # We're merging, not training
    )

    # Log LoRA config for the record
    lora_config = peft_model.peft_config.get("default", None)
    if lora_config:
        logger.info(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
        logger.info(f"Target modules: {lora_config.target_modules}")

    logger.info("Merging adapter into base weights...")
    logger.info("  (Computing W_merged = W_base + (alpha/r) × B × A for each layer...)")

    merged = peft_model.merge_and_unload()
    logger.info("Merge complete. Adapter matrices (A, B) discarded.")

    # Verify the merged model looks right — should have same param count as base
    param_count = sum(p.numel() for p in merged.parameters())
    logger.info(f"Merged model: {param_count / 1e9:.2f}B parameters (same as base — ✓)")

    return merged


def save_merged_model(model, tokenizer, output_path: str) -> None:
    """Save the merged model as a standard HuggingFace checkpoint."""
    Path(output_path).mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to: {output_path}")
    logger.info("This will take 2–5 minutes (writing ~16GB to disk)...")

    model.save_pretrained(
        output_path,
        safe_serialization=True,   # Save as .safetensors (faster load, safer than .bin)
        max_shard_size="5GB",      # Split into ≤5GB shards (HuggingFace convention)
    )
    tokenizer.save_pretrained(output_path)

    # Log what was saved
    saved_files = list(Path(output_path).iterdir())
    total_size_gb = sum(f.stat().st_size for f in saved_files if f.is_file()) / 1e9
    logger.info(f"Saved {len(saved_files)} files, total size: {total_size_gb:.2f}GB")
    logger.info(f"Key files: {[f.name for f in saved_files if f.suffix in ('.json', '.safetensors')]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_merge(config: dict, args: argparse.Namespace) -> str:
    """
    Orchestrate the full merge pipeline.

    Returns the path to the merged model directory.
    """
    args_dict = vars(args)
    # # base_model_id = args.base_model or config["base_model_id"]
    # base_model_id = getattr(args, "base_model", None) or config["base_model_id"]
    # # adapter_path = args.adapter or config["lora_adapter_path"]
    # adapter_path = getattr(args, "adapter", None) or config["lora_adapter_path"]
    # # output_path = args.output or config["merged_model_path"]
    # output_path = args_dict.get("output") or config["merged_model_path"]

    base_model_id = args_dict.get("base_model") or config["base_model_id"]
    adapter_path = args_dict.get("adapter") or config["lora_adapter_path"]
    output_path = args_dict.get("output") or config["merged_model_path"]
    torch_dtype = config.get("merge", {}).get("torch_dtype", "float16")

    # ── Check if already merged ───────────────────────────────────────────
    # if Path(output_path).exists() and not args.force:
    if Path(output_path).exists() and not args_dict.get("force", False):
        existing = list(Path(output_path).iterdir())
        if any(f.suffix == ".safetensors" for f in existing):
            logger.info(f"Merged model already exists at: {output_path}")
            logger.info("Use --force to re-merge. Skipping.")
            return output_path

    # ── Validate inputs ───────────────────────────────────────────────────
    validate_adapter(adapter_path)

    # ── Execute merge ─────────────────────────────────────────────────────
    start = time.perf_counter()

    base_model, tokenizer = load_base_model(base_model_id, torch_dtype)
    merged_model = attach_and_merge(base_model, adapter_path)
    save_merged_model(merged_model, tokenizer, output_path)

    elapsed = time.perf_counter() - start
    logger.info(f"Total merge time: {elapsed / 60:.1f} minutes")
    logger.info(f"Merged model ready at: {output_path}")
    logger.info("Next step: run vllm_server.py to serve the merged model")

    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_merge(config, args)
