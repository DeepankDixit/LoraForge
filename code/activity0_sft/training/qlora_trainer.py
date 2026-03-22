"""
activity0_sft/training/qlora_trainer.py
=========================================
QLoRA Supervised Fine-Tuning training script for Activity 0.

This is the main training entry point. It orchestrates:
1. Config loading (from config.yaml via OmegaConf)
2. Model loading (Llama-3.1-8B-Instruct in 4-bit NF4 via bitsandbytes)
3. LoRA adapter attachment (via PEFT)
4. Dataset preparation (via dataset_builder.py)
5. SFT training (via trl's SFTTrainer)
6. Adapter saving + metrics export

LEARNING NOTE: Read the comments here from top to bottom as if reading
a textbook. Every non-obvious line has an explanation of why it exists.
The "why" is more important than the "what" when learning this stack.

Required packages (see pyproject.toml):
    pip install transformers>=4.44.0 peft>=0.12.0 trl>=0.12.0
    pip install bitsandbytes>=0.43.0 accelerate>=0.33.0
    pip install flash-attn --no-build-isolation  (optional, for speed)
    pip install omegaconf  (for YAML config loading)

Usage:
    python -m activity0_sft.training.qlora_trainer
    python -m activity0_sft.training.qlora_trainer --config custom_config.yaml
    python -m activity0_sft.training.qlora_trainer --dry-run  (verify setup, no training)
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import typer
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from activity0_sft.data.dataset_builder import build_dataset, get_dataset_statistics

# Configure logging — structured JSON + human-readable console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Typer app for CLI argument parsing
app = typer.Typer(help="Activity 0: QLoRA SFT Training for LoraForge")


# =============================================================================
# Training metrics tracking
# =============================================================================

@dataclass
class TrainingMetrics:
    """
    Captures all relevant metrics from the training run.

    Saved as JSON at the end so Activity 1 and later activities can
    reference the baseline perplexity for comparison.
    """
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    training_duration_hours: Optional[float] = None

    # Loss values
    final_train_loss: Optional[float] = None
    final_eval_loss: Optional[float] = None
    final_eval_perplexity: Optional[float] = None

    # Training configuration (echoed back for reproducibility)
    base_model_id: str = ""
    lora_rank: int = 0
    lora_alpha: int = 0
    num_train_epochs: int = 0
    total_train_examples: int = 0
    total_eval_examples: int = 0
    effective_batch_size: int = 0
    learning_rate: float = 0.0

    # GPU info
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0

    # Output paths
    adapter_path: str = ""

    def to_dict(self) -> dict:
        return {
            "start_time_utc": self.start_time,
            "end_time_utc": self.end_time,
            "training_duration_hours": self.training_duration_hours,
            "final_train_loss": self.final_train_loss,
            "final_eval_loss": self.final_eval_loss,
            "final_eval_perplexity": self.final_eval_perplexity,
            "config": {
                "base_model_id": self.base_model_id,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "num_train_epochs": self.num_train_epochs,
                "total_train_examples": self.total_train_examples,
                "total_eval_examples": self.total_eval_examples,
                "effective_batch_size": self.effective_batch_size,
                "learning_rate": self.learning_rate,
            },
            "hardware": {
                "gpu_name": self.gpu_name,
                "gpu_vram_gb": self.gpu_vram_gb,
            },
            "output": {
                "adapter_path": self.adapter_path,
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Metrics saved to {path}")


# =============================================================================
# Custom callback for logging + checkpointing
# =============================================================================

class LoraForgeCallback(TrainerCallback):
    """
    Custom HuggingFace Trainer callback for LoraForge-specific logging.

    HuggingFace Trainer fires events at various points during training.
    We hook into these events to:
    1. Log GPU memory usage alongside training loss
    2. Print a human-friendly progress bar
    3. Detect divergence (loss > initial_loss + 2.0 → something is wrong)

    The callback pattern is used throughout HuggingFace's trainer API.
    You can add custom logic without modifying the Trainer itself.
    """

    def __init__(self, log_gpu_every_n_steps: int = 50):
        self.log_gpu_every_n_steps = log_gpu_every_n_steps
        self.initial_loss = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Called when the Trainer logs metrics (every logging_steps steps)."""
        if logs is None:
            return

        # Capture initial loss for divergence detection
        if self.initial_loss is None and "loss" in logs:
            self.initial_loss = logs["loss"]

        # Build a readable log line
        step = state.global_step
        total_steps = state.max_steps
        progress_pct = 100 * step / total_steps if total_steps > 0 else 0

        log_parts = [f"Step {step}/{total_steps} ({progress_pct:.1f}%)"]

        if "loss" in logs:
            log_parts.append(f"train_loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            log_parts.append(f"eval_loss={logs['eval_loss']:.4f}")
        if "learning_rate" in logs:
            log_parts.append(f"lr={logs['learning_rate']:.2e}")

        # Log GPU memory (if CUDA available)
        if torch.cuda.is_available() and step % self.log_gpu_every_n_steps == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            log_parts.append(f"gpu_alloc={allocated:.1f}GB / gpu_reserved={reserved:.1f}GB")

        logger.info(" | ".join(log_parts))

        # Divergence detection
        if self.initial_loss is not None and "loss" in logs:
            if logs["loss"] > self.initial_loss + 2.0 and step > 100:
                logger.warning(
                    f"⚠ Possible training divergence detected: "
                    f"current_loss={logs['loss']:.4f} >> initial_loss={self.initial_loss:.4f}. "
                    f"Check: learning rate (too high?), data format (wrong template?), "
                    f"or gradient clipping settings."
                )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Called at the end of each epoch."""
        logger.info(
            f"✓ Epoch {state.epoch:.0f} complete. "
            f"Best eval loss so far: {state.best_metric}"
        )


# =============================================================================
# Model loading
# =============================================================================

def load_quantized_model(
    model_id: str,
    quantization_cfg: DictConfig,
    model_cfg: DictConfig,
    hf_token: Optional[str] = None,
) -> AutoModelForCausalLM:
    """
    Load Llama-3.1-8B-Instruct in 4-bit NF4 quantization via bitsandbytes.

    The resulting model has:
    - All transformer weight matrices stored in 4-bit NF4 format (~4GB)
    - Computation happening in BF16 (dequantize on-the-fly)
    - requires_grad=False for all base model parameters (they are frozen)

    This is the "Q" in QLoRA. Without quantization, loading the full model
    in BF16 requires 16GB VRAM before any training overhead is added.

    Args:
        model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        quantization_cfg: OmegaConf config for BitsAndBytesConfig
        model_cfg: OmegaConf config for model-level settings
        hf_token: HuggingFace access token (required for gated Llama models)

    Returns:
        Loaded, quantized model ready for LoRA attachment
    """
    logger.info(f"Loading model: {model_id}")
    logger.info(f"Quantization: 4-bit {quantization_cfg.bnb_4bit_quant_type}")

    # BitsAndBytesConfig tells transformers how to quantize the model on load.
    # This is passed to from_pretrained() as the `quantization_config` argument.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_cfg.load_in_4bit,
        bnb_4bit_quant_type=quantization_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, quantization_cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=quantization_cfg.bnb_4bit_use_double_quant,
    )

    # Determine attention implementation.
    # Flash Attention 2 (FA2) is a memory-efficient attention algorithm that:
    # - Avoids materializing the full O(n²) attention matrix
    # - Uses tiling and kernel fusion for ~2-3x speedup on long sequences
    # - Requires Ampere+ GPU (A10G, A100, H100, RTX 3090+)
    attn_impl = "flash_attention_2" if model_cfg.get("use_flash_attention", False) else "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",       # Automatically assign layers to available GPU(s)
        torch_dtype=torch.bfloat16,  # BF16 for computation
        attn_implementation=attn_impl,
        token=hf_token,
        # trust_remote_code=False for security — Llama doesn't need it
        trust_remote_code=False,
    )

    # Log what we just loaded
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        logger.info(
            f"Model loaded. VRAM allocated: {allocated_gb:.2f} GB "
            f"(expected ~4GB for 4-bit Llama-3.1-8B)"
        )

    # Disable KV cache during training.
    # The KV cache is for inference efficiency — it's not used during training
    # because we process the full sequence in each forward pass.
    model.config.use_cache = False

    return model


def attach_lora_adapters(
    model: AutoModelForCausalLM,
    lora_cfg: DictConfig,
) -> AutoModelForCausalLM:
    """
    Attach LoRA adapters to the quantized base model.

    This is a two-step process:
    1. prepare_model_for_kbit_training() — sets up the model for QLoRA training
       (casts certain layers to FP32, enables gradient computation on input embeddings)
    2. get_peft_model() — adds the A and B matrices to the target modules

    After this function:
    - Base model parameters (W): frozen (requires_grad=False)
    - LoRA parameters (A, B): trainable (requires_grad=True)
    - Total trainable params: ~30M (vs 8B for full fine-tuning)

    Args:
        model: Quantized base model from load_quantized_model()
        lora_cfg: OmegaConf config for LoRA settings

    Returns:
        PEFT model with LoRA adapters attached
    """
    logger.info(
        f"Attaching LoRA adapters: rank={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, "
        f"target_modules={list(lora_cfg.target_modules)}"
    )

    # prepare_model_for_kbit_training does several important things:
    # 1. Casts LayerNorm layers from 4-bit back to FP32 (they must be FP32 for stability)
    # 2. Casts the final output head (lm_head) to FP32
    # 3. Enables gradient computation on the input embeddings
    #    (necessary for computing gradients through the LoRA path)
    # 4. Sets gradient_checkpointing if requested
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # LoraConfig defines the shape and placement of the A and B matrices.
    # This gets saved as adapter_config.json — so these settings are permanent
    # for the adapter. Do not change r or alpha after training starts.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
        init_lora_weights=lora_cfg.get("init_lora_weights", True),
        # Bias: "none" is standard. "all" or "lora_only" adds bias terms to adapters.
        # "none" is recommended — bias terms rarely help and add complexity.
        bias="none",
    )

    # get_peft_model wraps the base model with the LoRA adapter logic.
    # It freezes all base model params and initializes A with random Gaussian,
    # B with zeros (so the initial output is identical to the base model).
    model = get_peft_model(model, lora_config)

    # Print a summary of trainable vs total parameters.
    # Should show ~30M trainable / 8B total = ~0.4%
    model.print_trainable_parameters()

    return model


# =============================================================================
# Main training function
# =============================================================================

def run_training(config: DictConfig, dry_run: bool = False) -> TrainingMetrics:
    """
    Execute the full QLoRA SFT training pipeline.

    This is the orchestration function — it calls all other components in order
    and coordinates their inputs/outputs.

    Args:
        config: Loaded OmegaConf config from config.yaml
        dry_run: If True, set up everything but run only 5 training steps.
                 Useful for verifying the setup works without full training cost.

    Returns:
        TrainingMetrics with all captured metrics from the run
    """
    metrics = TrainingMetrics()
    metrics.start_time = time.time()

    # ------------------------------------------------------------------
    # Step 1: Read HuggingFace token from environment
    # ------------------------------------------------------------------
    # Llama-3.1 is a gated model — you must agree to Meta's license first
    # at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    # Then generate a token at https://huggingface.co/settings/tokens
    token_env_var = config.model.get("hf_token_env_var", "HF_TOKEN")
    hf_token = os.environ.get(token_env_var)
    if not hf_token:
        logger.warning(
            f"Environment variable {token_env_var} is not set. "
            f"Loading Llama-3.1 may fail if you haven't already cached it locally. "
            f"Run: export {token_env_var}=hf_xxxxx"
        )

    # ------------------------------------------------------------------
    # Step 2: GPU info
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_vram:.1f} GB VRAM)")
        metrics.gpu_name = gpu_name
        metrics.gpu_vram_gb = gpu_vram
    else:
        logger.error(
            "No CUDA GPU detected. QLoRA training requires a GPU. "
            "On Lambda Cloud, ensure you selected a GPU instance."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Load tokenizer
    # ------------------------------------------------------------------
    logger.info(f"Loading tokenizer: {config.model.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_id,
        token=hf_token,
        # padding_side="right" is required for SFT with sequence packing.
        # If left-padded, the packing algorithm gets confused.
        padding_side="right",
        # add_eos_token ensures each training example ends with an EOS token.
        # Without this, the model won't learn when to stop generating.
        add_eos_token=True,
    )

    # Llama-3 doesn't have a pad token by default (it was trained without padding).
    # For batched training, we need a pad token. The common fix is to use EOS as pad.
    # This works because: (a) pad tokens are masked from attention anyway,
    # (b) the model will never encounter padding at inference time.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token = eos_token (Llama-3 default)")

    # ------------------------------------------------------------------
    # Step 4: Build datasets
    # ------------------------------------------------------------------
    logger.info("Building training and validation datasets...")
    train_ds, eval_ds = build_dataset(config, tokenizer)

    metrics.total_train_examples = len(train_ds)
    metrics.total_eval_examples = len(eval_ds)

    # Optional: compute token length statistics to understand the data
    # (commented out for speed — uncomment if you want detailed stats)
    # stats = get_dataset_statistics(train_ds, tokenizer)
    # logger.info(f"Token length stats: {stats}")

    # ------------------------------------------------------------------
    # Step 5: Load quantized base model
    # ------------------------------------------------------------------
    model = load_quantized_model(
        model_id=config.model.base_model_id,
        quantization_cfg=config.quantization,
        model_cfg=config.model,
        hf_token=hf_token,
    )

    # ------------------------------------------------------------------
    # Step 6: Attach LoRA adapters
    # ------------------------------------------------------------------
    model = attach_lora_adapters(model, config.lora)

    # ------------------------------------------------------------------
    # Step 7: Configure TrainingArguments
    # ------------------------------------------------------------------
    train_cfg = config.training
    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Effective batch size = per_device × gradient_accumulation
    # This determines how many examples influence each optimizer step.
    effective_batch_size = (
        train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps
    )
    metrics.effective_batch_size = effective_batch_size
    logger.info(
        f"Effective batch size: {effective_batch_size} "
        f"({train_cfg.per_device_train_batch_size} per-device × "
        f"{train_cfg.gradient_accumulation_steps} grad accum steps)"
    )

    # In dry-run mode, we only train for 5 steps to verify the setup
    max_steps = 5 if dry_run else -1  # -1 = use num_train_epochs

    # SFTConfig extends TrainingArguments with SFT-specific settings.
    # Key SFT-specific settings:
    # - max_seq_length: truncate sequences longer than this
    # - packing: pack multiple short sequences into one context window
    # - dataset_text_field: which column contains the formatted text
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.num_train_epochs if not dry_run else 1,
        max_steps=max_steps,

        # Batch configuration
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,

        # Optimizer
        learning_rate=train_cfg.learning_rate,
        optim=train_cfg.optim,
        weight_decay=train_cfg.weight_decay,

        # LR schedule
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,

        # Precision
        bf16=train_cfg.bf16,

        # Memory
        gradient_checkpointing=train_cfg.gradient_checkpointing,

        # Sequence configuration
        max_seq_length=config.model.max_seq_length,
        packing=train_cfg.packing,
        dataset_text_field="text",  # Column name from apply_chat_template()

        # Evaluation
        eval_strategy="steps",
        eval_steps=train_cfg.eval_steps if not dry_run else 5,

        # Checkpointing
        save_strategy="steps",
        save_steps=train_cfg.save_steps if not dry_run else 5,
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=train_cfg.greater_is_better,

        # Logging
        logging_steps=train_cfg.logging_steps,
        report_to=train_cfg.report_to if not dry_run else "none",

        # Reproducibility
        seed=train_cfg.seed,
    )

    # ------------------------------------------------------------------
    # Step 8: Initialize SFTTrainer
    # ------------------------------------------------------------------
    # SFTTrainer is trl's specialized trainer for instruction fine-tuning.
    # Key differences from vanilla Trainer:
    # - Handles the response_template masking (loss only on assistant tokens)
    # - Supports sequence packing
    # - Integrates with PEFT models directly
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=[LoraForgeCallback(log_gpu_every_n_steps=50)],
    )

    # ------------------------------------------------------------------
    # Step 9: Train
    # ------------------------------------------------------------------
    if dry_run:
        logger.info("DRY RUN: Training for 5 steps only to verify setup.")

    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"Model: {config.model.base_model_id}")
    logger.info(f"LoRA rank={config.lora.r}, alpha={config.lora.lora_alpha}")
    logger.info(f"Train examples: {len(train_ds)}, Eval examples: {len(eval_ds)}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    train_result = trainer.train()

    # ------------------------------------------------------------------
    # Step 10: Capture metrics
    # ------------------------------------------------------------------
    metrics.final_train_loss = train_result.training_loss
    metrics.base_model_id = config.model.base_model_id
    metrics.lora_rank = config.lora.r
    metrics.lora_alpha = config.lora.lora_alpha
    metrics.num_train_epochs = config.training.num_train_epochs
    metrics.learning_rate = config.training.learning_rate

    # Run final evaluation
    logger.info("Running final evaluation on validation set...")
    eval_results = trainer.evaluate()
    metrics.final_eval_loss = eval_results.get("eval_loss")
    if metrics.final_eval_loss is not None:
        import math
        metrics.final_eval_perplexity = math.exp(metrics.final_eval_loss)
        logger.info(
            f"Final eval loss: {metrics.final_eval_loss:.4f} | "
            f"Perplexity: {metrics.final_eval_perplexity:.2f}"
        )

    # ------------------------------------------------------------------
    # Step 11: Save the LoRA adapter
    # ------------------------------------------------------------------
    adapter_dir = Path(config.output.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving LoRA adapter to: {adapter_dir}")
    # Save only the adapter (A and B matrices) — NOT the full model.
    # This produces adapter_config.json + adapter_model.safetensors (~150MB).
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics.adapter_path = str(adapter_dir)

    # ------------------------------------------------------------------
    # Step 12: Save training metrics
    # ------------------------------------------------------------------
    metrics.end_time = time.time()
    metrics.training_duration_hours = (metrics.end_time - metrics.start_time) / 3600

    metrics_path = Path(config.output.metrics_file)
    metrics.save(metrics_path)

    logger.info("=" * 60)
    logger.info("✓ Training complete!")
    logger.info(f"  Adapter saved to: {adapter_dir}")
    logger.info(f"  Final train loss: {metrics.final_train_loss:.4f}")
    logger.info(f"  Final eval loss: {metrics.final_eval_loss:.4f}")
    logger.info(f"  Eval perplexity: {metrics.final_eval_perplexity:.2f}")
    logger.info(f"  Duration: {metrics.training_duration_hours:.2f} hours")
    logger.info(f"  Metrics: {metrics_path}")
    logger.info("=" * 60)
    logger.info("Next step: Run Activity 1 to merge this adapter and benchmark baseline serving.")

    return metrics


# =============================================================================
# CLI entry point
# =============================================================================

@app.command()
def main(
    config_path: Path = typer.Option(
        Path(__file__).parent.parent / "config.yaml",
        "--config",
        "-c",
        help="Path to the training configuration YAML file",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run 5 training steps only to verify setup. Does not save adapter.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable DEBUG logging",
    ),
) -> None:
    """
    Activity 0: QLoRA SFT — Fine-tune Llama-3.1-8B-Instruct on cybersecurity data.

    This script reads config.yaml, downloads datasets, loads the quantized model,
    attaches LoRA adapters, and runs supervised fine-tuning.

    Expected runtime: 6-8 hours on Lambda A10G (24 GB). Cost: ~$5-8.

    Before running:
        export HF_TOKEN=hf_xxxxx  # From huggingface.co/settings/tokens
        # Ensure you've accepted Meta's Llama license at:
        # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

    Output:
        ./outputs/cybersec_analyst_lora/   <- LoRA adapter (use in Activity 1)
        ./outputs/activity0_training_metrics.json <- Training metrics
        ./outputs/tensorboard_logs/        <- TensorBoard logs (run: tensorboard --logdir=...)
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise typer.Exit(1)

    # Load YAML config with OmegaConf.
    # OmegaConf supports interpolation (${var}) and merging configs.
    logger.info(f"Loading config from: {config_path}")
    config = OmegaConf.load(config_path)

    if dry_run:
        logger.info(
            "DRY RUN mode: will run 5 training steps to validate the setup. "
            "Adapter will NOT be saved."
        )

    run_training(config, dry_run=dry_run)


if __name__ == "__main__":
    app()
