"""
activity2_quantization/quantizer/modelopt_quantizer.py
=======================================================
QuantizationProfiler — applies PTQ to the merged FP16 model using nvidia-modelopt.

WHAT THIS MODULE DOES
---------------------
For each requested quantization format (FP8, AWQ INT4, SmoothQuant INT8):
  1. Load the FP16 merged model from ./outputs/cybersec_analyst_merged
  2. Run nvidia-modelopt's mtq.quantize() with the format-specific quant_config
     and the 512-sample calibration forward loop
  3. Measure VRAM footprint and calibration duration
  4. Save the quantized model checkpoint to ./outputs/quantized/{format}/
  5. Run a quick perplexity evaluation on 100 examples to detect accuracy regressions

WHY NVIDIA MODELOPT (vs alternatives)
--------------------------------------
  - AutoAWQ:   Only does AWQ. No FP8, no SmoothQuant, no TRT-LLM export.
  - AutoGPTQ:  Only does GPTQ INT4. Slower calibration, no per-channel activation scaling.
  - bitsandbytes: Load-time quantization only. No calibration, ~1-3% accuracy loss.
  - ModelOpt:  Single API for AWQ, SmoothQuant, FP8, GPTQ, and direct TRT-LLM export.
               Used by NVIDIA internally for production deployments.

THE MTQ.QUANTIZE() CALL
------------------------
  mtq.quantize(model, quant_config, forward_loop=calibration_dataloader)

  Under the hood, this:
    1. Registers QuantDescriptors on every Linear layer (hooks that intercept forward passes)
    2. Runs the model on calibration data — hooks collect per-channel activation stats
    3. Computes scale factors from the collected statistics:
       - FP8: max(|x|) / max_representable_fp8
       - AWQ: per-group weight scales derived from activation magnitude ranking
       - SmoothQuant: per-channel s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
    4. Bakes the scale factors into QuantLinear modules (replaces Linear with QuantLinear)
    5. The model weights are reparameterized: W_fp16 → (W_quantized + scale_fp16_per_group)

  This is a *one-way lossy transformation*. After quantization, you cannot recover
  the original FP16 weights from the quantized checkpoint.

Run directly:
  python -m activity2_quantization.quantizer.modelopt_quantizer --format fp8
  python -m activity2_quantization.quantizer.modelopt_quantizer --format awq_int4
  python -m activity2_quantization.quantizer.modelopt_quantizer --format smoothquant_int8
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml

logger = logging.getLogger(__name__)

# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class QuantizationResult:
    """Results from one quantization run."""
    format_name: str
    success: bool
    output_path: Optional[str]
    vram_before_gb: float
    vram_after_gb: float
    vram_delta_gb: float            # Negative = memory savings
    calibration_duration_seconds: float
    perplexity: Optional[float]     # On 100-example quick eval
    perplexity_delta_pct: Optional[float]  # vs FP16 baseline
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ── QuantizationProfiler ──────────────────────────────────────────────────────

class QuantizationProfiler:
    """
    Applies post-training quantization to the FP16 merged model using nvidia-modelopt.

    One instance per quantization format. Manages the full lifecycle:
    load → calibrate → quantize → save → quick eval.

    Usage:
        config = yaml.safe_load(open("config.yaml"))
        calib_data = CalibrationDatasetBuilder(config).build()
        profiler = QuantizationProfiler(config)
        result = profiler.quantize("fp8", calib_data)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.merged_model_path = Path(
            config.get("merged_model_path", "./outputs/cybersec_analyst_merged")
        )
        self.output_dir = Path(
            config.get("quantized_output_dir", "./outputs/quantized")
        )
        self.base_model_id = config.get("base_model_id", "meta-llama/Llama-3.1-8B-Instruct")
        self.quant_formats_config: dict = config.get("quantization", {}).get("formats", {})
        self.eval_config: dict = config.get("evaluation", {})

    # ── Public API ─────────────────────────────────────────────────────────────

    def quantize(
        self,
        format_name: str,
        calib_dataset,  # CalibrationDataset from calibration.py
        run_quick_eval: bool = True,
        baseline_perplexity: Optional[float] = None,
    ) -> QuantizationResult:
        """
        Apply quantization for a single format and save the checkpoint.

        Args:
            format_name:          One of "fp8", "awq_int4", "smoothquant_int8".
            calib_dataset:        CalibrationDataset built by CalibrationDatasetBuilder.
            run_quick_eval:       Whether to run a 100-example perplexity eval after quant.
            baseline_perplexity:  FP16 perplexity for delta computation (from Activity 1).

        Returns:
            QuantizationResult with VRAM stats, timing, perplexity delta, and output path.
        """
        fmt_config = self.quant_formats_config.get(format_name)
        if fmt_config is None:
            return QuantizationResult(
                format_name=format_name, success=False, output_path=None,
                vram_before_gb=0, vram_after_gb=0, vram_delta_gb=0,
                calibration_duration_seconds=0, perplexity=None,
                perplexity_delta_pct=None,
                error=f"Unknown format: {format_name}. "
                      f"Valid: {list(self.quant_formats_config.keys())}",
            )

        if not fmt_config.get("enabled", True):
            logger.info(f"Format {format_name} is disabled in config. Skipping.")
            return QuantizationResult(
                format_name=format_name, success=False, output_path=None,
                vram_before_gb=0, vram_after_gb=0, vram_delta_gb=0,
                calibration_duration_seconds=0, perplexity=None,
                perplexity_delta_pct=None, error="disabled",
            )

        output_path = self.output_dir / format_name
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"QUANTIZING: {format_name}")
        logger.info(f"  Source model: {self.merged_model_path}")
        logger.info(f"  Output path:  {output_path}")
        logger.info("=" * 60)

        try:
            result = self._run_quantization(
                format_name, fmt_config, calib_dataset, output_path,
                run_quick_eval, baseline_perplexity,
            )
            logger.info(
                f"✓ {format_name} quantization complete — "
                f"VRAM: {result.vram_before_gb:.1f}→{result.vram_after_gb:.1f} GB "
                f"(Δ{result.vram_delta_gb:+.1f} GB) | "
                f"Time: {result.calibration_duration_seconds:.0f}s"
            )
            return result

        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                f"OOM during {format_name} quantization. "
                f"Try reducing calibration batch size or max_seq_length. Error: {e}"
            )
            torch.cuda.empty_cache()
            return QuantizationResult(
                format_name=format_name, success=False, output_path=None,
                vram_before_gb=0, vram_after_gb=0, vram_delta_gb=0,
                calibration_duration_seconds=0, perplexity=None,
                perplexity_delta_pct=None, error=f"OOM: {e}",
            )
        except Exception as e:
            logger.error(f"Quantization failed for {format_name}: {e}", exc_info=True)
            return QuantizationResult(
                format_name=format_name, success=False, output_path=None,
                vram_before_gb=0, vram_after_gb=0, vram_delta_gb=0,
                calibration_duration_seconds=0, perplexity=None,
                perplexity_delta_pct=None, error=str(e),
            )

    # ── Private Implementation ─────────────────────────────────────────────────

    def _run_quantization(
        self,
        format_name: str,
        fmt_config: dict,
        calib_dataset,
        output_path: Path,
        run_quick_eval: bool,
        baseline_perplexity: Optional[float],
    ) -> QuantizationResult:
        """Core quantization logic."""
        import modelopt.torch.quantization as mtq
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ── Load model in FP16 ─────────────────────────────────────────────────
        logger.info(f"Loading model from {self.merged_model_path}...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        vram_before_gb = torch.cuda.memory_allocated() / 1024**3

        model = AutoModelForCausalLM.from_pretrained(
            str(self.merged_model_path),
            torch_dtype=torch.float16,
            device_map="auto",       # Spread across available GPUs (or single GPU)
            trust_remote_code=False,
        )
        model.eval()

        vram_loaded_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(
            f"Model loaded: {vram_loaded_gb:.2f} GB VRAM "
            f"({vram_loaded_gb - vram_before_gb:+.2f} GB delta)"
        )

        # ── Build quantization config ──────────────────────────────────────────
        quant_config = self._build_modelopt_quant_config(format_name, fmt_config)

        # ── Build calibration forward loop ─────────────────────────────────────
        # ModelOpt expects a callable that accepts the model and calls model(batch)
        # for each calibration batch. We wrap calib_dataset.to_dataloader().
        device = next(model.parameters()).device

        def calibration_forward_loop(model):
            """Forward loop passed to mtq.quantize(). Runs calibration batches."""
            logger.info(f"Running {len(calib_dataset)} calibration forward passes...")
            for batch_idx, batch in enumerate(calib_dataset.to_dataloader(batch_size=1)):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    model(**batch)
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"  Calibration progress: {batch_idx + 1}/{len(calib_dataset)}")

        # ── Apply quantization via ModelOpt ────────────────────────────────────
        # This is the core call: ModelOpt inserts QuantDescriptor hooks, runs the
        # forward loop to collect activation stats, then bakes scale factors in.
        logger.info("Running mtq.quantize() — collecting activation statistics...")
        calib_start = time.perf_counter()

        mtq.quantize(model, quant_config, forward_loop=calibration_forward_loop)

        calib_duration = time.perf_counter() - calib_start
        logger.info(f"mtq.quantize() complete in {calib_duration:.0f}s")

        # ── Measure post-quantization VRAM ─────────────────────────────────────
        vram_after_gb = torch.cuda.memory_allocated() / 1024**3
        vram_delta_gb = vram_after_gb - vram_loaded_gb
        logger.info(
            f"Post-quantization VRAM: {vram_after_gb:.2f} GB "
            f"(model delta: {vram_delta_gb:+.2f} GB)"
        )

        # ── Save quantized model ───────────────────────────────────────────────
        logger.info(f"Saving quantized model to {output_path}...")
        save_start = time.perf_counter()

        # Save model state dict with quantization metadata
        mtq.save(model, str(output_path))

        # Also save tokenizer alongside model (needed by vLLM to serve the model)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        tokenizer.save_pretrained(str(output_path))

        save_duration = time.perf_counter() - save_start
        logger.info(f"Saved in {save_duration:.0f}s")

        # ── Quick perplexity eval ──────────────────────────────────────────────
        perplexity = None
        perplexity_delta_pct = None

        if run_quick_eval:
            perplexity = self._quick_perplexity_eval(model, device)
            if baseline_perplexity and perplexity:
                perplexity_delta_pct = (
                    (perplexity - baseline_perplexity) / baseline_perplexity * 100
                )
                logger.info(
                    f"Perplexity: {perplexity:.4f} "
                    f"(vs FP16 baseline {baseline_perplexity:.4f}, "
                    f"Δ{perplexity_delta_pct:+.2f}%)"
                )

        # ── Cleanup ────────────────────────────────────────────────────────────
        del model
        torch.cuda.empty_cache()

        return QuantizationResult(
            format_name=format_name,
            success=True,
            output_path=str(output_path),
            vram_before_gb=vram_loaded_gb,
            vram_after_gb=vram_after_gb,
            vram_delta_gb=vram_delta_gb,
            calibration_duration_seconds=calib_duration,
            perplexity=perplexity,
            perplexity_delta_pct=perplexity_delta_pct,
            metadata={
                "format_config": fmt_config,
                "save_duration_seconds": save_duration,
                "output_size_gb": self._dir_size_gb(output_path),
            },
        )

    def _build_modelopt_quant_config(self, format_name: str, fmt_config: dict) -> dict:
        """
        Build the quant_config dict that mtq.quantize() expects.

        Format-specific configs are defined in config.yaml under
        quantization.formats.{format}.modelopt_config.
        This method augments them with any derived fields.
        """
        modelopt_cfg = fmt_config.get("modelopt_config", {})
        quant_cfg = modelopt_cfg.get("quant_cfg", {})

        if format_name == "smoothquant_int8":
            # SmoothQuant requires an additional smoothing pass before quantization.
            # alpha controls how much difficulty migrates to weights vs activations.
            alpha = fmt_config.get("alpha", 0.5)
            return {
                "quant_cfg": quant_cfg,
                "algorithm": "smoothquant",
                "smoothquant_args": {"alpha": alpha},
            }
        elif format_name == "awq_int4":
            group_size = fmt_config.get("group_size", 128)
            return {
                "quant_cfg": quant_cfg,
                "algorithm": "awq",
                "awq_args": {"group_size": group_size},
            }
        else:
            # FP8 and any other format: just pass quant_cfg directly
            return {"quant_cfg": quant_cfg}

    def _quick_perplexity_eval(self, model, device, num_samples: int = 100) -> Optional[float]:
        """
        Run a quick 100-sample perplexity evaluation on the quantized model.

        This is a fast sanity check — not a comprehensive eval.
        We use a fixed set of cybersecurity prompts for consistency.
        """
        from datasets import load_dataset

        try:
            eval_cfg = self.eval_config
            dataset_id = eval_cfg.get("test_dataset", "HuggingFaceH4/ultrachat_200k")
            split = eval_cfg.get("test_split_name", "test_sft")
            max_len = eval_cfg.get("max_seq_length", 1024)

            tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            logger.info(f"Quick perplexity eval ({num_samples} samples)...")
            dataset = load_dataset(dataset_id, split=split, streaming=True)

            total_loss = 0.0
            total_tokens = 0
            samples_processed = 0

            for example in dataset:
                if samples_processed >= num_samples:
                    break
                messages = example.get("messages", [])
                if not messages:
                    continue
                text = " ".join(m.get("content", "") for m in messages[:2])
                if not text.strip():
                    continue

                enc = tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True)
                input_ids = enc["input_ids"].to(device)
                if input_ids.shape[1] < 4:
                    continue

                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss.item()

                total_loss += loss * (input_ids.shape[1] - 1)
                total_tokens += input_ids.shape[1] - 1
                samples_processed += 1

            if total_tokens == 0:
                return None

            avg_loss = total_loss / total_tokens
            perplexity = float(torch.exp(torch.tensor(avg_loss)))
            return perplexity

        except Exception as e:
            logger.warning(f"Quick perplexity eval failed: {e}")
            return None

    @staticmethod
    def _dir_size_gb(path: Path) -> float:
        """Return total size of a directory in GB."""
        total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total_bytes / 1024**3


# Lazy import for AutoTokenizer (needed in _quick_perplexity_eval method body)
try:
    from transformers import AutoTokenizer
except ImportError:
    pass


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_quantization(
    config: dict,
    format_name: str,
    calib_dataset,
    args=None,
    baseline_perplexity: Optional[float] = None,
) -> QuantizationResult:
    """Quantize the model in the specified format. Called from main.py."""
    profiler = QuantizationProfiler(config)
    return profiler.quantize(
        format_name=format_name,
        calib_dataset=calib_dataset,
        run_quick_eval=True,
        baseline_perplexity=baseline_perplexity,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run quantization for one format")
    parser.add_argument("--config", default="./activity2_quantization/config.yaml")
    parser.add_argument(
        "--format", required=True,
        choices=["fp8", "awq_int4", "smoothquant_int8"],
        help="Quantization format to apply",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from activity2_quantization.quantizer.calibration import CalibrationDatasetBuilder
    calib_data = CalibrationDatasetBuilder(config).build()
    result = run_quantization(config, args.format, calib_data, args)

    print(f"\nResult: success={result.success}")
    if result.success:
        print(f"  VRAM: {result.vram_before_gb:.2f} → {result.vram_after_gb:.2f} GB")
        print(f"  Perplexity: {result.perplexity}")
        print(f"  Output: {result.output_path}")
    else:
        print(f"  Error: {result.error}")
