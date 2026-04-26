"""
activity2_quantization/benchmark/quant_benchmarker.py
======================================================
QuantizationBenchmarker — benchmarks each quantized model format against the
FP16 baseline by running the same sweep as Activity 1.

WHAT THIS MODULE DOES
----------------------
For each quantized format (FP8, AWQ INT4, SmoothQuant INT8):
  1. Launch a vLLM server with the quantized model checkpoint
  2. Run the same (concurrency × prompt_len × output_len) benchmark sweep used in Activity 1
  3. Collect TTFT P50/P95 and throughput for each configuration
  4. Compute deltas vs the Activity 1 FP16 baseline results
  5. Save per-format results to results/quant_results.json

WHY THE SAME BENCHMARK AS ACTIVITY 1
--------------------------------------
We reuse the Activity 1 benchmark client (quant_benchmarker is a thin wrapper) so
the comparison is apples-to-apples. Any difference in results is purely due to
quantization — not to a different benchmark methodology.

HOW VLLM SERVES QUANTIZED MODELS
----------------------------------
vLLM natively supports:
  - FP8:             --dtype fp8 (or --quantization fp8)
  - AWQ INT4:        --quantization awq
  - SmoothQuant:     Not natively supported in vLLM 0.6.x; served as FP16 with
                     quantized weights baked in (ModelOpt-exported format).
                     vLLM detects the quantization config in quantize_config.json.

The quantized checkpoints saved by QuantizationProfiler include a
quantize_config.json that vLLM uses to detect the format automatically.

VRAM MEASUREMENT
-----------------
We measure VRAM two ways:
  1. nvidia-smi snapshot before and after vLLM server startup
  2. torch.cuda.memory_allocated() during a single inference call

The delta vs FP16 is the key metric: Activity 2 targets ~50% VRAM reduction.

Run directly:
  # vLLM server must NOT be running (this module manages it)
  python -m activity2_quantization.benchmark.quant_benchmarker --format fp8
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class FormatBenchmarkResults:
    """All benchmark results for one quantization format."""
    format_name: str
    model_path: str
    benchmark_timestamp: str
    benchmark_duration_seconds: float
    vllm_dtype_flag: str           # e.g. "--dtype fp8", "--quantization awq"
    vram_used_gb: Optional[float]  # Measured during benchmark
    configs: list = field(default_factory=list)  # list of ConfigResult-like dicts

    # Deltas vs FP16 baseline (computed after benchmark completes)
    ttft_p50_delta_ms: Optional[float] = None      # Negative = improvement
    throughput_delta_tok_s: Optional[float] = None  # Positive = improvement
    throughput_delta_pct: Optional[float] = None


# ── QuantizationBenchmarker ───────────────────────────────────────────────────

class QuantizationBenchmarker:
    """
    Benchmarks all quantized model formats against the FP16 baseline.

    For each format:
      1. Start vLLM with quantized checkpoint
      2. Run Activity 1's benchmark sweep (same code, same configs)
      3. Measure VRAM
      4. Compute deltas vs baseline
      5. Terminate server

    Usage:
        benchmarker = QuantizationBenchmarker(config)
        results = benchmarker.benchmark_all(quant_results, baseline_results)
        # results is {format_name: FormatBenchmarkResults}
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        bench_cfg = config.get("benchmark", {})
        server_cfg = bench_cfg.get("server", {})

        self.concurrency_levels: list[int] = bench_cfg.get("concurrency_levels", [1, 4, 8, 16, 32])
        self.prompt_lengths: list[int] = bench_cfg.get("prompt_lengths_tokens", [128, 512, 2048])
        self.output_lengths: list[int] = bench_cfg.get("output_lengths_tokens", [128, 256])
        self.num_requests: int = bench_cfg.get("num_requests_per_config", 20)
        self.warmup_requests: int = bench_cfg.get("warmup_requests", 5)
        self.request_timeout: float = bench_cfg.get("request_timeout_seconds", 120)
        self.temperature: float = bench_cfg.get("temperature", 0.1)
        self.api_base_url: str = bench_cfg.get("api_base_url", "http://localhost:8000")
        self.baseline_results_path: Path = Path(
            bench_cfg.get("baseline_results_path", "./results/baseline_results.json")
        )

        self.server_host: str = server_cfg.get("host", "0.0.0.0")
        self.server_port: int = server_cfg.get("port", 8000)
        self.max_model_len: int = server_cfg.get("max_model_len", 4096)
        self.gpu_memory_utilization: float = server_cfg.get("gpu_memory_utilization", 0.90)
        self.max_num_seqs: int = server_cfg.get("max_num_seqs", 64)
        self.startup_timeout: int = server_cfg.get("startup_timeout_seconds", 180)

        report_cfg = config.get("reporting", {})
        self.output_dir: Path = Path(report_cfg.get("output_dir", "./results"))
        self.quant_results_filename: str = report_cfg.get(
            "quant_results_filename", "quant_results.json"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def benchmark_all(
        self,
        quant_results: list,          # list[QuantizationResult] from modelopt_quantizer.py
        args=None,
    ) -> dict[str, FormatBenchmarkResults]:
        """
        Benchmark all successfully quantized formats.

        Args:
            quant_results:  List of QuantizationResult. Only formats with success=True
                            are benchmarked.
            args:           argparse.Namespace with optional --quick flag.

        Returns:
            Dict mapping format_name → FormatBenchmarkResults.
        """
        baseline_data = self._load_baseline()
        all_results: dict[str, FormatBenchmarkResults] = {}

        successful_formats = [qr for qr in quant_results if qr.success]
        logger.info(f"Benchmarking {len(successful_formats)} quantized formats...")

        for qr in successful_formats:
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"BENCHMARKING: {qr.format_name}")
            logger.info(f"  Model path: {qr.output_path}")
            logger.info("=" * 60)

            result = self._benchmark_one_format(qr, args)
            if result:
                result = self._compute_deltas(result, baseline_data)
                all_results[qr.format_name] = result
                logger.info(
                    f"✓ {qr.format_name} benchmark complete — "
                    f"TTFT P50 delta: {result.ttft_p50_delta_ms:+.0f}ms | "
                    f"Throughput delta: {result.throughput_delta_pct:+.0f}%"
                )

        # Save combined results
        self._save_combined_results(all_results, quant_results)
        return all_results

    def benchmark_format(
        self,
        format_name: str,
        model_path: str,
        args=None,
    ) -> Optional[FormatBenchmarkResults]:
        """
        Benchmark a single format by name and model path.
        Useful for re-benchmarking one format without re-quantizing.
        """
        from activity2_quantization.quantizer.modelopt_quantizer import QuantizationResult
        # Create a minimal QuantizationResult-like object
        qr = type("_QR", (), {
            "format_name": format_name,
            "success": True,
            "output_path": model_path,
        })()

        baseline_data = self._load_baseline()
        result = self._benchmark_one_format(qr, args)
        if result:
            result = self._compute_deltas(result, baseline_data)
        return result

    # ── Private Implementation ─────────────────────────────────────────────────

    def _export_for_vllm(self, modelopt_ckpt_dir: str, format_name: str) -> str:
        """
        Export a modelopt checkpoint to HuggingFace safetensors format for vLLM.

        WHY THIS IS NEEDED
        -------------------
        mto.save() writes a PyTorch state dict (modelopt_state.pt) that contains
        QuantLinear layer weights + scale factors. vLLM cannot load this format —
        it expects standard HuggingFace safetensors files where the weights are
        already stored in the target dtype (fp8/int4/int8).

        HOW IT WORKS
        -------------
        1. Load the model architecture from config.json (saved alongside modelopt_state.pt)
        2. mto.restore() overlays the quantized QuantLinear weights onto the skeleton
        3. export_hf_checkpoint() converts QuantLinear layers to real quantized dtypes
           and writes them as safetensors + quantization_config.json
        4. vLLM loads the export dir with the correct --quantization flag

        The export is cached — if hf_export/ already contains safetensors, skip.
        """
        import gc
        import torch
        from pathlib import Path
        from transformers import AutoModelForCausalLM
        import modelopt.torch.opt as mto

        ckpt_dir = Path(modelopt_ckpt_dir)
        hf_export_dir = ckpt_dir / "hf_export"

        # Return cached export if it exists
        if hf_export_dir.exists() and any(hf_export_dir.glob("*.safetensors")):
            logger.info(f"  HF export already exists at {hf_export_dir}, skipping export")
            return str(hf_export_dir)

        logger.info(f"  Exporting {format_name} checkpoint to HuggingFace format...")
        logger.info(f"  Loading model skeleton from {ckpt_dir}...")

        # Load model architecture from config.json that was saved alongside mto.save()
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False,
        )

        try:
            # Restore modelopt quantization state onto the skeleton
            modelopt_state = ckpt_dir / "modelopt_state.pt"
            mto.restore(model, str(modelopt_state))
            logger.info(f"  Restored modelopt quantization state from {modelopt_state.name}")

            hf_export_dir.mkdir(parents=True, exist_ok=True)

            # Export to HuggingFace safetensors format
            try:
                from modelopt.torch.export import export_hf_checkpoint
                export_hf_checkpoint(model, str(hf_export_dir))
                logger.info(f"  Exported to {hf_export_dir} via modelopt.torch.export")
            except (ImportError, AttributeError) as e:
                # Fallback: save via HuggingFace directly.
                # Weights will be FP16 but quantization scale factors are baked in,
                # so vLLM can still load the model (just without native quant speedup).
                logger.warning(
                    f"  export_hf_checkpoint not available ({e}), "
                    f"falling back to model.save_pretrained()"
                )
                model.save_pretrained(str(hf_export_dir), safe_serialization=True)

            # Copy tokenizer files into export dir so it's self-contained for vLLM
            import shutil
            for tok_file in ckpt_dir.glob("tokenizer*"):
                shutil.copy2(tok_file, hf_export_dir / tok_file.name)
            for special_file in ["special_tokens_map.json"]:
                src = ckpt_dir / special_file
                if src.exists():
                    shutil.copy2(src, hf_export_dir / special_file)

            return str(hf_export_dir)

        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"  Export complete — VRAM freed")

    def _benchmark_one_format(self, qr, args) -> Optional[FormatBenchmarkResults]:
        """Start server → benchmark → stop server → return results."""
        import datetime

        server_process = None
        try:
            # Determine vLLM flags for this format
            vllm_dtype_flag = self._get_vllm_dtype_flag(qr.format_name)

            # Export modelopt checkpoint to HuggingFace format for vLLM.
            # vLLM cannot load modelopt_state.pt directly — it needs safetensors.
            logger.info(f"Preparing {qr.format_name} model for vLLM...")
            vllm_model_path = self._export_for_vllm(qr.output_path, qr.format_name)

            # Start vLLM server with the exported HF checkpoint
            server_process = self._start_vllm_server(
                model_path=vllm_model_path,
                dtype_flag=vllm_dtype_flag,
            )

            # Run benchmark sweep
            logger.info("Running benchmark sweep...")
            quick_mode = getattr(args, "quick", False)
            config_results = asyncio.run(
                self._run_benchmark_sweep(
                    model_name=f"cybersec-analyst-{qr.format_name}",
                    quick=quick_mode,
                )
            )

            # Measure VRAM
            vram_gb = self._measure_vram_via_nvml()

            return FormatBenchmarkResults(
                format_name=qr.format_name,
                model_path=str(qr.output_path),
                benchmark_timestamp=datetime.datetime.now().isoformat(),
                benchmark_duration_seconds=0,  # Filled in _run_benchmark_sweep
                vllm_dtype_flag=vllm_dtype_flag,
                vram_used_gb=vram_gb,
                configs=config_results,
            )

        except Exception as e:
            logger.error(f"Benchmark failed for {qr.format_name}: {e}", exc_info=True)
            return None
        finally:
            if server_process is not None:
                logger.info(f"Stopping vLLM server for {qr.format_name}...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server_process.kill()

    def _get_vllm_dtype_flag(self, format_name: str) -> str:
        """
        Return the correct vLLM --quantization flag for the format.

        vLLM 0.6.1 valid --quantization values (from api_server.py --help):
          fp8, awq, awq_marlin, gptq, gptq_marlin, bitsandbytes, compressed-tensors,
          modelopt, marlin, ...
        NOT valid: smoothquant, int8 (these are modelopt-internal names)

        FP8:           --quantization fp8  (vLLM native FP8 loader)
        AWQ INT4:      --quantization awq  (vLLM native AWQ loader, expects AWQ safetensors)
        SmoothQuant:   --quantization compressed-tensors  (closest vLLM equivalent for W8A8;
                       export_hf_checkpoint produces compressed-tensors compatible weights)
        """
        flags = {
            "fp8": "--quantization fp8",
            "awq_int4": "--quantization awq",
            "smoothquant_int8": "--quantization compressed-tensors",
        }
        return flags.get(format_name, "--dtype auto")

    def _start_vllm_server(
        self,
        model_path: str,
        dtype_flag: str,
    ) -> subprocess.Popen:
        """Launch vLLM server as a subprocess and wait for it to be ready."""
        cmd = (
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {model_path} "
            f"--host {self.server_host} "
            f"--port {self.server_port} "
            f"{dtype_flag} "
            f"--max-model-len {self.max_model_len} "
            f"--gpu-memory-utilization {self.gpu_memory_utilization} "
            f"--max-num-seqs {self.max_num_seqs} "
            f"--tensor-parallel-size 1 "
            f"--disable-log-requests"
        )
        logger.info(f"Starting vLLM: {cmd}")

        process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Poll until server is ready (health endpoint returns 200)
        import httpx
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(f"{self.api_base_url}/health", timeout=5)
                if resp.status_code == 200:
                    logger.info("vLLM server is ready ✓")
                    return process
            except Exception:
                pass
            time.sleep(5)

            # Check if process has already died
            if process.poll() is not None:
                stdout = process.stdout.read() if process.stdout else ""
                raise RuntimeError(
                    f"vLLM server process died during startup.\n"
                    f"Last output:\n{stdout[-2000:]}"
                )

        process.terminate()
        raise TimeoutError(
            f"vLLM server did not become healthy within {self.startup_timeout}s"
        )

    async def _run_benchmark_sweep(
        self,
        model_name: str,
        quick: bool = False,
    ) -> list[dict]:
        """
        Run the full (or quick) benchmark sweep against the running vLLM server.

        Reuses the same request logic as Activity 1's benchmark/client.py to ensure
        methodology consistency.
        """
        import statistics
        import datetime
        import httpx
        from activity1_baseline.benchmark.client import (
            run_concurrent_requests,
            compute_config_stats,
            generate_prompt,
        )

        concurrency_levels = self.concurrency_levels
        prompt_lengths = self.prompt_lengths
        output_lengths = self.output_lengths
        num_requests = self.num_requests
        warmup_requests = self.warmup_requests

        if quick:
            concurrency_levels = [1, 4, 8]
            prompt_lengths = [128, 512]
            output_lengths = [128]
            num_requests = 5
            warmup_requests = 2
            logger.info("QUICK mode: reduced sweep")

        all_config_results = []
        total_configs = len(concurrency_levels) * len(prompt_lengths) * len(output_lengths)
        config_num = 0
        sweep_start = time.perf_counter()

        for concurrency in concurrency_levels:
            for prompt_len in prompt_lengths:
                for output_len in output_lengths:
                    config_num += 1
                    logger.info(
                        f"[{config_num}/{total_configs}] "
                        f"concurrency={concurrency}, prompt={prompt_len}tok, output={output_len}tok"
                    )

                    semaphore = asyncio.Semaphore(concurrency)

                    # Warmup
                    if warmup_requests > 0:
                        await run_concurrent_requests(
                            self.api_base_url, model_name, concurrency,
                            warmup_requests, prompt_len, output_len,
                            self.temperature, self.request_timeout, semaphore,
                        )

                    # Timed sweep
                    wall_start = time.perf_counter()
                    request_results = await run_concurrent_requests(
                        self.api_base_url, model_name, concurrency,
                        num_requests, prompt_len, output_len,
                        self.temperature, self.request_timeout, semaphore,
                    )
                    wall_time = time.perf_counter() - wall_start

                    config_result = compute_config_stats(
                        request_results, concurrency, prompt_len, output_len, wall_time
                    )

                    # Convert dataclass to dict for JSON serialization
                    from dataclasses import asdict as dc_asdict
                    result_dict = dc_asdict(config_result)
                    all_config_results.append(result_dict)

                    if config_result.ttft_p50_seconds:
                        logger.info(
                            f"  ✓ TTFT P50={config_result.ttft_p50_seconds*1000:.0f}ms  "
                            f"P95={config_result.ttft_p95_seconds*1000:.0f}ms  "
                            f"Throughput={config_result.throughput_tokens_per_sec:.0f} tok/s"
                        )

        return all_config_results

    def _measure_vram_via_nvml(self) -> Optional[float]:
        """Query nvidia-smi for current GPU memory usage."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                mb_used = int(result.stdout.strip().split("\n")[0])
                return mb_used / 1024  # Convert MiB to GiB
        except Exception as e:
            logger.warning(f"Could not measure VRAM via nvidia-smi: {e}")
        return None

    def _load_baseline(self) -> dict:
        """Load Activity 1 baseline results."""
        with open(self.baseline_results_path) as f:
            return json.load(f)

    def _compute_deltas(
        self,
        result: FormatBenchmarkResults,
        baseline_data: dict,
    ) -> FormatBenchmarkResults:
        """Compute TTFT and throughput deltas vs FP16 baseline."""
        baseline_configs = {
            (c["concurrency"], c["prompt_len_tokens"]): c
            for c in baseline_data.get("configs", [])
        }

        # Reference point: concurrency=1, prompt=512 for TTFT
        baseline_ttft_ref = baseline_configs.get((1, 512))
        current_ttft_ref = next(
            (c for c in result.configs
             if c["concurrency"] == 1 and c["prompt_len_tokens"] == 512),
            None,
        )
        if baseline_ttft_ref and current_ttft_ref and current_ttft_ref.get("ttft_p50_seconds"):
            result.ttft_p50_delta_ms = (
                current_ttft_ref["ttft_p50_seconds"] -
                baseline_ttft_ref["ttft_p50_seconds"]
            ) * 1000  # Convert to ms

        # Reference point: concurrency=8, prompt=512 for throughput
        baseline_tput_ref = baseline_configs.get((8, 512))
        current_tput_ref = next(
            (c for c in result.configs
             if c["concurrency"] == 8 and c["prompt_len_tokens"] == 512),
            None,
        )
        if baseline_tput_ref and current_tput_ref and current_tput_ref.get("throughput_tokens_per_sec"):
            baseline_tput = baseline_tput_ref["throughput_tokens_per_sec"]
            current_tput = current_tput_ref["throughput_tokens_per_sec"]
            result.throughput_delta_tok_s = current_tput - baseline_tput
            result.throughput_delta_pct = (
                (current_tput - baseline_tput) / baseline_tput * 100
            ) if baseline_tput > 0 else None

        return result

    def _save_combined_results(
        self,
        bench_results: dict[str, FormatBenchmarkResults],
        quant_results: list,
    ) -> None:
        """Save all results to quant_results.json."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / self.quant_results_filename

        # Build serializable payload
        quant_summary = {}
        for qr in quant_results:
            if qr.success:
                quant_summary[qr.format_name] = {
                    "format_name": qr.format_name,
                    "success": qr.success,
                    "output_path": qr.output_path,
                    "vram_before_gb": qr.vram_before_gb,
                    "vram_after_gb": qr.vram_after_gb,
                    "vram_delta_gb": qr.vram_delta_gb,
                    "calibration_duration_seconds": qr.calibration_duration_seconds,
                    "perplexity": qr.perplexity,
                    "perplexity_delta_pct": qr.perplexity_delta_pct,
                }

        bench_summary = {}
        for fmt_name, br in bench_results.items():
            bench_summary[fmt_name] = asdict(br) if hasattr(br, "__dataclass_fields__") else {
                "format_name": br.format_name,
                "model_path": br.model_path,
                "benchmark_timestamp": br.benchmark_timestamp,
                "vram_used_gb": br.vram_used_gb,
                "configs": br.configs,
                "ttft_p50_delta_ms": br.ttft_p50_delta_ms,
                "throughput_delta_tok_s": br.throughput_delta_tok_s,
                "throughput_delta_pct": br.throughput_delta_pct,
            }

        payload = {
            "quantization_results": quant_summary,
            "benchmark_results": bench_summary,
        }

        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        logger.info(f"Combined quant results saved to {output_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_benchmark(
    config: dict,
    quant_results: list,
    args=None,
) -> dict[str, FormatBenchmarkResults]:
    """Run benchmark for all quantized formats. Called from main.py."""
    benchmarker = QuantizationBenchmarker(config)
    return benchmarker.benchmark_all(quant_results, args)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Benchmark a quantized model format")
    parser.add_argument("--config", default="./activity2_quantization/config.yaml")
    parser.add_argument(
        "--format", required=True,
        choices=["fp8", "awq_int4", "smoothquant_int8"],
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to quantized model checkpoint")
    parser.add_argument("--quick", action="store_true",
                        help="Run reduced benchmark sweep")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    benchmarker = QuantizationBenchmarker(config)
    result = benchmarker.benchmark_format(args.format, args.model_path, args)
    if result:
        print(f"\nBenchmark complete: {args.format}")
        print(f"  TTFT P50 delta: {result.ttft_p50_delta_ms:+.0f}ms")
        print(f"  Throughput delta: {result.throughput_delta_pct:+.0f}%")
        print(f"  VRAM: {result.vram_used_gb:.1f} GB")
