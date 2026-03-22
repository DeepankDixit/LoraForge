"""
activity1_baseline/main.py
============================
Orchestrator for the complete Activity 1 pipeline.

Runs all five steps in sequence:
  1. Merge:     base model + LoRA adapter → merged FP16 model
  2. Serve:     launch vLLM server in the background
  3. Benchmark: sweep concurrency/prompt/output configs → TTFT + throughput
  4. Evaluate:  perplexity + qualitative quality check
  5. Report:    generate baseline_results.json + baseline_report.md

This script coordinates the steps so you don't have to run each manually.
It handles startup/teardown of the vLLM server subprocess automatically.

For individual step control:
  python -m activity1_baseline.merge.merge         # Step 1 only
  python -m activity1_baseline.vllm_server          # Step 2 (foreground)
  python -m activity1_baseline.benchmark.client    # Step 3 only
  python -m activity1_baseline.evaluation.perplexity  # Step 4 only
  python -m activity1_baseline.reporting.report_generator  # Step 5 only

Run:
  python -m activity1_baseline.main
  python -m activity1_baseline.main --skip-merge     # If merged model already exists
  python -m activity1_baseline.main --skip-eval      # Skip perplexity (faster)
  python -m activity1_baseline.main --quick          # Reduced benchmark sweep

Expected wall time on Lambda Cloud A10G:
  Merge:     8–12 minutes  (load 16GB base + save merged)
  Server:    2–4 minutes   (vLLM startup + CUDA graph compilation)
  Benchmark: 30–60 minutes (sweeping all configs with N=20 requests each)
  Eval:      20–40 minutes (perplexity on 500 examples)
  Report:    <1 minute
  TOTAL:     ~60–120 minutes end-to-end
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete Activity 1 baseline pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, default="./activity1_baseline/config.yaml")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge step (use if merged model already exists)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip perplexity evaluation (saves 20-40 minutes)")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced benchmark sweep (for testing pipeline end-to-end)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate report from existing results (skip all other steps)")
    return parser.parse_args()


# ── Pipeline Steps ────────────────────────────────────────────────────────────

def step_merge(config: dict, args: argparse.Namespace) -> str:
    """Step 1: Merge LoRA adapter into base model."""
    if args.skip_merge:
        merged_path = config.get("merged_model_path", "./outputs/cybersec_analyst_merged")
        if not Path(merged_path).exists():
            raise FileNotFoundError(
                f"--skip-merge specified but merged model not found: {merged_path}\n"
                f"Remove --skip-merge to run the merge step."
            )
        logger.info(f"Step 1: Skipping merge (model exists at {merged_path})")
        return merged_path

    logger.info("=" * 60)
    logger.info("STEP 1: MERGE — Baking LoRA adapter into base model weights")
    logger.info("=" * 60)

    from activity1_baseline.merge.merge import run_merge
    merged_path = run_merge(config, args)
    logger.info(f"Step 1 complete: merged model at {merged_path}")
    return merged_path


def step_start_server(config: dict, args: argparse.Namespace) -> subprocess.Popen:
    """Step 2: Launch vLLM server in background."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: SERVE — Launching vLLM server")
    logger.info("=" * 60)

    from activity1_baseline.vllm_server import run_server
    process = run_server(config, args)
    return process


def step_benchmark(config: dict, args: argparse.Namespace) -> None:
    """Step 3: Run benchmark sweep."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: BENCHMARK — Sweeping concurrency levels and prompt lengths")
    logger.info("=" * 60)

    from activity1_baseline.benchmark.client import run_benchmark
    asyncio.run(run_benchmark(config, args))
    logger.info("Step 3 complete: benchmark results saved")


def step_evaluate(config: dict, args: argparse.Namespace) -> None:
    """Step 4: Run perplexity evaluation."""
    if args.skip_eval:
        logger.info("Step 4: Skipping evaluation (--skip-eval specified)")
        return

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: EVALUATE — Computing perplexity and qualitative quality")
    logger.info("=" * 60)

    from activity1_baseline.evaluation.perplexity import run_evaluation
    run_evaluation(config, args)
    logger.info("Step 4 complete: accuracy results saved")


def step_report(config: dict, args: argparse.Namespace) -> None:
    """Step 5: Generate report."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: REPORT — Generating baseline_results.json and baseline_report.md")
    logger.info("=" * 60)

    from activity1_baseline.reporting.report_generator import run_report_generator
    run_report_generator(config, args)
    logger.info("Step 5 complete: report generated")


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create output directories
    Path(config.get("reporting", {}).get("output_dir", "./results")).mkdir(
        parents=True, exist_ok=True
    )

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║          LoraForge — Activity 1: Baseline Deployment          ║")
    logger.info("║      FP16 Baseline: Merge → Serve → Benchmark → Report       ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info("")

    pipeline_start = time.perf_counter()
    server_process = None

    try:
        if args.report_only:
            logger.info("REPORT ONLY mode — generating report from existing results")
            step_report(config, args)
            return

        # Step 1: Merge
        step_merge(config, args)

        # Step 2: Start server
        server_process = step_start_server(config, args)

        # Step 3: Benchmark
        step_benchmark(config, args)

        # Step 4: Evaluate (server must still be running for vLLM-served eval)
        # Note: perplexity evaluation uses the model directly (not via vLLM API)
        # so we can stop the server before eval to free VRAM
        logger.info("Stopping vLLM server to free VRAM for direct evaluation...")
        server_process.terminate()
        server_process.wait(timeout=30)
        server_process = None

        step_evaluate(config, args)

        # Step 5: Report
        step_report(config, args)

        elapsed = time.perf_counter() - pipeline_start
        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════════╗")
        logger.info(f"║  Activity 1 complete in {elapsed/60:.0f} minutes                              ║")
        logger.info("║  Results: ./results/baseline_results.json                    ║")
        logger.info("║  Report:  ./results/baseline_report.md                       ║")
        logger.info("║  Next:    Activity 2 (quantization) to beat these numbers    ║")
        logger.info("╚══════════════════════════════════════════════════════════════╝")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed at step: {e}", exc_info=True)
        raise
    finally:
        if server_process is not None:
            logger.info("Cleaning up vLLM server process...")
            server_process.terminate()
            try:
                server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_process.kill()


if __name__ == "__main__":
    main()
