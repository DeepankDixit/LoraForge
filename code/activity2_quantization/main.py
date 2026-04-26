"""
activity2_quantization/main.py
================================
Orchestrator for the complete Activity 2 quantization pipeline.

Runs all five steps in sequence (or individually):
  1. Calibrate:  Build 512-sample calibration dataset from UltraChat
  2. Quantize:   Apply PTQ via nvidia-modelopt (FP8, AWQ INT4, SmoothQuant INT8)
  3. Benchmark:  Sweep concurrency/prompt/output configs → TTFT + throughput
  4. Select:     Score formats, recommend best fit for each deployment goal
  5. Report:     Generate activity2_quantization_report.md + .html

For individual step control:
  python -m activity2_quantization.main --mode calibrate
  python -m activity2_quantization.main --mode quantize --format fp8
  python -m activity2_quantization.main --mode quantize --format all
  python -m activity2_quantization.main --mode benchmark --format fp8
  python -m activity2_quantization.main --mode select
  python -m activity2_quantization.main --mode report
  python -m activity2_quantization.main --mode all        # Full pipeline

Expected wall time on Lambda Cloud A10G:
  Calibrate:   5–10 minutes   (stream 512 examples, tokenize, cache to .pt)
  Quantize:    60–90 minutes  (per format: 20–30 min each; 3 formats total)
  Benchmark:   30–60 minutes  (30-config sweep per format; 3 formats total)
  Select:      <1 minute
  Report:      <1 minute
  TOTAL:       ~3–4 hours end-to-end

Run:
  python -m activity2_quantization.main --mode all
  python -m activity2_quantization.main --mode all --quick   # Reduced benchmark sweep
  python -m activity2_quantization.main --mode quantize --format fp8  # Single format only
  python -m activity2_quantization.main --mode report        # Report from existing results
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ── Valid choices ─────────────────────────────────────────────────────────────
VALID_FORMATS = ["fp8", "awq_int4", "smoothquant_int8", "all"]
VALID_MODES = ["calibrate", "quantize", "benchmark", "select", "report", "all"]


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Activity 2 post-training quantization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./activity2_quantization/config.yaml",
        help="Path to config.yaml (default: ./activity2_quantization/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="all",
        help=(
            "Pipeline step to run. 'all' runs the full pipeline end-to-end. "
            "Use individual modes for granular control."
        ),
    )
    parser.add_argument(
        "--format",
        choices=VALID_FORMATS,
        default="all",
        help=(
            "Quantization format to quantize/benchmark. "
            "Only used with --mode quantize or --mode benchmark. "
            "'all' applies all enabled formats from config."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced benchmark sweep (for testing pipeline end-to-end quickly)",
    )
    parser.add_argument(
        "--skip-calibrate",
        action="store_true",
        help="Skip calibration step — use existing cached calibration dataset",
    )
    parser.add_argument(
        "--force-rebuild-calib",
        action="store_true",
        help="Force rebuild calibration dataset even if cache exists",
    )
    parser.add_argument(
        "--skip-quick-eval",
        action="store_true",
        help="Skip quick perplexity eval after each quantization (saves ~5 min per format)",
    )
    return parser.parse_args()


# ── Pipeline Steps ────────────────────────────────────────────────────────────

def step_calibrate(config: dict, args: argparse.Namespace):
    """Step 1: Build and cache calibration dataset."""
    logger.info("=" * 60)
    logger.info("STEP 1: CALIBRATE — Building PTQ calibration dataset")
    logger.info("=" * 60)

    from activity2_quantization.quantizer.calibration import CalibrationDatasetBuilder
    builder = CalibrationDatasetBuilder(config)
    dataset = builder.build(force_rebuild=args.force_rebuild_calib)
    logger.info(f"Step 1 complete: {len(dataset)} calibration samples ready")
    return dataset


def step_quantize(
    config: dict,
    args: argparse.Namespace,
    calib_dataset,
    formats: list[str],
) -> list:
    """Step 2: Apply PTQ for each requested format."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STEP 2: QUANTIZE — Applying PTQ ({', '.join(formats)})")
    logger.info("=" * 60)

    from activity2_quantization.quantizer.modelopt_quantizer import QuantizationProfiler

    profiler = QuantizationProfiler(config)
    run_quick_eval = not args.skip_quick_eval

    # Load FP16 perplexity for delta computation (from Activity 1)
    baseline_ppl = 3.68  # Known from baseline_report.md

    results = []
    for fmt in formats:
        logger.info(f"\n--- Quantizing format: {fmt} ---")
        result = profiler.quantize(
            format_name=fmt,
            calib_dataset=calib_dataset,
            run_quick_eval=run_quick_eval,
            baseline_perplexity=baseline_ppl,
        )
        results.append(result)
        if result.success:
            logger.info(f"✓ {fmt} complete: saved to {result.output_path}")
        else:
            logger.error(f"✗ {fmt} failed: {result.error}")

    successful = sum(1 for r in results if r.success)
    logger.info(f"Step 2 complete: {successful}/{len(formats)} formats quantized successfully")
    return results


def step_benchmark(
    config: dict,
    args: argparse.Namespace,
    quant_results: list,
) -> dict:
    """Step 3: Benchmark all quantized formats."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: BENCHMARK — Sweeping quantized formats vs FP16 baseline")
    logger.info("=" * 60)

    from activity2_quantization.benchmark.quant_benchmarker import QuantizationBenchmarker
    benchmarker = QuantizationBenchmarker(config)
    bench_results = benchmarker.benchmark_all(quant_results, args)

    logger.info(f"Step 3 complete: {len(bench_results)} formats benchmarked")
    return bench_results


def step_select(
    config: dict,
    quant_results: list,
    bench_results: dict,
):
    """Step 4: Score formats and produce selection recommendations."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: SELECT — Scoring formats across deployment dimensions")
    logger.info("=" * 60)

    from activity2_quantization.quantizer.format_selector import FormatSelector

    baseline_path = config.get("benchmark", {}).get(
        "baseline_results_path", "./results/baseline_results.json"
    )
    with open(baseline_path) as f:
        baseline_data = json.load(f)

    # Convert bench_results (dict of FormatBenchmarkResults) to the format expected
    # by FormatSelector: {format_name: list[config_dicts]}
    bench_configs = {
        fn: br.configs if hasattr(br, "configs") else br.get("configs", [])
        for fn, br in bench_results.items()
    }

    selector = FormatSelector(config)
    selection_report = selector.select(quant_results, bench_configs, baseline_data)

    logger.info(f"Step 4 complete: best overall = {selection_report.best_overall}")
    return selection_report


def step_report(
    config: dict,
    quant_results: list,
    selection_report,
) -> Path:
    """Step 5: Generate Markdown + HTML comparison report."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: REPORT — Generating quantization comparison report")
    logger.info("=" * 60)

    from activity2_quantization.reporting.comparison_reporter import QuantizationReporter

    report_cfg = config.get("reporting", {})
    quant_results_path = (
        Path(report_cfg.get("output_dir", "./results"))
        / report_cfg.get("quant_results_filename", "quant_results.json")
    )

    with open(quant_results_path) as f:
        quant_data = json.load(f)

    reporter = QuantizationReporter(config)
    path = reporter.generate(quant_data, selection_report)

    logger.info(f"Step 5 complete: report at {path}")
    return path


# ── Format Resolution ─────────────────────────────────────────────────────────

def resolve_formats(args: argparse.Namespace, config: dict) -> list[str]:
    """Resolve --format flag to an explicit list of format names."""
    if args.format == "all":
        formats_config = config.get("quantization", {}).get("formats", {})
        enabled = [
            name for name, cfg in formats_config.items()
            if cfg.get("enabled", True)
        ]
        if not enabled:
            logger.warning("No formats enabled in config. Check quantization.formats section.")
        return enabled
    return [args.format]


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    Path(config.get("reporting", {}).get("output_dir", "./results")).mkdir(
        parents=True, exist_ok=True
    )

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║         LoraForge — Activity 2: Quantization Pipeline         ║")
    logger.info("║    Calibrate → Quantize → Benchmark → Select → Report        ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"Mode: {args.mode} | Format: {args.format} | Quick: {args.quick}")
    logger.info("")

    pipeline_start = time.perf_counter()
    formats = resolve_formats(args, config)

    try:
        # ── Mode: report only ──────────────────────────────────────────────────
        if args.mode == "report":
            step_report(config, quant_results=[], selection_report=None)
            return

        # ── Mode: select only ─────────────────────────────────────────────────
        if args.mode == "select":
            # Load results from disk
            report_cfg = config.get("reporting", {})
            quant_results_path = (
                Path(report_cfg.get("output_dir", "./results"))
                / report_cfg.get("quant_results_filename", "quant_results.json")
            )
            if not quant_results_path.exists():
                raise FileNotFoundError(
                    f"quant_results.json not found at {quant_results_path}. "
                    f"Run --mode benchmark first."
                )
            with open(quant_results_path) as f:
                quant_data = json.load(f)

            # Reconstruct minimal result objects
            quant_results = [
                _MinimalQuantResult(fn, True, v.get("vram_after_gb"), v.get("perplexity"), v.get("perplexity_delta_pct"))
                for fn, v in quant_data.get("quantization_results", {}).items()
            ]
            bench_results = quant_data.get("benchmark_results", {})
            bench_configs = {fn: v.get("configs", []) for fn, v in bench_results.items()}

            from activity2_quantization.quantizer.format_selector import FormatSelector
            import json as _json
            baseline_path = config.get("benchmark", {}).get(
                "baseline_results_path", "./results/baseline_results.json"
            )
            with open(baseline_path) as f:
                baseline_data = _json.load(f)
            selector = FormatSelector(config)
            report = selector.select(quant_results, bench_configs, baseline_data)
            print(report.summary_table())
            return

        # ── Mode: calibrate only ──────────────────────────────────────────────
        if args.mode == "calibrate":
            step_calibrate(config, args)
            return

        # ── Modes requiring calibration ───────────────────────────────────────
        if args.mode in ("quantize", "all"):
            # Step 1: Calibrate
            if not args.skip_calibrate:
                calib_dataset = step_calibrate(config, args)
            else:
                from activity2_quantization.quantizer.calibration import CalibrationDatasetBuilder
                calib_dataset = CalibrationDatasetBuilder(config).build()

            # Step 2: Quantize
            quant_results = step_quantize(config, args, calib_dataset, formats)

            if args.mode == "quantize":
                return  # Stop after quantize if only quantize was requested

        # ── Modes requiring quant results ─────────────────────────────────────
        if args.mode == "benchmark":
            # Load existing quant results if we didn't just quantize
            quant_results = _load_quant_results_from_disk(config, formats)

        if args.mode in ("benchmark", "all"):
            bench_results = step_benchmark(config, args, quant_results)
        else:
            bench_results = {}

        # ── Steps 4 + 5 ───────────────────────────────────────────────────────
        selection_report = None
        if args.mode == "all":
            selection_report = step_select(config, quant_results, bench_results)
            step_report(config, quant_results, selection_report)

        elapsed = time.perf_counter() - pipeline_start
        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════════╗")
        logger.info(f"║  Activity 2 complete in {elapsed/60:.0f} minutes                              ║")
        logger.info("║  Results:  ./results/quant_results.json                       ║")
        logger.info("║  Report:   ./results/activity2_quantization_report.md         ║")
        logger.info("║  Next:     Activity 3 (KV cache prefix sharing)               ║")
        logger.info("╚══════════════════════════════════════════════════════════════╝")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _MinimalQuantResult:
    """Thin wrapper for reconstructed QuantizationResult from JSON."""
    def __init__(self, format_name, success, vram_after_gb, perplexity, perplexity_delta_pct):
        self.format_name = format_name
        self.success = success
        self.vram_after_gb = vram_after_gb
        self.perplexity = perplexity
        self.perplexity_delta_pct = perplexity_delta_pct
        self.output_path = None


def _load_quant_results_from_disk(config: dict, formats: list[str]) -> list:
    """Load quantization results from disk for benchmark-only mode."""
    quant_output_dir = Path(config.get("quantized_output_dir", "./outputs/quantized"))
    results = []
    for fmt in formats:
        model_path = quant_output_dir / fmt
        if model_path.exists():
            results.append(type("_QR", (), {
                "format_name": fmt,
                "success": True,
                "output_path": str(model_path),
                "vram_before_gb": 0,
                "vram_after_gb": 0,
                "vram_delta_gb": 0,
                "calibration_duration_seconds": 0,
                "perplexity": None,
                "perplexity_delta_pct": None,
                "error": None,
            })())
        else:
            logger.warning(f"Quantized model not found for {fmt} at {model_path}. Skipping.")
    return results


if __name__ == "__main__":
    main()
