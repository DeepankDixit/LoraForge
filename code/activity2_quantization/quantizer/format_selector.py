"""
activity2_quantization/quantizer/format_selector.py
=====================================================
FormatSelector — scores all quantized formats and recommends the best fit
for each deployment goal.

THE SELECTION PROBLEM
----------------------
After running all three quantization formats, we have a tradeoff table:

  Format              | VRAM (GB) | TTFT P50 (ms) | Throughput (tok/s) | Perplexity Δ
  --------------------|-----------|---------------|-------------------|-------------
  FP16 baseline       |  16.8     |  85           |  205              |  0%
  FP8                 |  ~8.5     |  ~70          |  ~350             |  ~0.3%
  SmoothQuant INT8    |  ~8.5     |  ~72          |  ~400             |  ~0.8%
  AWQ INT4            |  ~4.5     |  ~65          |  ~280             |  ~1.8%

No single format wins on all three dimensions. The right choice depends on the
deployment constraint:

  Latency-critical:   Minimize TTFT — AWQ INT4 (smallest KV cache → more room for batching)
  Throughput-critical: Maximize tok/s — SmoothQuant W8A8 (INT8 multiply throughput on A10G)
  Accuracy-critical:  Minimize perplexity Δ — FP8 (near-lossless)
  VRAM-constrained:   Minimize VRAM — AWQ INT4 (~4.5 GB, fits on 8 GB GPU)

FormatSelector computes a weighted composite score for each axis and returns
structured recommendations with justifications.

Run directly:
  python -m activity2_quantization.quantizer.format_selector
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class FormatScore:
    """Scores for a single quantization format on each deployment axis."""
    format_name: str

    # Raw metrics (from benchmark and quantization results)
    vram_gb: Optional[float]
    ttft_p50_ms: Optional[float]       # At concurrency=1, prompt=512 tokens
    throughput_tok_s: Optional[float]  # At concurrency=8, prompt=512 tokens
    perplexity: Optional[float]
    perplexity_delta_pct: Optional[float]  # vs FP16 baseline

    # Normalized scores [0, 1] where higher is better
    latency_score: float = 0.0
    throughput_score: float = 0.0
    accuracy_score: float = 0.0
    vram_score: float = 0.0

    # Composite score (weighted sum)
    composite_score: float = 0.0

    # Human-readable recommendation
    best_for: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SelectionReport:
    """Structured output of FormatSelector.select()."""
    formats: list[FormatScore]

    # Top recommendation for each goal
    best_latency: Optional[str]
    best_throughput: Optional[str]
    best_accuracy: Optional[str]
    best_vram: Optional[str]
    best_overall: Optional[str]        # Highest composite score

    baseline_vram_gb: float
    baseline_ttft_p50_ms: float
    baseline_throughput_tok_s: float
    baseline_perplexity: float

    def summary_table(self) -> str:
        """Return an ASCII summary table of all scores."""
        lines = [
            "\n┌─────────────────────────────────────────────────────────────────────┐",
            "│            Format Selection Summary                                  │",
            "├────────────────────┬──────────┬─────────┬──────────┬───────────────┤",
            "│ Format             │ VRAM(GB) │ TTFT(ms)│ Tok/s    │ Perplexity Δ  │",
            "├────────────────────┼──────────┼─────────┼──────────┼───────────────┤",
        ]
        for fs in sorted(self.formats, key=lambda x: x.composite_score, reverse=True):
            lines.append(
                f"│ {fs.format_name:<18} │ "
                f"{fs.vram_gb or 0:>7.1f}  │ "
                f"{fs.ttft_p50_ms or 0:>6.0f}  │ "
                f"{fs.throughput_tok_s or 0:>7.0f}  │ "
                f"{fs.perplexity_delta_pct or 0:>+10.2f}%   │"
            )
        lines += [
            "└────────────────────┴──────────┴─────────┴──────────┴───────────────┘",
            f"\n  Best for latency:    {self.best_latency}",
            f"  Best for throughput: {self.best_throughput}",
            f"  Best for accuracy:   {self.best_accuracy}",
            f"  Best for VRAM:       {self.best_vram}",
            f"  Best overall:        {self.best_overall}",
        ]
        return "\n".join(lines)


# ── FormatSelector ────────────────────────────────────────────────────────────

class FormatSelector:
    """
    Scores and ranks quantization formats across deployment dimensions.

    Scoring approach: min-max normalization within each metric axis,
    then weighted sum across axes. This is deliberately simple — the point
    is not to give a single "correct" answer but to make the tradeoffs
    legible and justifiable.

    Usage:
        selector = FormatSelector(config)
        report = selector.select(quant_results, benchmark_results, baseline_results)
        print(report.summary_table())
    """

    def __init__(self, config: dict) -> None:
        sel_cfg = config.get("format_selection", {})
        self.max_perplexity_delta_pct: float = sel_cfg.get(
            "max_acceptable_perplexity_delta_pct", 5.0
        )
        self.target_throughput_gain_pct: float = sel_cfg.get(
            "target_throughput_gain_pct", 50.0
        )
        self.target_ttft_reduction_pct: float = sel_cfg.get(
            "target_ttft_reduction_pct", 20.0
        )
        self.w_latency: float = sel_cfg.get("latency_weight", 0.33)
        self.w_throughput: float = sel_cfg.get("throughput_weight", 0.34)
        self.w_accuracy: float = sel_cfg.get("accuracy_weight", 0.33)

    def select(
        self,
        quant_results: list,        # list[QuantizationResult]
        bench_results: dict,        # {format_name: list[ConfigResult]}
        baseline_results: dict,     # Loaded from baseline_results.json
    ) -> SelectionReport:
        """
        Score all formats and produce a SelectionReport.

        Args:
            quant_results:    List of QuantizationResult from QuantizationProfiler.
            bench_results:    Dict mapping format_name → list of ConfigResult dicts
                              (from QuantizationBenchmarker).
            baseline_results: Parsed baseline_results.json from Activity 1.

        Returns:
            SelectionReport with scores, rankings, and recommendations.
        """
        baseline = self._extract_baseline_metrics(baseline_results)
        format_scores = []

        for qr in quant_results:
            if not qr.success:
                logger.warning(f"Skipping {qr.format_name}: quantization failed")
                continue

            bench = bench_results.get(qr.format_name, [])
            metrics = self._extract_format_metrics(qr, bench, baseline)
            format_scores.append(metrics)

        if not format_scores:
            logger.error("No successful quantization results to score.")
            return SelectionReport(
                formats=[], best_latency=None, best_throughput=None,
                best_accuracy=None, best_vram=None, best_overall=None,
                baseline_vram_gb=baseline["vram_gb"],
                baseline_ttft_p50_ms=baseline["ttft_p50_ms"],
                baseline_throughput_tok_s=baseline["throughput_tok_s"],
                baseline_perplexity=baseline["perplexity"],
            )

        # Normalize scores within each axis (min-max, higher is better)
        self._normalize_scores(format_scores, baseline)

        # Compute composite score
        for fs in format_scores:
            fs.composite_score = (
                self.w_latency * fs.latency_score
                + self.w_throughput * fs.throughput_score
                + self.w_accuracy * fs.accuracy_score
            )

        # Assign best_for tags
        for fs in format_scores:
            self._assign_best_for_tags(fs, baseline)

        # Build SelectionReport
        report = SelectionReport(
            formats=format_scores,
            best_latency=self._best(format_scores, "latency_score"),
            best_throughput=self._best(format_scores, "throughput_score"),
            best_accuracy=self._best(format_scores, "accuracy_score"),
            best_vram=self._best(format_scores, "vram_score"),
            best_overall=self._best(format_scores, "composite_score"),
            baseline_vram_gb=baseline["vram_gb"],
            baseline_ttft_p50_ms=baseline["ttft_p50_ms"],
            baseline_throughput_tok_s=baseline["throughput_tok_s"],
            baseline_perplexity=baseline["perplexity"],
        )

        logger.info(report.summary_table())
        return report

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _extract_baseline_metrics(self, baseline_results: dict) -> dict:
        """Pull key metrics from baseline_results.json."""
        configs = baseline_results.get("configs", [])
        # TTFT P50 at concurrency=1, prompt=512 tokens
        ttft_ref = next(
            (c for c in configs if c["concurrency"] == 1 and c["prompt_len_tokens"] == 512),
            None,
        )
        # Throughput at concurrency=8, prompt=512 tokens
        tput_ref = next(
            (c for c in configs if c["concurrency"] == 8 and c["prompt_len_tokens"] == 512),
            None,
        )
        return {
            "vram_gb": 16.8,   # Known from Activity 1 report
            "ttft_p50_ms": (ttft_ref["ttft_p50_seconds"] * 1000) if ttft_ref else 85.0,
            "throughput_tok_s": tput_ref["throughput_tokens_per_sec"] if tput_ref else 205.0,
            "perplexity": 3.68,  # From Activity 1 merged model perplexity
        }

    def _extract_format_metrics(self, qr, bench_configs: list, baseline: dict) -> FormatScore:
        """Extract raw metrics from a QuantizationResult + its benchmark configs."""
        # TTFT at concurrency=1, prompt=512
        ttft_ref = next(
            (c for c in bench_configs
             if c.get("concurrency") == 1 and c.get("prompt_len_tokens") == 512),
            None,
        )
        # Throughput at concurrency=8, prompt=512
        tput_ref = next(
            (c for c in bench_configs
             if c.get("concurrency") == 8 and c.get("prompt_len_tokens") == 512),
            None,
        )

        return FormatScore(
            format_name=qr.format_name,
            vram_gb=qr.vram_after_gb if qr.vram_after_gb else None,
            ttft_p50_ms=(ttft_ref["ttft_p50_seconds"] * 1000) if ttft_ref else None,
            throughput_tok_s=tput_ref.get("throughput_tokens_per_sec") if tput_ref else None,
            perplexity=qr.perplexity,
            perplexity_delta_pct=qr.perplexity_delta_pct,
        )

    def _normalize_scores(self, format_scores: list[FormatScore], baseline: dict) -> None:
        """
        Normalize each metric to [0, 1]. Higher score = better on that axis.

        Latency:   lower TTFT → higher score
        Throughput: higher tok/s → higher score
        Accuracy:  lower perplexity Δ → higher score
        VRAM:      lower VRAM → higher score
        """
        # Filter out None values
        valid_ttft = [fs.ttft_p50_ms for fs in format_scores if fs.ttft_p50_ms is not None]
        valid_tput = [fs.throughput_tok_s for fs in format_scores if fs.throughput_tok_s is not None]
        valid_ppl  = [abs(fs.perplexity_delta_pct or 0) for fs in format_scores]
        valid_vram = [fs.vram_gb for fs in format_scores if fs.vram_gb is not None]

        # Include baseline in normalization range so improvements are meaningful
        valid_ttft.append(baseline["ttft_p50_ms"])
        valid_tput.append(baseline["throughput_tok_s"])
        valid_vram.append(baseline["vram_gb"])
        valid_ppl.append(0.0)  # Baseline has 0% perplexity delta

        def normalize_lower_is_better(val, vals):
            if val is None:
                return 0.0
            mn, mx = min(vals), max(vals)
            if mx == mn:
                return 1.0
            return 1.0 - (val - mn) / (mx - mn)

        def normalize_higher_is_better(val, vals):
            if val is None:
                return 0.0
            mn, mx = min(vals), max(vals)
            if mx == mn:
                return 1.0
            return (val - mn) / (mx - mn)

        for fs in format_scores:
            fs.latency_score = normalize_lower_is_better(fs.ttft_p50_ms, valid_ttft)
            fs.throughput_score = normalize_higher_is_better(fs.throughput_tok_s, valid_tput)
            # Accuracy: penalize formats with perplexity delta above threshold
            ppl_delta = abs(fs.perplexity_delta_pct or 0)
            if ppl_delta > self.max_perplexity_delta_pct:
                fs.accuracy_score = 0.0
            else:
                fs.accuracy_score = normalize_lower_is_better(ppl_delta, valid_ppl)
            fs.vram_score = normalize_lower_is_better(fs.vram_gb, valid_vram)

    def _assign_best_for_tags(self, fs: FormatScore, baseline: dict) -> None:
        """Assign human-readable deployment recommendation tags."""
        tags = []

        # Latency: TTFT improvement ≥ threshold
        if fs.ttft_p50_ms and baseline["ttft_p50_ms"]:
            ttft_improvement = (1 - fs.ttft_p50_ms / baseline["ttft_p50_ms"]) * 100
            if ttft_improvement >= self.target_ttft_reduction_pct:
                tags.append(f"latency-sensitive (TTFT -{ttft_improvement:.0f}%)")

        # Throughput: tok/s improvement ≥ threshold
        if fs.throughput_tok_s and baseline["throughput_tok_s"]:
            tput_improvement = (
                fs.throughput_tok_s / baseline["throughput_tok_s"] - 1
            ) * 100
            if tput_improvement >= self.target_throughput_gain_pct:
                tags.append(f"high-throughput (+{tput_improvement:.0f}% tok/s)")

        # Accuracy: low perplexity degradation
        ppl_delta = abs(fs.perplexity_delta_pct or 0)
        if ppl_delta < 1.0:
            tags.append("accuracy-critical (<1% perplexity loss)")

        # VRAM: fits on smaller GPU
        if fs.vram_gb and fs.vram_gb < 8.0:
            tags.append(f"VRAM-constrained ({fs.vram_gb:.1f} GB, fits 8 GB GPU)")

        fs.best_for = tags if tags else ["general purpose"]
        fs.notes = (
            f"Perplexity delta: {fs.perplexity_delta_pct:+.2f}% vs FP16"
            if fs.perplexity_delta_pct is not None else ""
        )

    @staticmethod
    def _best(format_scores: list[FormatScore], attr: str) -> Optional[str]:
        """Return name of format with highest score on given attribute."""
        valid = [fs for fs in format_scores if getattr(fs, attr, 0) > 0]
        if not valid:
            return None
        return max(valid, key=lambda fs: getattr(fs, attr)).format_name


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_format_selection(
    config: dict,
    quant_results: list,
    bench_results: dict,
    baseline_results_path: str,
    args=None,
) -> SelectionReport:
    """Run format selection. Called from main.py."""
    with open(baseline_results_path) as f:
        baseline_data = json.load(f)

    selector = FormatSelector(config)
    return selector.select(quant_results, bench_results, baseline_data)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run format selection from saved results")
    parser.add_argument("--config", default="./activity2_quantization/config.yaml")
    parser.add_argument(
        "--quant-results", default="./results/quant_results.json",
        help="Path to quant_results.json generated by QuantizationBenchmarker",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    baseline_path = config.get("benchmark", {}).get(
        "baseline_results_path", "./results/baseline_results.json"
    )
    with open(args.quant_results) as f:
        quant_data = json.load(f)

    # Reconstruct minimal QuantizationResult-like objects from saved JSON
    from dataclasses import dataclass as dc

    @dc
    class _QR:
        format_name: str
        success: bool
        vram_after_gb: Optional[float]
        perplexity: Optional[float]
        perplexity_delta_pct: Optional[float]

    quant_results = [
        _QR(
            format_name=v["format_name"],
            success=v.get("success", True),
            vram_after_gb=v.get("vram_after_gb"),
            perplexity=v.get("perplexity"),
            perplexity_delta_pct=v.get("perplexity_delta_pct"),
        )
        for v in quant_data.get("quantization_results", {}).values()
    ]
    bench_results = quant_data.get("benchmark_results", {})

    with open(baseline_path) as f:
        baseline_data = json.load(f)

    selector = FormatSelector(config)
    report = selector.select(quant_results, bench_results, baseline_data)
    print(report.summary_table())
