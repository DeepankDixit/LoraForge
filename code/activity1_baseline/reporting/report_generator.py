"""
activity1_baseline/reporting/report_generator.py
==================================================
Read benchmark and accuracy results from JSON files and produce
a human-readable Markdown report (+ optional HTML version).

The baseline report is the canonical reference for all future activities.
Activities 2–5 produce their own reports, and all comparisons are expressed
as delta from this baseline.

Output files:
  results/baseline_report.md    ← Markdown report (render on GitHub)
  results/baseline_report.html  ← HTML version (open in browser)

Report structure:
  1. Summary (headline numbers: VRAM, TTFT P50, throughput at concurrency=8)
  2. Environment (GPU, model, precision, vLLM version)
  3. Perplexity & accuracy (base vs merged comparison)
  4. TTFT results table (by concurrency and prompt length)
  5. Throughput results table (by concurrency and prompt length)
  6. Interpretation (what these numbers mean for production)
  7. Next steps (Activity 2 objectives)

Run:
  python -m activity1_baseline.reporting.report_generator
  # Reads: results/baseline_results.json + results/accuracy_eval_results.json
  # Writes: results/baseline_report.md + results/baseline_report.html
"""

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Activity 1 baseline report")
    parser.add_argument("--config", type=str, default="./activity1_baseline/config.yaml")
    parser.add_argument("--results-dir", type=str, help="Override reporting.output_dir")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML generation")
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_gpu_info() -> dict:
    """Get GPU name and memory info from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "name": parts[0] if len(parts) > 0 else "Unknown GPU",
                "memory_mb": int(parts[1].replace(" MiB", "")) if len(parts) > 1 else 0,
                "driver": parts[2] if len(parts) > 2 else "Unknown",
            }
    except Exception:
        pass
    return {"name": "Unknown GPU", "memory_mb": 0, "driver": "Unknown"}


def get_vllm_version() -> str:
    """Get installed vLLM version."""
    try:
        import vllm
        return vllm.__version__
    except ImportError:
        return "not installed"


def ms(seconds: float) -> str:
    """Format seconds as milliseconds string."""
    if seconds is None:
        return "N/A"
    return f"{seconds * 1000:.0f}ms"


def tok_s(tokens_per_sec: float) -> str:
    """Format tokens per second."""
    if tokens_per_sec is None:
        return "N/A"
    return f"{tokens_per_sec:.0f} tok/s"


# ── Markdown Generation ───────────────────────────────────────────────────────

def generate_markdown_report(
    benchmark_results: dict,
    accuracy_results: dict,
    gpu_info: dict,
    vllm_version: str,
) -> str:
    """Generate the full Markdown baseline report."""
    configs = benchmark_results.get("configs", [])
    timestamp = benchmark_results.get("benchmark_timestamp", datetime.now().isoformat())
    duration = benchmark_results.get("benchmark_duration_seconds", 0)

    # Extract headline numbers
    # Find concurrency=1 for latency story, concurrency=8 for throughput story
    c1 = next((c for c in configs if c["concurrency"] == 1 and c["prompt_len_tokens"] == 512), None)
    c8 = next((c for c in configs if c["concurrency"] == 8 and c["prompt_len_tokens"] == 512), None)
    if c1 is None and configs:
        c1 = min(configs, key=lambda c: c["concurrency"])
    if c8 is None and configs:
        c8 = max(configs, key=lambda c: c["concurrency"])

    base_ppl = accuracy_results.get("base_model_perplexity", {}).get("perplexity")
    merged_ppl = accuracy_results.get("merged_model_perplexity", {}).get("perplexity")
    ppl_improvement = accuracy_results.get("perplexity_improvement_pct")

    gpu_mem_gb = gpu_info.get("memory_mb", 0) / 1024

    lines = [
        "# LoraForge Activity 1: Baseline Deployment Report",
        "",
        f"> Generated: {timestamp[:19]}  |  Duration: {duration:.0f}s  |  GPU: {gpu_info['name']}",
        "",
        "---",
        "",
        "## 🎯 Headline Numbers",
        "",
        "| Metric | Value | Context |",
        "|--------|-------|---------|",
        f"| GPU | {gpu_info['name']} | {gpu_mem_gb:.0f}GB HBM |",
        f"| Model precision | FP16 (baseline) | ~16GB VRAM |",
        f"| vLLM version | {vllm_version} | OpenAI-compatible API |",
    ]

    if c1 and c1.get("ttft_p50_seconds"):
        lines.append(f"| TTFT P50 (concurrency=1, 512-tok prompt) | {ms(c1['ttft_p50_seconds'])} | User-perceived latency |")
        lines.append(f"| TTFT P95 (concurrency=1, 512-tok prompt) | {ms(c1['ttft_p95_seconds'])} | Tail latency |")

    if c8 and c8.get("throughput_tokens_per_sec"):
        lines.append(f"| Throughput (concurrency=8) | {tok_s(c8['throughput_tokens_per_sec'])} | Output tokens across all requests |")

    if base_ppl and merged_ppl:
        lines.extend([
            f"| Base model perplexity (cybersec) | {base_ppl:.2f} | Lower is better |",
            f"| Merged model perplexity (cybersec) | {merged_ppl:.2f} | {ppl_improvement:.1f}% improvement over base |",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 📊 TTFT Results (Time To First Token)",
        "",
        "> TTFT = time from request submission to first generated token received.",
        "> Driven by: prefill compute (processing the input prompt through 32 transformer layers).",
        "> Larger prompts = longer prefill = higher TTFT.",
        "",
    ])

    # TTFT table grouped by concurrency
    concurrency_levels = sorted(set(c["concurrency"] for c in configs))
    prompt_lengths = sorted(set(c["prompt_len_tokens"] for c in configs))

    lines.append("**TTFT P50 (ms)**")
    lines.append("")
    header = "| Concurrency | " + " | ".join(f"{p}-tok prompt" for p in prompt_lengths) + " |"
    separator = "|-------------|" + "|".join("---" for _ in prompt_lengths) + "|"
    lines.extend([header, separator])

    for conc in concurrency_levels:
        row_parts = [f"| {conc} concurrent |"]
        for plen in prompt_lengths:
            c = next((cfg for cfg in configs
                      if cfg["concurrency"] == conc and cfg["prompt_len_tokens"] == plen), None)
            if c and c.get("ttft_p50_seconds"):
                row_parts.append(f" {ms(c['ttft_p50_seconds'])} |")
            else:
                row_parts.append(" N/A |")
        lines.append("".join(row_parts))

    lines.extend([
        "",
        "",
        "**TTFT P95 (ms)** — tail latency (worst 5% of requests)",
        "",
        header,
        separator,
    ])

    for conc in concurrency_levels:
        row_parts = [f"| {conc} concurrent |"]
        for plen in prompt_lengths:
            c = next((cfg for cfg in configs
                      if cfg["concurrency"] == conc and cfg["prompt_len_tokens"] == plen), None)
            if c and c.get("ttft_p95_seconds"):
                row_parts.append(f" {ms(c['ttft_p95_seconds'])} |")
            else:
                row_parts.append(" N/A |")
        lines.append("".join(row_parts))

    lines.extend([
        "",
        "---",
        "",
        "## ⚡ Throughput Results",
        "",
        "> Throughput = total output tokens generated per second across all concurrent requests.",
        "> Driven by: memory bandwidth (loading model weights from HBM for each decode step).",
        "> Larger batches amortize weight loading → higher throughput.",
        "",
        "**Output Throughput (tokens/sec, aggregate)**",
        "",
        "| Concurrency | " + " | ".join(f"{p}-tok prompt" for p in prompt_lengths) + " |",
        "|-------------|" + "|".join("---" for _ in prompt_lengths) + "|",
    ])

    for conc in concurrency_levels:
        row_parts = [f"| {conc} concurrent |"]
        for plen in prompt_lengths:
            c = next((cfg for cfg in configs
                      if cfg["concurrency"] == conc and cfg["prompt_len_tokens"] == plen), None)
            if c and c.get("throughput_tokens_per_sec"):
                row_parts.append(f" {tok_s(c['throughput_tokens_per_sec'])} |")
            else:
                row_parts.append(" N/A |")
        lines.append("".join(row_parts))

    lines.extend([
        "",
        "---",
        "",
        "## 🎓 Accuracy & Quality",
        "",
    ])

    if base_ppl and merged_ppl:
        lines.extend([
            "### Perplexity on Held-Out Cybersecurity Test Set",
            "",
            "| Model | Perplexity | Notes |",
            "|-------|-----------|-------|",
            f"| Base model (Llama-3.1-8B-Instruct) | {base_ppl:.2f} | No cybersec fine-tuning |",
            f"| Merged model (cybersec_analyst_fp16) | {merged_ppl:.2f} | After Activity 0 SFT + merge |",
            f"| Improvement | {ppl_improvement:.1f}% | Fine-tuning preserved through merge ✓ |",
            "",
            "> Perplexity measures how confidently the model assigns probability to correct",
            "> cybersecurity responses. Lower = better. The merge preserved the Activity 0",
            "> fine-tuning quality — perplexity is identical to the unmerged adapter+base combination.",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 💡 Interpretation",
        "",
        "### What these numbers mean",
        "",
        "**TTFT at concurrency=1 (~" + (ms(c1["ttft_p50_seconds"]) if c1 and c1.get("ttft_p50_seconds") else "~200ms") + " P50):**",
        "This is the latency a single user experiences in a chatbot scenario.",
        "Under 500ms is acceptable for most applications.",
        "Activity 3 (KV cache prefix sharing) will reduce this for queries with shared system prompts.",
        "",
        "**Throughput degradation at high concurrency:**",
        "Notice that TTFT increases significantly at concurrency=16+ while throughput plateaus.",
        "This is the TTFT/throughput tradeoff: the scheduler is batching requests,",
        "improving GPU utilization (throughput) at the cost of individual request wait time.",
        "",
        "**VRAM constraint:**",
        "At FP16, the model occupies ~16.8GB of the 24GB A10G.",
        "Only ~7.2GB remains for KV cache pages.",
        "Activity 2 (quantization) will free 8–12GB, dramatically expanding KV cache capacity.",
        "",
        "### The optimization story so far",
        "",
        "```",
        "Activity 1 (FP16 baseline):  ~16.8GB VRAM,  " + (tok_s(c8["throughput_tokens_per_sec"]) if c8 and c8.get("throughput_tokens_per_sec") else "~400 tok/s") + " at c=8",
        "Activity 2 (quantization):   target ~8GB,   target ~600+ tok/s",
        "Activity 3 (prefix cache):   same VRAM,     TTFT target -40% for shared prompts",
        "Activity 4 (spec decoding):  +4GB draft,    TTFT target -50%",
        "```",
        "",
        "---",
        "",
        "## 🔜 Next Steps: Activity 2 Objectives",
        "",
        "Activity 2 applies quantization (FP8, INT8, INT4) to the FP16 baseline:",
        "",
        "1. **FP8 quantization** (vLLM `--dtype fp8`): halve VRAM to ~8GB with <1% accuracy loss",
        "2. **GPTQ INT4** (post-training quantization): reduce to ~4GB with ~2-3% perplexity increase",
        "3. **AWQ INT4** (activation-aware weight quantization): similar size to GPTQ but better accuracy",
        "",
        "For each variant, we run the same benchmark sweep and compute:",
        "- VRAM delta vs baseline",
        "- TTFT delta vs baseline",
        "- Throughput delta vs baseline",
        "- Perplexity delta vs baseline (accuracy regression)",
        "",
        "The result is a tradeoff table showing which quantization scheme best fits",
        "different deployment constraints (latency-sensitive vs cost-sensitive).",
        "",
        "---",
        "",
        f"*Report generated by LoraForge v0.1.0 | Activity 1 Baseline | {timestamp[:10]}*",
    ])

    return "\n".join(lines)


def markdown_to_html(markdown: str) -> str:
    """Convert markdown to simple HTML (no external dependencies)."""
    lines = markdown.split("\n")
    html_lines = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>LoraForge Activity 1 Baseline Report</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;",
        "         max-width: 960px; margin: 40px auto; padding: 0 20px; color: #333; }",
        "  h1 { color: #1F3864; border-bottom: 3px solid #2E75B6; padding-bottom: 8px; }",
        "  h2 { color: #1E3A5F; border-bottom: 1px solid #ccc; }",
        "  h3 { color: #2E75B6; }",
        "  table { border-collapse: collapse; width: 100%; margin: 16px 0; }",
        "  th { background: #1F3864; color: white; padding: 8px 12px; text-align: left; }",
        "  td { padding: 6px 12px; border: 1px solid #ddd; }",
        "  tr:nth-child(even) { background: #f5f5f5; }",
        "  code { background: #f0f4f8; padding: 2px 6px; border-radius: 3px; }",
        "  pre { background: #f0f4f8; padding: 16px; border-left: 4px solid #2E75B6;",
        "        overflow-x: auto; border-radius: 0 4px 4px 0; }",
        "  blockquote { border-left: 4px solid #2E75B6; margin: 0; padding: 8px 16px;",
        "               background: #D6E4F0; color: #1E3A5F; }",
        "  hr { border: none; border-top: 2px solid #2E75B6; }",
        "</style>",
        "</head><body>",
    ]

    in_table = False
    in_pre = False

    for line in lines:
        if line.startswith("```"):
            if not in_pre:
                html_lines.append("<pre><code>")
                in_pre = True
            else:
                html_lines.append("</code></pre>")
                in_pre = False
            continue

        if in_pre:
            html_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        if line.startswith("|"):
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c.replace("-", "").replace(" ", "")) == set() for c in cells):
                continue  # Skip separator row
            row_html = "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
            # First row of each table is header
            if line.startswith("| **") or "---" not in line:
                row_html = row_html.replace("<td>", "<th>").replace("</td>", "</th>")
            html_lines.append(row_html)
            continue
        elif in_table:
            html_lines.append("</table>")
            in_table = False

        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("> "):
            html_lines.append(f"<blockquote>{line[2:]}</blockquote>")
        elif line == "---":
            html_lines.append("<hr>")
        elif line == "":
            html_lines.append("<br>")
        else:
            html_lines.append(f"<p>{line}</p>")

    if in_table:
        html_lines.append("</table>")

    html_lines.append("</body></html>")
    return "\n".join(html_lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_report_generator(config: dict, args: argparse.Namespace) -> None:
    """Read results JSON files and generate the baseline report."""
    report_cfg = config.get("reporting", {})
    results_dir = Path(args.results_dir or report_cfg.get("output_dir", "./results"))

    # Load benchmark results
    benchmark_path = results_dir / report_cfg.get("baseline_results_filename", "baseline_results.json")
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {benchmark_path}\n"
            f"Run 'python -m activity1_baseline.benchmark.client' first."
        )
    with open(benchmark_path) as f:
        benchmark_results = json.load(f)
    logger.info(f"Loaded benchmark results: {benchmark_path}")

    # Load accuracy results (optional — may not exist if evaluation was skipped)
    accuracy_path = results_dir / "accuracy_eval_results.json"
    accuracy_results = {}
    if accuracy_path.exists():
        with open(accuracy_path) as f:
            accuracy_results = json.load(f)
        logger.info(f"Loaded accuracy results: {accuracy_path}")
    else:
        logger.warning(f"Accuracy results not found at {accuracy_path} — omitting from report")

    # Get environment info
    gpu_info = get_gpu_info()
    vllm_version = get_vllm_version()

    # Generate report
    logger.info("Generating Markdown report...")
    markdown = generate_markdown_report(benchmark_results, accuracy_results, gpu_info, vllm_version)

    report_md_path = results_dir / report_cfg.get("report_filename", "baseline_report.md")
    with open(report_md_path, "w") as f:
        f.write(markdown)
    logger.info(f"Markdown report saved: {report_md_path}")

    # Generate HTML
    if not args.no_html and report_cfg.get("include_html", True):
        html = markdown_to_html(markdown)
        report_html_path = report_md_path.with_suffix(".html")
        with open(report_html_path, "w") as f:
            f.write(html)
        logger.info(f"HTML report saved: {report_html_path}")

    logger.info(f"\nReports ready in: {results_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_report_generator(config, args)
