"""
activity1_baseline/benchmark/client.py
========================================
Benchmark the vLLM server by sweeping concurrency levels, prompt lengths,
and output lengths. Measures TTFT and throughput for each configuration.

Benchmark design:
  - For each (concurrency, prompt_len, output_len) combination:
      1. Send `warmup_requests` first (discarded — warms up CUDA graphs and
         PagedAttention page allocations so first-request cold-start doesn't bias results)
      2. Send `num_requests_per_config` timed requests at the given concurrency
      3. Measure per-request TTFT using streaming
      4. Measure aggregate throughput (tokens/s across all concurrent requests)
  - Results are written to baseline_results.json after each config completes
    (crash-safe — partial results are not lost)

TTFT measurement requires streaming (see Concept 04, Part 5):
  - Without streaming: client waits for full response → TTFT = total latency
  - With streaming: we time from request send to first chunk received

Concurrency is implemented with asyncio — each "concurrent" request is an
async task. Using asyncio (not threads) avoids thread overhead and matches
how production inference clients work.

Run:
  # Server must be running first: python -m activity1_baseline.vllm_server
  python -m activity1_baseline.benchmark.client
  python -m activity1_baseline.benchmark.client --quick   # Reduced sweep for testing
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    """Result for a single HTTP request."""
    success: bool
    ttft_seconds: Optional[float]      # Time to first token (None if failed)
    total_seconds: Optional[float]     # Total request duration
    output_tokens: Optional[int]       # Number of tokens generated
    error: Optional[str] = None        # Error message if success=False


@dataclass
class ConfigResult:
    """Aggregated results for one (concurrency, prompt_len, output_len) configuration."""
    concurrency: int
    prompt_len_tokens: int
    output_len_tokens: int
    num_requests: int
    num_successful: int

    # TTFT statistics (seconds)
    ttft_p50_seconds: Optional[float]
    ttft_p95_seconds: Optional[float]
    ttft_p99_seconds: Optional[float]
    ttft_mean_seconds: Optional[float]

    # Throughput statistics
    throughput_tokens_per_sec: Optional[float]   # Output tokens/sec across all requests
    requests_per_sec: Optional[float]            # Completed requests per second


@dataclass
class BenchmarkResults:
    """Full benchmark results written to baseline_results.json."""
    model_name: str
    server_url: str
    benchmark_timestamp: str
    benchmark_duration_seconds: float
    configs: list[ConfigResult] = field(default_factory=list)


# ── Prompt Generation ─────────────────────────────────────────────────────────

def generate_prompt(target_tokens: int) -> str:
    """
    Generate a cybersecurity-themed prompt of approximately `target_tokens` tokens.

    We use a template that ensures the model produces substantive output.
    The prompt length is controlled by padding with context sentences.

    Approximation: ~1.3 tokens per word for English text.
    """
    base_prompts = [
        "Explain the CVE-2021-44228 vulnerability (Log4Shell) in detail, including the attack vector, affected systems, and remediation steps.",
        "What is the MITRE ATT&CK framework? Describe how it is used in threat intelligence and incident response.",
        "Describe the phases of a typical cyber kill chain and what defensive measures can be applied at each phase.",
        "Explain the difference between symmetric and asymmetric encryption. Provide examples of protocols that use each.",
        "What is lateral movement in a network intrusion? Describe three specific techniques an attacker might use and how to detect them.",
        "Explain how SSL/TLS certificate validation works and what happens during the handshake process.",
        "What are the key differences between a vulnerability assessment and a penetration test?",
        "Describe the OWASP Top 10 web application security risks. Which three do you consider most critical and why?",
    ]

    # Select a prompt that fits the target length
    # For longer prompts, we add context to pad to the target length
    import hashlib
    # Deterministic selection based on target_tokens for reproducibility
    idx = int(hashlib.md5(str(target_tokens).encode()).hexdigest(), 16) % len(base_prompts)
    base = base_prompts[idx]

    # Estimate current token count (approximation: 4 chars per token)
    current_chars = len(base)
    target_chars = target_tokens * 4

    if current_chars >= target_chars:
        return base

    # Pad with relevant cybersecurity context to reach target length
    padding_sentences = [
        "Consider the context of enterprise security operations, threat hunting, and incident response. ",
        "Focus your analysis on practical implications for security engineers working in SOC environments. ",
        "Include specific examples from real-world incidents where possible. ",
        "Consider how this applies to both on-premises and cloud-based infrastructure. ",
        "Address how automation and SIEM tools can help with detection and response. ",
        "Include relevant CVE numbers, MITRE ATT&CK technique IDs, or industry standards where applicable. ",
        "Consider the perspective of both the attacker and the defender in your analysis. ",
        "Discuss how this threat landscape has evolved over the past five years. ",
    ]

    padded = base + " "
    i = 0
    while len(padded) < target_chars:
        padded += padding_sentences[i % len(padding_sentences)]
        i += 1

    return padded[:target_chars]


# ── Async Request Client ──────────────────────────────────────────────────────

async def send_streaming_request(
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.1,
    request_timeout: float = 120.0,
) -> RequestResult:
    """
    Send a single streaming request and measure TTFT + total duration.

    Uses httpx for async HTTP with streaming support.
    We do NOT use the OpenAI client library here because it doesn't expose
    the raw streaming chunks we need for precise TTFT measurement.
    """
    import httpx

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,   # ← Required for TTFT measurement
    }

    request_start = time.perf_counter()
    first_token_time = None
    output_chars = 0

    try:
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    return RequestResult(
                        success=False,
                        ttft_seconds=None,
                        total_seconds=None,
                        output_tokens=None,
                        error=f"HTTP {response.status_code}: {error_text[:200]}",
                    )

                # Process SSE (Server-Sent Events) stream
                # Each line is: "data: {json_chunk}" or "data: [DONE]"
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    line_data = line[6:]   # Strip "data: " prefix
                    if line_data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(line_data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                                # ↑ TTFT is measured HERE — the moment we receive
                                # the first non-empty content token
                            output_chars += len(content)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue  # Skip malformed chunks

    except httpx.TimeoutException:
        return RequestResult(
            success=False,
            ttft_seconds=None,
            total_seconds=None,
            output_tokens=None,
            error=f"Request timed out after {request_timeout}s",
        )
    except Exception as e:
        return RequestResult(
            success=False,
            ttft_seconds=None,
            total_seconds=None,
            output_tokens=None,
            error=str(e),
        )

    total_time = time.perf_counter() - request_start
    ttft = (first_token_time - request_start) if first_token_time else None
    # Approximate output tokens (1 token ≈ 4 chars for English)
    output_tokens = max(1, output_chars // 4)

    return RequestResult(
        success=True,
        ttft_seconds=ttft,
        total_seconds=total_time,
        output_tokens=output_tokens,
    )


# ── Concurrency Runner ────────────────────────────────────────────────────────

async def run_concurrent_requests(
    base_url: str,
    model_name: str,
    concurrency: int,
    num_requests: int,
    prompt_len: int,
    output_len: int,
    temperature: float,
    request_timeout: float,
    semaphore: asyncio.Semaphore,
) -> list[RequestResult]:
    """
    Run `num_requests` requests with `concurrency` simultaneous requests.

    Uses a semaphore to cap the number of concurrent in-flight requests.
    This simulates a real-world client that has a fixed worker pool.
    """
    prompt = generate_prompt(prompt_len)

    async def single_request() -> RequestResult:
        async with semaphore:
            return await send_streaming_request(
                base_url, model_name, prompt, output_len, temperature, request_timeout
            )

    tasks = [asyncio.create_task(single_request()) for _ in range(num_requests)]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_config_stats(
    results: list[RequestResult],
    concurrency: int,
    prompt_len: int,
    output_len: int,
    wall_time_seconds: float,
) -> ConfigResult:
    """Compute aggregate statistics from a list of request results."""
    successful = [r for r in results if r.success and r.ttft_seconds is not None]
    failed = len(results) - len(successful)

    if failed > 0:
        logger.warning(f"  {failed}/{len(results)} requests failed")

    if not successful:
        return ConfigResult(
            concurrency=concurrency,
            prompt_len_tokens=prompt_len,
            output_len_tokens=output_len,
            num_requests=len(results),
            num_successful=0,
            ttft_p50_seconds=None, ttft_p95_seconds=None,
            ttft_p99_seconds=None, ttft_mean_seconds=None,
            throughput_tokens_per_sec=None, requests_per_sec=None,
        )

    ttft_values = sorted(r.ttft_seconds for r in successful)
    total_output_tokens = sum(r.output_tokens for r in successful if r.output_tokens)

    def percentile(data, p):
        idx = max(0, int(len(data) * p / 100) - 1)
        return data[idx]

    throughput = total_output_tokens / wall_time_seconds if wall_time_seconds > 0 else 0
    req_per_sec = len(successful) / wall_time_seconds if wall_time_seconds > 0 else 0

    return ConfigResult(
        concurrency=concurrency,
        prompt_len_tokens=prompt_len,
        output_len_tokens=output_len,
        num_requests=len(results),
        num_successful=len(successful),
        ttft_p50_seconds=percentile(ttft_values, 50),
        ttft_p95_seconds=percentile(ttft_values, 95),
        ttft_p99_seconds=percentile(ttft_values, 99),
        ttft_mean_seconds=statistics.mean(ttft_values),
        throughput_tokens_per_sec=throughput,
        requests_per_sec=req_per_sec,
    )


# ── Main Benchmark Orchestrator ───────────────────────────────────────────────

async def run_benchmark(config: dict, args: argparse.Namespace) -> BenchmarkResults:
    """
    Sweep all (concurrency, prompt_len, output_len) configurations and collect results.
    """
    import datetime
    import httpx

    server_cfg = config.get("server", {})
    bench_cfg = config.get("benchmark", {})
    report_cfg = config.get("reporting", {})

    base_url = bench_cfg.get("api_base_url", "http://localhost:8000")
    model_name = "cybersec-analyst-fp16"
    output_dir = Path(report_cfg.get("output_dir", "./results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config sweep
    concurrency_levels = bench_cfg.get("concurrency_levels", [1, 4, 8, 16])
    prompt_lengths = bench_cfg.get("prompt_lengths_tokens", [128, 512, 2048])
    output_lengths = bench_cfg.get("output_lengths_tokens", [128, 256])
    num_requests = bench_cfg.get("num_requests_per_config", 20)
    warmup_requests = bench_cfg.get("warmup_requests", 5)
    request_timeout = bench_cfg.get("request_timeout_seconds", 120)
    temperature = bench_cfg.get("temperature", 0.1)

    if args.quick:
        # Quick mode: reduced sweep for testing
        concurrency_levels = [1, 4]
        prompt_lengths = [128, 512]
        output_lengths = [64]
        num_requests = 5
        warmup_requests = 2
        logger.info("QUICK MODE: Using reduced sweep")

    # Verify server is up
    logger.info(f"Checking server health at {base_url}...")
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{base_url}/health")
            if resp.status_code != 200:
                raise RuntimeError(f"Server health check failed: {resp.status_code}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach vLLM server at {base_url}.\n"
                f"Start it with: python -m activity1_baseline.vllm_server\n"
                f"Error: {e}"
            )
    logger.info("Server is healthy ✓")

    results = BenchmarkResults(
        model_name=model_name,
        server_url=base_url,
        benchmark_timestamp=datetime.datetime.now().isoformat(),
        benchmark_duration_seconds=0,
    )

    total_configs = len(concurrency_levels) * len(prompt_lengths) * len(output_lengths)
    config_num = 0
    benchmark_start = time.perf_counter()

    for concurrency in concurrency_levels:
        for prompt_len in prompt_lengths:
            for output_len in output_lengths:
                config_num += 1
                logger.info(
                    f"\n[{config_num}/{total_configs}] "
                    f"concurrency={concurrency}, prompt={prompt_len}tok, output={output_len}tok"
                )

                semaphore = asyncio.Semaphore(concurrency)

                # ── Warmup ────────────────────────────────────────────────
                if warmup_requests > 0:
                    logger.info(f"  Warmup: {warmup_requests} requests (discarded)...")
                    await run_concurrent_requests(
                        base_url, model_name, concurrency, warmup_requests,
                        prompt_len, output_len, temperature, request_timeout, semaphore
                    )

                # ── Timed benchmark ───────────────────────────────────────
                logger.info(f"  Benchmark: {num_requests} requests...")
                wall_start = time.perf_counter()

                request_results = await run_concurrent_requests(
                    base_url, model_name, concurrency, num_requests,
                    prompt_len, output_len, temperature, request_timeout, semaphore
                )

                wall_time = time.perf_counter() - wall_start

                # ── Compute stats ─────────────────────────────────────────
                config_result = compute_config_stats(
                    request_results, concurrency, prompt_len, output_len, wall_time
                )
                results.configs.append(config_result)

                if config_result.ttft_p50_seconds:
                    logger.info(
                        f"  ✓ TTFT P50={config_result.ttft_p50_seconds*1000:.0f}ms  "
                        f"P95={config_result.ttft_p95_seconds*1000:.0f}ms  "
                        f"Throughput={config_result.throughput_tokens_per_sec:.0f} tok/s"
                    )

                # ── Save intermediate results (crash-safe) ────────────────
                intermediate_path = output_dir / "benchmark_in_progress.json"
                with open(intermediate_path, "w") as f:
                    json.dump(
                        {**asdict(results), "configs": [asdict(c) for c in results.configs]},
                        f, indent=2, default=str
                    )

    results.benchmark_duration_seconds = time.perf_counter() - benchmark_start

    # Save final results
    final_path = output_dir / report_cfg.get("baseline_results_filename", "baseline_results.json")
    with open(final_path, "w") as f:
        json.dump(
            {**asdict(results), "configs": [asdict(c) for c in results.configs]},
            f, indent=2, default=str
        )

    logger.info(f"\nBenchmark complete in {results.benchmark_duration_seconds:.0f}s")
    logger.info(f"Results saved to: {final_path}")

    return results


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Activity 1 vLLM baseline server")
    parser.add_argument("--config", type=str, default="./activity1_baseline/config.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="Run a reduced benchmark sweep (for testing)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    asyncio.run(run_benchmark(config, args))
