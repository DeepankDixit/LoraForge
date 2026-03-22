"""
activity1_baseline/vllm_server.py
===================================
Launch a vLLM server exposing an OpenAI-compatible API for the merged FP16 model.

What this script does:
  1. Reads config.yaml for model path and server settings
  2. Validates that the merged model exists (run merge.py first)
  3. Constructs the vLLM CLI command with the correct flags
  4. Launches vLLM as a subprocess, tailing its logs
  5. Waits for the server to become healthy (polls /health endpoint)
  6. Reports the server URL and GPU memory stats on startup

After this script reports "Server is ready", you can:
  - Send requests: python -m activity1_baseline.benchmark.client
  - Test manually: curl http://localhost:8000/v1/models
  - Use OpenAI Python client: openai.OpenAI(base_url="http://localhost:8000/v1")

The vLLM startup sequence (see Concept 04, Part 7):
  1. Load 16GB FP16 weights from disk to GPU HBM (~30–60s)
  2. Compile CUDA graphs for common batch sizes/sequence lengths (~30–90s first time)
  3. Initialize PagedAttention KV cache pages (~5s)
  4. Start HTTP server on the configured port

Run:
  python -m activity1_baseline.vllm_server
  # Ctrl+C to stop
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)

# Minimum vLLM version required for features used in this project
MIN_VLLM_VERSION = (0, 5, 0)


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch vLLM server for Activity 1 baseline")
    parser.add_argument("--config", type=str, default="./activity1_baseline/config.yaml")
    parser.add_argument("--model", type=str, help="Override merged_model_path from config")
    parser.add_argument("--port", type=int, help="Override server.port from config")
    parser.add_argument("--no-wait", action="store_true",
                        help="Launch server and return immediately (don't wait for ready)")
    return parser.parse_args()


# ── Version Check ─────────────────────────────────────────────────────────────

def check_vllm_version() -> None:
    """Verify vLLM is installed and meets the minimum version requirement."""
    try:
        import vllm
        version_str = vllm.__version__
        version_parts = tuple(int(x) for x in version_str.split(".")[:3] if x.isdigit())
        if version_parts < MIN_VLLM_VERSION:
            logger.warning(
                f"vLLM version {version_str} may be outdated. "
                f"Recommended: >={'.'.join(str(v) for v in MIN_VLLM_VERSION)}"
            )
        else:
            logger.info(f"vLLM version: {version_str} ✓")
    except ImportError:
        raise ImportError(
            "vLLM is not installed. Install with:\n"
            "  pip install vllm>=0.5.0\n"
            "Note: vLLM requires a CUDA-capable GPU."
        )


# ── Model Validation ─────────────────────────────────────────────────────────

def validate_merged_model(model_path: str) -> None:
    """Check that the merged model is ready before starting the server."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Merged model not found: {model_path}\n"
            f"Run 'python -m activity1_baseline.merge.merge' first."
        )
    # Check for required files
    has_config = (path / "config.json").exists()
    has_weights = any(f.suffix == ".safetensors" for f in path.iterdir())
    has_tokenizer = (path / "tokenizer.json").exists() or (path / "tokenizer_config.json").exists()

    if not (has_config and has_weights and has_tokenizer):
        raise ValueError(
            f"Merged model directory seems incomplete: {model_path}\n"
            f"  config.json: {'✓' if has_config else '✗'}\n"
            f"  *.safetensors: {'✓' if has_weights else '✗'}\n"
            f"  tokenizer: {'✓' if has_tokenizer else '✗'}\n"
            f"Re-run merge.py with --force to regenerate."
        )
    logger.info(f"Merged model validated: {model_path} ✓")


# ── Build vLLM Command ────────────────────────────────────────────────────────

def build_vllm_command(config: dict, model_path: str, port: int) -> list[str]:
    """
    Construct the vLLM server launch command.

    vLLM is launched via its CLI entrypoint, which internally calls the
    FastAPI server with PagedAttention scheduling.

    The OpenAI-compatible endpoint will be at:
      POST http://localhost:{port}/v1/chat/completions
      GET  http://localhost:{port}/v1/models
      GET  http://localhost:{port}/health
    """
    server_cfg = config.get("server", {})

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",

        "--model", model_path,
        # ↑ Path to the merged model (standard HuggingFace checkpoint directory)

        "--host", server_cfg.get("host", "0.0.0.0"),
        "--port", str(port),

        "--dtype", server_cfg.get("dtype", "float16"),
        # ↑ Weight precision. Must match what merge.py saved.
        # "float16" = standard FP16 (Activity 1 baseline)
        # "float8"  = FP8 (Activity 2 will change this)
        # "auto"    = vLLM infers from model config (risky — be explicit)

        "--max-model-len", str(server_cfg.get("max_model_len", 4096)),
        # ↑ Maximum total sequence length (prompt + generation).
        # Lower values → more KV cache pages available → higher concurrency.

        "--gpu-memory-utilization", str(server_cfg.get("gpu_memory_utilization", 0.90)),
        # ↑ Fraction of GPU VRAM to use. 0.90 = 90% on A10G = 21.6GB.
        # vLLM allocates: model weights first, then uses remaining for KV cache pages.

        "--max-num-seqs", str(server_cfg.get("max_num_seqs", 64)),
        # ↑ Maximum simultaneous requests in the scheduler.
        # Our benchmark max concurrency is 32 — set this higher so we don't
        # artificially throttle the benchmark.

        "--tensor-parallel-size", str(server_cfg.get("tensor_parallel_size", 1)),
        # ↑ Number of GPUs to shard the model across. 1 = single GPU (A10G).
        # Increase for multi-GPU setups: 4 × A100s → tensor_parallel_size=4.

        "--trust-remote-code",
        # ↑ Required for some models. Llama-3.1 doesn't strictly need it,
        # but included for compatibility.

        "--served-model-name", "cybersec-analyst-fp16",
        # ↑ The name clients use in the "model" field of API requests.
        # Cleaner than a long file path.
    ]

    return cmd


# ── Health Check ─────────────────────────────────────────────────────────────

def wait_for_server(base_url: str, timeout_seconds: int = 120) -> bool:
    """
    Poll the vLLM /health endpoint until the server is ready or timeout expires.

    Returns True if the server is ready, False if timeout was reached.
    """
    health_url = f"{base_url}/health"
    logger.info(f"Waiting for server at {health_url} (timeout: {timeout_seconds}s)...")

    start = time.perf_counter()
    attempt = 0

    while time.perf_counter() - start < timeout_seconds:
        attempt += 1
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                elapsed = time.perf_counter() - start
                logger.info(f"Server is ready! (took {elapsed:.1f}s, {attempt} attempts)")
                return True
        except requests.exceptions.ConnectionError:
            pass  # Server not up yet — keep polling
        except requests.exceptions.Timeout:
            pass  # Server started but health endpoint slow — keep polling

        if attempt % 10 == 0:
            elapsed = time.perf_counter() - start
            logger.info(f"  Still waiting... ({elapsed:.0f}s elapsed)")

        time.sleep(2)

    logger.error(f"Server failed to start within {timeout_seconds}s")
    return False


# ── GPU Memory Report ─────────────────────────────────────────────────────────

def report_server_stats(base_url: str) -> None:
    """
    Query the vLLM server for model info and log GPU memory stats.

    This is useful for confirming the FP16 model loaded correctly and
    how much KV cache space is available.
    """
    try:
        models_resp = requests.get(f"{base_url}/v1/models", timeout=10)
        if models_resp.status_code == 200:
            models = models_resp.json().get("data", [])
            for model in models:
                logger.info(f"  Serving model: {model.get('id')} (max_model_len={model.get('max_model_len')})")
    except Exception as e:
        logger.warning(f"Could not fetch model info: {e}")

    # Try to get GPU memory info via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().split("\n")):
                used, free, total = [int(x.strip()) for x in line.split(",")]
                logger.info(f"  GPU {i}: {used/1024:.1f}GB used / {total/1024:.1f}GB total "
                            f"({free/1024:.1f}GB free for KV cache pages)")
    except Exception:
        logger.info("  (nvidia-smi not available — cannot report GPU memory)")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_server(config: dict, args: argparse.Namespace) -> subprocess.Popen:
    """
    Launch the vLLM server as a subprocess and wait for it to be ready.

    Returns the subprocess handle (useful for programmatic teardown).
    """
    model_path = args.model or config.get("merged_model_path", "./outputs/cybersec_analyst_merged")
    port = args.port or config.get("server", {}).get("port", 8000)
    base_url = f"http://localhost:{port}"
    timeout = config.get("server", {}).get("startup_timeout_seconds", 120)

    check_vllm_version()
    validate_merged_model(model_path)

    cmd = build_vllm_command(config, model_path, port)
    logger.info(f"Launching vLLM server:")
    logger.info(f"  {' '.join(cmd)}")
    logger.info("")
    logger.info("Expected startup timeline:")
    logger.info("  ~30–60s: Loading 16GB weights from disk to GPU HBM")
    logger.info("  ~30–90s: Compiling CUDA graphs (first run only — cached afterward)")
    logger.info("  ~5s:     Initializing PagedAttention KV cache pages")
    logger.info("  → Server ready")
    logger.info("")

    # Launch vLLM — inherits stdout/stderr so logs are visible
    process = subprocess.Popen(cmd)

    if args.no_wait:
        logger.info("--no-wait specified. Server launched in background.")
        return process

    # Wait for the server to become healthy
    ready = wait_for_server(base_url, timeout)

    if not ready:
        process.terminate()
        raise RuntimeError(
            f"vLLM server failed to start within {timeout}s. "
            f"Check logs above for CUDA/OOM errors."
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  vLLM server is READY at {base_url}")
    logger.info(f"  Model endpoint: {base_url}/v1/chat/completions")
    logger.info(f"  OpenAI model name: cybersec-analyst-fp16")
    logger.info(f"  Next: python -m activity1_baseline.benchmark.client")
    logger.info("=" * 60)
    logger.info("")

    report_server_stats(base_url)

    return process


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    process = run_server(config, args)

    # Keep running until Ctrl+C
    def shutdown(sig, frame):
        logger.info("Shutting down vLLM server...")
        process.terminate()
        process.wait()
        logger.info("Server stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("Server running. Press Ctrl+C to stop.")
    process.wait()
