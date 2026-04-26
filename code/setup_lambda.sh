#!/usr/bin/env bash
# ============================================================================
# LoraForge — Lambda Cloud Environment Setup
# ============================================================================
#
# PURPOSE: Reproducible one-shot setup for any Lambda Cloud GPU instance.
#   Installs all Python dependencies and verifies the setup is correct.
#   Does NOT write env vars — you set those yourself (see printed instructions at end).
#
# USAGE:
#   # Step 1: Create and activate a virtual environment
#   python -m venv nvenv
#   source nvenv/bin/activate
#
#   # Step 2: Run this script
#   bash code/setup_lambda.sh
#
#   # Step 3: Set env vars manually — copy the echo commands printed at the end
#   # into your terminal, then: source ~/.bashrc
#
# WHAT THIS SCRIPT DOES:
#   1. Upgrade pip (avoids "legacy resolver" warnings that can cause bad installs)
#   2. Install PyTorch with CUDA 12.4 wheels from the official index
#   3. Install the rest of requirements.txt (non-torch packages)
#   4. Install nvidia-modelopt separately (needs --extra-index-url for NVIDIA registry)
#   5. Create cache directories
#   6. Run a self-check to confirm everything is importable
#   7. Print the env var commands for you to run manually
#
# EXPECTED WALL TIME: ~10-15 minutes (downloading ~15 GB of packages)
# ============================================================================

set -euo pipefail   # Exit on error, unset var, pipe failure — fail fast and loudly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

# ── Color helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

log_info()    { echo -e "${BLUE}[setup]${NC} $*"; }
log_success() { echo -e "${GREEN}[setup]${NC} ✓ $*"; }
log_warn()    { echo -e "${YELLOW}[setup]${NC} ⚠ $*"; }
log_error()   { echo -e "${RED}[setup]${NC} ✗ $*"; }

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            LoraForge — Lambda Cloud Environment Setup        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Prerequisite check ───────────────────────────────────────────────────────
log_info "Checking prerequisites..."

# Must be inside the virtual environment
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    log_error "Not inside a virtual environment."
    log_error "Run: python -m venv nvenv && source nvenv/bin/activate"
    log_error "Then re-run: bash code/setup_lambda.sh"
    exit 1
fi
log_success "Virtual env active: ${VIRTUAL_ENV}"

# Must have CUDA
if ! command -v nvidia-smi &>/dev/null; then
    log_error "nvidia-smi not found. Is this a GPU instance?"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
log_success "GPU: ${GPU_NAME} | CUDA: ${CUDA_VER}"

# ── Step 1: Upgrade pip ──────────────────────────────────────────────────────
log_info "Step 1/5: Upgrading pip..."
# Why: Old pip uses a legacy dependency resolver that can install incompatible packages.
# Newer pip's resolver backtracks properly and respects pinned constraints.
pip install --upgrade pip setuptools wheel --quiet
log_success "pip upgraded"

# ── Step 2: Install PyTorch with CUDA wheels ──────────────────────────────────
log_info "Step 2/5: Installing PyTorch with CUDA 12.4 wheels..."
# Why: PyTorch on PyPI ships CPU-only wheels. The CUDA build is only on
# the PyTorch index (download.pytorch.org). Must specify --index-url explicitly.
# cuda124 = CUDA 12.4 (matches Lambda's driver; CUDA is backward compatible).
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet
log_success "PyTorch 2.4.0 + CUDA 12.4 installed"

# ── Step 3: Install core requirements ────────────────────────────────────────
log_info "Step 3/5: Installing requirements.txt (excluding torch, modelopt)..."
# Why: Install the rest of the stack from requirements.txt. We exclude torch (already
# installed above) and nvidia-modelopt (needs NVIDIA index, installed below).
# The --extra-index-url for vLLM covers its CUDA-specific pre-built wheels.
pip install -r "${REQUIREMENTS}" \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --ignore-installed torch torchvision torchaudio \
    --quiet
log_success "Core requirements installed"

# ── Step 4: Install nvidia-modelopt ──────────────────────────────────────────
log_info "Step 4/5: Installing nvidia-modelopt from NVIDIA index..."
# Why: nvidia-modelopt is NVIDIA's quantization and TRT-LLM export toolkit.
# It's hosted on NVIDIA's own PyPI mirror (pypi.ngc.nvidia.com), not on PyPI.
# Installing without --extra-index-url will give a "package not found" error.
pip install "nvidia-modelopt[torch]==0.17.0" \
    --extra-index-url https://pypi.ngc.nvidia.com \
    --quiet
log_success "nvidia-modelopt 0.17.0 installed"

# ── Step 5: Create cache directories ─────────────────────────────────────────
log_info "Step 5/6: Creating cache directories..."
# Why: HuggingFace and Torch write large model files to these paths.
# Pre-creating them avoids permission errors on first download.
mkdir -p "${HOME}/loraforge_cache/hf/hub"
mkdir -p "${HOME}/loraforge_cache/hf/datasets"
mkdir -p "${HOME}/loraforge_cache/torch"
mkdir -p "${HOME}/loraforge_cache/modelopt"
log_success "Cache directories created at ~/loraforge_cache/"

# ── Self-check ───────────────────────────────────────────────────────────────
log_info "Step 6/6 (pre-check): Importing key packages..."

python - << 'PYCHECK'
import sys
checks = [
    ("torch", "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'  torch {torch.__version__} — CUDA {torch.version.cuda}')"),
    ("transformers", "import transformers; print(f'  transformers {transformers.__version__}')"),
    ("peft", "import peft; print(f'  peft {peft.__version__}')"),
    ("datasets", "import datasets; print(f'  datasets {datasets.__version__}')"),
    ("vllm", "import vllm; print(f'  vllm {vllm.__version__}')"),
    ("nvidia_modelopt", "import modelopt; print(f'  nvidia-modelopt {modelopt.__version__}')"),
    ("omegaconf", "import omegaconf; print(f'  omegaconf {omegaconf.__version__}')"),
    ("pandas", "import pandas; print(f'  pandas {pandas.__version__}')"),
    ("rich", "import rich; print(f'  rich {rich.__version__}')"),
]

failures = []
for name, check in checks:
    try:
        exec(check)
    except Exception as e:
        print(f"  FAIL {name}: {e}", file=sys.stderr)
        failures.append(name)

if failures:
    print(f"\nFailed imports: {failures}", file=sys.stderr)
    sys.exit(1)
PYCHECK

log_success "All key packages importable"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Packages installed!                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  ─────────────────────────────────────────────────────────────"
echo "  STEP 6/6 — Set environment variables manually"
echo "  Copy the block below into your terminal."
echo "  Replace hf_xxxx with your real HuggingFace token."
echo "  ─────────────────────────────────────────────────────────────"
echo ""
echo "  # Paste into terminal:"
echo "  echo 'export HF_HOME=\"\${HOME}/loraforge_cache/hf\"' >> ~/.bashrc"
echo "  echo 'export TRANSFORMERS_CACHE=\"\${HOME}/loraforge_cache/hf/hub\"' >> ~/.bashrc"
echo "  echo 'export HF_DATASETS_CACHE=\"\${HOME}/loraforge_cache/hf/datasets\"' >> ~/.bashrc"
echo "  echo 'export TORCH_HOME=\"\${HOME}/loraforge_cache/torch\"' >> ~/.bashrc"
echo "  echo 'export MODELOPT_CACHE=\"\${HOME}/loraforge_cache/modelopt\"' >> ~/.bashrc"
echo "  echo 'export CUDA_VISIBLE_DEVICES=\"0\"' >> ~/.bashrc"
echo "  echo 'export CUBLAS_WORKSPACE_CONFIG=\":4096:8\"' >> ~/.bashrc"
echo "  echo 'export TORCH_ALLOW_TF32=\"1\"' >> ~/.bashrc"
echo "  echo 'export USE_TF=0' >> ~/.bashrc"
echo "  echo 'export USE_JAX=0' >> ~/.bashrc"
echo "  echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc"
echo "  echo 'export HF_TOKEN=\"hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx\"' >> ~/.bashrc"
echo "  source ~/.bashrc"
echo ""
echo "  ─────────────────────────────────────────────────────────────"
echo "  Variable reference:"
echo "    HF_HOME / TRANSFORMERS_CACHE / HF_DATASETS_CACHE"
echo "      → Where HF downloads models and datasets (~15–30 GB)"
echo "      → Must point to a large disk, not the OS root partition"
echo "    TORCH_HOME / MODELOPT_CACHE"
echo "      → PyTorch hub cache + ModelOpt calibration cache"
echo "    CUDA_VISIBLE_DEVICES=0"
echo "      → Single-GPU activities. Change to '0,1' for multi-GPU."
echo "    CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo "      → Deterministic CUDA matmul (needed for reproducible quant calibration)"
echo "    TORCH_ALLOW_TF32=1"
echo "      → Enables TF32 on Ampere+ (~2x matmul speed, negligible precision loss)"
echo "    USE_TF=0  USE_JAX=0"
echo "      → Prevents Transformers from importing TensorFlow/JAX backends"
echo "      → Without these, 'import transformers' triggers slow TF/JAX detection"
echo "    TOKENIZERS_PARALLELISM=false"
echo "      → Prevents tokenizer fork warnings in multi-worker DataLoader"
echo "    HF_TOKEN"
echo "      → Your HuggingFace API token (huggingface.co/settings/tokens)"
echo "      → Required for gated models (Llama-3.1-8B-Instruct)"
echo "  ─────────────────────────────────────────────────────────────"
echo ""
