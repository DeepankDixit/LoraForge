"""
Cost estimator for LoraForge.
Estimates cost per 1M tokens on major cloud GPU providers.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class GPUPricing:
    """Pricing for a specific GPU instance."""
    gpu_name: str
    provider: str
    cost_per_hour_usd: float
    vram_gb: float
    memory_bandwidth_tb_s: float  # Useful for estimating decode throughput


# GPU pricing as of early 2026 (Lambda Cloud / RunPod)
GPU_PRICING: Dict[str, GPUPricing] = {
    "a10_24gb": GPUPricing("NVIDIA A10 24GB", "Lambda Cloud", 0.75, 24, 0.6),
    "a100_40gb": GPUPricing("NVIDIA A100 40GB SXM", "Lambda Cloud", 1.50, 40, 1.555),
    "a100_80gb": GPUPricing("NVIDIA A100 80GB SXM", "Lambda Cloud", 2.50, 80, 2.0),
    "h100_80gb": GPUPricing("NVIDIA H100 80GB SXM", "Lambda Cloud", 3.50, 80, 3.35),
    "rtx_4090": GPUPricing("NVIDIA RTX 4090 24GB", "RunPod", 0.74, 24, 1.008),
    "t4_16gb": GPUPricing("NVIDIA T4 16GB", "Google Cloud", 0.35, 16, 0.32),
}


@dataclass
class CostEstimate:
    """Cost estimate for serving a model at a given throughput."""
    gpu_name: str
    provider: str
    quantization_format: str
    throughput_tokens_per_sec: float
    cost_per_hour_usd: float
    cost_per_million_tokens_usd: float
    tokens_per_dollar: float
    monthly_cost_at_1b_tokens_usd: float

    def summary(self) -> str:
        return (
            f"{self.gpu_name} ({self.quantization_format}): "
            f"${self.cost_per_million_tokens_usd:.3f}/1M tokens | "
            f"${self.monthly_cost_at_1b_tokens_usd:.0f}/month @ 1B tokens"
        )


def estimate_cost(
    throughput_tokens_per_sec: float,
    gpu_key: str,
    quantization_format: str = "fp16",
) -> Optional[CostEstimate]:
    """
    Estimate serving cost given measured throughput and GPU type.

    Args:
        throughput_tokens_per_sec: Measured throughput from benchmark (tokens/sec).
        gpu_key: GPU identifier from GPU_PRICING dict (e.g., "a100_40gb").
        quantization_format: Human-readable format label (e.g., "INT4 AWQ").

    Returns:
        CostEstimate with per-token and monthly cost, or None if gpu_key invalid.
    """
    pricing = GPU_PRICING.get(gpu_key)
    if pricing is None:
        return None

    tokens_per_hour = throughput_tokens_per_sec * 3600
    cost_per_million = (pricing.cost_per_hour_usd / tokens_per_hour) * 1_000_000
    tokens_per_dollar = tokens_per_hour / pricing.cost_per_hour_usd
    monthly_cost_1b = (1_000_000_000 / tokens_per_hour) * pricing.cost_per_hour_usd

    return CostEstimate(
        gpu_name=pricing.gpu_name,
        provider=pricing.provider,
        quantization_format=quantization_format,
        throughput_tokens_per_sec=throughput_tokens_per_sec,
        cost_per_hour_usd=pricing.cost_per_hour_usd,
        cost_per_million_tokens_usd=cost_per_million,
        tokens_per_dollar=tokens_per_dollar,
        monthly_cost_at_1b_tokens_usd=monthly_cost_1b,
    )


def compare_formats(
    format_throughputs: Dict[str, float],
    gpu_key: str = "a100_40gb",
) -> Dict[str, CostEstimate]:
    """
    Compare cost across quantization formats for the same GPU.

    Args:
        format_throughputs: Dict mapping format name to measured throughput (tok/s).
            e.g., {"FP16": 1200, "FP8": 2000, "INT4 AWQ": 3000}
        gpu_key: GPU to compare on.

    Returns:
        Dict mapping format name to CostEstimate.
    """
    results = {}
    for fmt, throughput in format_throughputs.items():
        estimate = estimate_cost(throughput, gpu_key, fmt)
        if estimate:
            results[fmt] = estimate
    return results
