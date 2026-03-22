"""
activity0_sft/evaluation/eval.py
==================================
Post-training evaluation for Activity 0: SFT adapter quality assessment.

This module provides two types of evaluation:
1. Quantitative: perplexity on a held-out cybersecurity test set
2. Qualitative: side-by-side comparison of base vs fine-tuned model responses
   on hand-crafted cybersecurity prompts

LEARNING NOTE: Perplexity is NOT a perfect metric for instruction-following
quality. A model can have low perplexity (high probability assigned to the
test set tokens) while still producing unhelpful or inaccurate responses.

The gold standard is human evaluation (expensive) or using a judge LLM
(GPT-4 or Claude) to score responses on helpfulness, accuracy, and detail.
For LoraForge, we use perplexity as a quick sanity check + manual inspection.

After Activity 1 (baseline deployment), we'll add more rigorous benchmarks:
- MMLU-CS (cybersecurity subset of MMLU benchmark)
- CyberSecEval (Meta's safety + capability benchmark for cybersecurity LLMs)

Usage:
    python -m activity0_sft.evaluation.eval --adapter-path ./outputs/cybersec_analyst_lora
    python -m activity0_sft.evaluation.eval --compare-base  (runs base model side-by-side)
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import typer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)
app = typer.Typer(help="Activity 0: Evaluate the trained cybersecurity LoRA adapter")


# =============================================================================
# Perplexity computation
# =============================================================================

def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int = 2048,
    batch_size: int = 4,
    device: str = "cuda",
) -> float:
    """
    Compute average perplexity over a list of text sequences.

    Perplexity = exp(average cross-entropy loss per token)
    = exp(-(1/N) × Σ log P(token_i | context))

    Lower perplexity = model assigns higher probability to the correct tokens
    = better fit to the data distribution.

    For reference:
    - Random model: perplexity ≈ vocab_size ≈ 128,256 (for Llama-3)
    - Llama-3.1-8B base on general text: perplexity ≈ 5-10
    - Fine-tuned model on domain text: should decrease vs base model

    Typical results you should expect:
    - Base model on cybersecurity QA: perplexity ≈ 15-25
    - Fine-tuned adapter on cybersecurity QA: perplexity ≈ 8-15
    - A 20-40% reduction in perplexity is a healthy SFT result

    Args:
        model: The language model (base or PEFT-wrapped)
        tokenizer: Corresponding tokenizer
        texts: List of formatted text strings (output of apply_chat_template)
        max_length: Truncate sequences longer than this
        batch_size: How many sequences to process at once
        device: "cuda" or "cpu"

    Returns:
        Average perplexity over all input texts
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0
    num_batches = math.ceil(len(texts) / batch_size)

    logger.info(f"Computing perplexity over {len(texts)} examples in {num_batches} batches...")

    with torch.no_grad():
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx : batch_idx + batch_size]

            # Tokenize the batch
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,        # Pad to the longest sequence in the batch
                return_attention_mask=True,
            )

            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            # For perplexity, labels = input_ids (shifted by 1 internally in the model).
            # The model computes loss = cross_entropy(logits[:-1], input_ids[1:]).
            # We set padding positions to -100 to exclude them from the loss.
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Mask padding tokens

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # outputs.loss is the MEAN cross-entropy loss over non-masked tokens.
            # We weight by the number of valid (non-padding) tokens to compute
            # a proper average across batches of different lengths.
            num_valid_tokens = (labels != -100).sum().item()
            batch_loss = outputs.loss.item() * num_valid_tokens

            total_loss += batch_loss
            total_tokens += num_valid_tokens

            if (batch_idx // batch_size + 1) % 10 == 0:
                logger.info(
                    f"  Batch {batch_idx // batch_size + 1}/{num_batches} | "
                    f"Running loss: {total_loss/total_tokens:.4f}"
                )

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    logger.info(f"Perplexity: {perplexity:.2f} (avg_loss={avg_loss:.4f} over {total_tokens} tokens)")
    return perplexity


# =============================================================================
# Qualitative evaluation: generate responses and compare
# =============================================================================

# Curated prompts that test specific cybersecurity knowledge domains.
# Chosen to cover: CVE/vulnerability analysis, MITRE ATT&CK, incident response,
# network security, and general security reasoning.
EVAL_PROMPTS = [
    {
        "category": "vulnerability_analysis",
        "prompt": "Explain CVE-2021-44228 (Log4Shell). What makes it particularly dangerous and what was the immediate mitigation?",
    },
    {
        "category": "mitre_attack",
        "prompt": "What is MITRE ATT&CK technique T1059.001 (PowerShell)? Describe how an attacker uses it and what defenders should monitor.",
    },
    {
        "category": "incident_response",
        "prompt": "You receive an alert: unusual outbound traffic from a workstation to IP 185.220.101.45 on port 443. Walk me through your initial triage steps.",
    },
    {
        "category": "network_security",
        "prompt": "What is the difference between a stateful and stateless firewall? When would you choose each?",
    },
    {
        "category": "concepts",
        "prompt": "Explain the difference between authentication and authorization in the context of a zero-trust architecture.",
    },
    {
        "category": "threat_intelligence",
        "prompt": "What is the MITRE ATT&CK framework and how is it different from the Lockheed Martin Cyber Kill Chain?",
    },
]

CYBERSEC_SYSTEM_PROMPT = (
    "You are an expert cybersecurity analyst with deep knowledge of "
    "vulnerabilities, threat intelligence, MITRE ATT&CK techniques, network "
    "security, and incident response. Provide accurate, detailed, and "
    "actionable security analysis."
)


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    system_prompt: str = CYBERSEC_SYSTEM_PROMPT,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    device: str = "cuda",
) -> str:
    """
    Generate a response from the model for a given prompt.

    Uses greedy-ish decoding (low temperature=0.1) for reproducibility.
    At temperature=0, decoding is fully deterministic (greedy).
    At temperature=1, sampling is proportional to the model's raw logits.

    For evaluation purposes, we want low temperature so results are
    reproducible across runs. At inference time (deployment), temperature
    might be higher for more creative responses.

    Args:
        model: The language model to generate from
        tokenizer: Corresponding tokenizer
        prompt: User's question/prompt
        system_prompt: System context to prepend
        max_new_tokens: Maximum tokens to generate in the response
        temperature: Sampling temperature (0 = greedy, 1 = full sampling)
        device: Target device

    Returns:
        Generated response text (assistant turn only, not including prompt)
    """
    # Format using ChatML template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # apply_chat_template with add_generation_prompt=True adds the
    # "<|start_header_id|>assistant<|end_header_id|>" suffix, which tells
    # the model to start generating the assistant's response.
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # DIFFERENT from training — we want the model to generate
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            # Repetition penalty reduces the probability of tokens that have
            # already appeared in the output. Prevents repetitive responses.
            repetition_penalty=1.1,
            # Stop generating when we hit the EOS token
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens (not the input prompt)
    response_ids = output_ids[0][input_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response.strip()


def run_qualitative_evaluation(
    fine_tuned_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_model: Optional[AutoModelForCausalLM] = None,
    prompts: Optional[list[dict]] = None,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> list[dict]:
    """
    Run qualitative evaluation: generate responses for each eval prompt.

    If base_model is provided, generates responses from both models and
    returns them side-by-side for easy comparison.

    Args:
        fine_tuned_model: The QLoRA-adapted model
        tokenizer: Tokenizer
        base_model: Optional base model (Llama-3.1-8B-Instruct without adapter)
        prompts: List of evaluation prompts (defaults to EVAL_PROMPTS)
        max_new_tokens: Max tokens to generate per response
        device: Target device

    Returns:
        List of dicts with prompt, category, fine_tuned_response, base_response (if provided)
    """
    if prompts is None:
        prompts = EVAL_PROMPTS

    results = []
    for i, eval_item in enumerate(prompts):
        prompt = eval_item["prompt"]
        category = eval_item["category"]
        logger.info(f"\n[{i+1}/{len(prompts)}] Category: {category}")
        logger.info(f"Prompt: {prompt[:100]}...")

        # Generate from fine-tuned model
        logger.info("Generating response from fine-tuned model...")
        ft_response = generate_response(
            fine_tuned_model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
        )
        logger.info(f"Fine-tuned response:\n{ft_response[:300]}...")

        result = {
            "category": category,
            "prompt": prompt,
            "fine_tuned_response": ft_response,
        }

        # Optionally generate from base model for comparison
        if base_model is not None:
            logger.info("Generating response from base model...")
            base_response = generate_response(
                base_model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
            )
            result["base_response"] = base_response
            logger.info(f"Base model response:\n{base_response[:300]}...")

        results.append(result)

    return results


# =============================================================================
# Main evaluation runner
# =============================================================================

def run_evaluation(
    adapter_path: Path,
    base_model_id: str,
    hf_token: Optional[str] = None,
    compare_base: bool = False,
    output_path: Optional[Path] = None,
    device: str = "cuda",
) -> dict:
    """
    Load the adapter and run both quantitative and qualitative evaluation.

    Args:
        adapter_path: Path to the saved LoRA adapter directory
        base_model_id: HuggingFace model ID of the base model
        hf_token: HuggingFace token for gated models
        compare_base: If True, also run evaluation on base model for comparison
        output_path: Path to save evaluation results JSON
        device: Target device

    Returns:
        Dict containing all evaluation results
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    logger.info(f"Loading tokenizer from adapter: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Load base model (in BF16 for evaluation — not QLoRA)
    # ------------------------------------------------------------------
    # For evaluation, we load in FP16/BF16 (not 4-bit) because:
    # 1. We're not training, so we don't need the memory savings of 4-bit
    # 2. Full precision gives more accurate perplexity measurements
    # 3. If you're on a GPU with limited VRAM, you can still load 4-bit for eval
    logger.info(f"Loading base model: {base_model_id}")
    logger.info("(For evaluation, loading in BF16 — not 4-bit. May require 16GB VRAM)")

    # If VRAM is limited, use 4-bit for eval too
    if torch.cuda.is_available():
        free_vram = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()
        ) / 1e9
        logger.info(f"Free VRAM: {free_vram:.1f} GB")

        if free_vram < 18:
            logger.info("Less than 18GB free VRAM — loading in 4-bit for evaluation")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            torch_dtype = torch.bfloat16
        else:
            bnb_config = None
            torch_dtype = torch.bfloat16
    else:
        bnb_config = None
        torch_dtype = torch.float32

    base_model_raw = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch_dtype,
        token=hf_token,
        trust_remote_code=False,
    )

    # ------------------------------------------------------------------
    # Load LoRA adapter on top of base model
    # ------------------------------------------------------------------
    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    fine_tuned_model = PeftModel.from_pretrained(
        base_model_raw,
        str(adapter_path),
        is_trainable=False,  # Evaluation only — no gradients needed
    )
    fine_tuned_model.eval()

    # ------------------------------------------------------------------
    # Optionally load a separate base model for comparison
    # (requires enough VRAM to hold both models simultaneously — only on A100/H100)
    # ------------------------------------------------------------------
    comparison_model = None
    if compare_base:
        logger.warning(
            "Loading a second model for comparison requires ~32GB VRAM. "
            "Only do this on A100 or H100. Skipping on smaller GPUs."
        )
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 35e9:
            comparison_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                token=hf_token,
            )
        else:
            logger.warning(
                "Not enough VRAM for side-by-side comparison. "
                "Run evaluation sequentially (evaluate one model, then swap)."
            )

    # ------------------------------------------------------------------
    # Qualitative evaluation
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("QUALITATIVE EVALUATION")
    logger.info("=" * 60)

    qualitative_results = run_qualitative_evaluation(
        fine_tuned_model=fine_tuned_model,
        tokenizer=tokenizer,
        base_model=comparison_model,
        device=device,
    )

    # ------------------------------------------------------------------
    # Print comparison in a readable format
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 60)

    for result in qualitative_results:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Category: {result['category'].upper()}")
        logger.info(f"Prompt: {result['prompt']}")
        logger.info(f"\n[FINE-TUNED MODEL]:\n{result['fine_tuned_response']}")
        if "base_response" in result:
            logger.info(f"\n[BASE MODEL]:\n{result['base_response']}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    evaluation_results = {
        "adapter_path": str(adapter_path),
        "base_model_id": base_model_id,
        "evaluation_timestamp": time.time(),
        "duration_seconds": time.time() - start_time,
        "qualitative": qualitative_results,
        "instructions": (
            "To compare models: look for responses where the fine-tuned model "
            "gives more specific, domain-appropriate answers. "
            "Key improvements to look for: (1) uses correct security terminology, "
            "(2) structures responses with actionable steps, "
            "(3) references specific standards (MITRE ATT&CK, CVE format, CVSS)."
        ),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        logger.info(f"\nEvaluation results saved to: {output_path}")

    return evaluation_results


# =============================================================================
# CLI entry point
# =============================================================================

@app.command()
def main(
    adapter_path: Path = typer.Option(
        Path("./outputs/cybersec_analyst_lora"),
        "--adapter-path",
        help="Path to the saved LoRA adapter directory",
    ),
    base_model_id: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct",
        "--base-model",
        help="HuggingFace model ID of the base model",
    ),
    compare_base: bool = typer.Option(
        False,
        "--compare-base",
        help="Also generate responses from the base model for comparison (requires more VRAM)",
    ),
    output_path: Optional[Path] = typer.Option(
        Path("./outputs/activity0_eval_results.json"),
        "--output",
        "-o",
        help="Path to save evaluation results JSON",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
    ),
) -> None:
    """
    Evaluate the trained cybersecurity LoRA adapter.

    Runs qualitative evaluation (response generation) on a set of
    cybersecurity prompts. Optionally compares against the base model.

    After running, inspect the saved JSON to see if the fine-tuned model
    gives more detailed, domain-appropriate responses.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set — will use cached model if available")

    if not adapter_path.exists():
        logger.error(
            f"Adapter directory not found: {adapter_path}. "
            "Run training first: python -m activity0_sft.training.qlora_trainer"
        )
        raise typer.Exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No GPU detected — generation will be very slow on CPU.")

    run_evaluation(
        adapter_path=adapter_path,
        base_model_id=base_model_id,
        hf_token=hf_token,
        compare_base=compare_base,
        output_path=output_path,
        device=device,
    )


if __name__ == "__main__":
    app()
