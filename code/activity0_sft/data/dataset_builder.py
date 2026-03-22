"""
activity0_sft/data/dataset_builder.py
======================================
Dataset download, formatting, and preparation pipeline for Activity 0 SFT.

This module handles:
1. Downloading datasets from HuggingFace Hub
2. Converting them to ChatML format (required by Llama-3.1-Instruct)
3. Applying the tokenizer's chat template
4. Splitting into train/validation sets
5. Caching processed datasets to disk for fast re-use

LEARNING NOTE:
The single most important concept here is the ChatML format and the distinction
between which tokens contribute to the loss. We use SFTTrainer's
"response_template" mechanism to mask out system/user tokens — only the
assistant's response tokens are used in the gradient update.

Why this matters: without masking, the model would be trained to predict
"<|start_header_id|>user<|end_header_id|>" tokens, which would make it
learn to mimic the user format — not useful. We only want it to learn
to generate assistant responses.

Usage:
    from activity0_sft.data.dataset_builder import build_dataset
    train_ds, eval_ds = build_dataset(config, tokenizer)
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# The system prompt injected into every training example.
# This tells the model what persona it should adopt.
# It's important to be consistent — use the same system prompt at inference.
DEFAULT_CYBERSEC_SYSTEM_PROMPT = (
    "You are an expert cybersecurity analyst with deep knowledge of "
    "vulnerabilities, threat intelligence, MITRE ATT&CK techniques, network "
    "security, and incident response. Provide accurate, detailed, and "
    "actionable security analysis."
)

# Minimum number of characters in a response.
# Filters out degenerate examples like "Yes." or "No." that don't teach anything.
MIN_RESPONSE_LENGTH = 50

# Maximum number of characters we allow before truncation at the data level.
# The tokenizer will also enforce max_seq_length, but this is a cheap pre-filter
# to avoid tokenizing obviously-too-long examples.
MAX_EXAMPLE_CHARS = 8000


# =============================================================================
# Primary dataset: Trendyol Cybersecurity
# =============================================================================

def load_trendyol_dataset(
    hf_dataset_id: str,
    split: str,
    instruction_column: str,
    response_column: str,
    system_prompt: str,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Load and format the Trendyol cybersecurity instruction-tuning dataset.

    The Trendyol dataset has two relevant columns:
    - instruction: The user's question or task (e.g., "What is SQL injection?")
    - output: The expected assistant response

    We convert each row into a list of message dicts (the format that
    tokenizer.apply_chat_template() expects):
    [
        {"role": "system", "content": "You are a cybersecurity analyst..."},
        {"role": "user", "content": "What is SQL injection?"},
        {"role": "assistant", "content": "SQL injection is..."}
    ]

    Args:
        hf_dataset_id: HuggingFace dataset identifier
        split: Dataset split to load (typically "train")
        instruction_column: Name of the column containing user instructions
        response_column: Name of the column containing assistant responses
        system_prompt: System message to prepend to every example
        cache_dir: Directory to cache the downloaded dataset

    Returns:
        Dataset with a "messages" column containing formatted message lists
    """
    logger.info(f"Loading Trendyol dataset: {hf_dataset_id} [{split}]")

    # Load from HuggingFace Hub.
    # The first call downloads ~50MB; subsequent calls use the local cache.
    raw_dataset = load_dataset(
        hf_dataset_id,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=False,  # Security: never trust arbitrary code from Hub
    )

    logger.info(
        f"Trendyol dataset loaded: {len(raw_dataset)} examples. "
        f"Columns: {raw_dataset.column_names}"
    )

    # Validate that the expected columns exist.
    # This is a defensive check — dataset structure can change on HuggingFace.
    required_columns = {instruction_column, response_column}
    missing = required_columns - set(raw_dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset {hf_dataset_id} is missing expected columns: {missing}. "
            f"Available columns: {raw_dataset.column_names}"
        )

    # Apply format conversion to each row.
    # Using batched=True is faster because it processes multiple rows at once.
    def format_trendyol_example(examples: dict[str, list]) -> dict[str, list]:
        """
        Convert a batch of raw rows into the ChatML message format.

        The 'examples' dict has column names as keys, each mapped to a list of
        values (one per example in the batch). We process all examples at once
        for efficiency.
        """
        formatted_messages = []
        for instruction, response in zip(
            examples[instruction_column], examples[response_column]
        ):
            # Skip degenerate examples
            if not instruction or not response:
                continue
            if len(str(response).strip()) < MIN_RESPONSE_LENGTH:
                continue
            if len(str(instruction)) + len(str(response)) > MAX_EXAMPLE_CHARS:
                continue

            # Build the message list.
            # CRITICAL: The order is [system, user, assistant].
            # Changing this order will break the model's instruction-following.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(instruction).strip()},
                {"role": "assistant", "content": str(response).strip()},
            ]
            formatted_messages.append(messages)

        return {"messages": formatted_messages}

    formatted_dataset = raw_dataset.map(
        format_trendyol_example,
        batched=True,
        batch_size=1000,
        remove_columns=raw_dataset.column_names,  # Drop original columns
        desc="Formatting Trendyol dataset",
    )

    # Filter out any None/empty messages that slipped through
    formatted_dataset = formatted_dataset.filter(
        lambda x: x["messages"] is not None and len(x["messages"]) == 3
    )

    logger.info(
        f"Trendyol formatted: {len(formatted_dataset)} examples "
        f"(filtered from {len(raw_dataset)})"
    )
    return formatted_dataset


# =============================================================================
# Secondary dataset: CVE Records
# =============================================================================

def load_cve_dataset(
    hf_dataset_id: str,
    split: str,
    messages_column: str,
    sample_n: Optional[int],
    sample_seed: int,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Load and format the CVE records training dataset.

    This dataset already uses a chat format (list of messages), but we still
    need to validate and normalize it to match the Trendyol format.

    The dataset structure is typically:
    {
        "conversations": [
            {"from": "human", "value": "Tell me about CVE-2021-44228"},
            {"from": "gpt", "value": "CVE-2021-44228, known as Log4Shell..."}
        ]
    }

    Note the different role labels: "human" / "gpt" vs "user" / "assistant".
    We normalize these to the HuggingFace standard ("user" / "assistant").

    Args:
        hf_dataset_id: HuggingFace dataset identifier
        split: Dataset split to load
        messages_column: Column name containing the conversation messages
        sample_n: If set, randomly sample this many examples from the dataset
        sample_seed: Random seed for reproducible sampling
        cache_dir: Directory to cache the downloaded dataset

    Returns:
        Dataset with a "messages" column in normalized ChatML format
    """
    logger.info(f"Loading CVE dataset: {hf_dataset_id} [{split}]")

    raw_dataset = load_dataset(
        hf_dataset_id,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )

    logger.info(f"CVE dataset loaded: {len(raw_dataset)} examples")

    # Subsample before processing — no point formatting examples we won't use
    if sample_n is not None and sample_n < len(raw_dataset):
        logger.info(f"Sampling {sample_n} examples from {len(raw_dataset)} CVE records")
        indices = random.Random(sample_seed).sample(range(len(raw_dataset)), sample_n)
        raw_dataset = raw_dataset.select(indices)

    # Role name normalization map
    # Different datasets use different naming conventions for roles
    ROLE_MAP = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
        "bot": "assistant",
    }

    def format_cve_example(examples: dict[str, list]) -> dict[str, list]:
        """
        Normalize CVE conversation format to standard ChatML messages.

        We:
        1. Add a cybersecurity system prompt if one isn't already present
        2. Normalize role names (human → user, gpt → assistant)
        3. Filter out malformed conversations
        """
        formatted_messages = []
        for conversation in examples[messages_column]:
            try:
                if not conversation or not isinstance(conversation, list):
                    continue

                # Build normalized message list
                messages = []

                # Check if first message is already a system message
                has_system = (
                    len(conversation) > 0
                    and conversation[0].get("from", conversation[0].get("role", "")) == "system"
                )

                if not has_system:
                    # Inject our standard cybersecurity system prompt
                    messages.append({
                        "role": "system",
                        "content": DEFAULT_CYBERSEC_SYSTEM_PROMPT,
                    })

                for msg in conversation:
                    # Handle both "from"/"value" and "role"/"content" formats
                    raw_role = msg.get("from") or msg.get("role", "")
                    content = msg.get("value") or msg.get("content", "")

                    normalized_role = ROLE_MAP.get(raw_role.lower(), None)
                    if normalized_role is None:
                        # Unknown role — skip this entire conversation
                        messages = []
                        break

                    if not content or len(str(content).strip()) < 10:
                        continue

                    messages.append({
                        "role": normalized_role,
                        "content": str(content).strip(),
                    })

                # Valid conversation must have at least: [system, user, assistant]
                if len(messages) < 3:
                    continue

                # Ensure the conversation ends with an assistant message
                # (we don't want to train on incomplete exchanges)
                if messages[-1]["role"] != "assistant":
                    continue

                # Check response length
                assistant_content = messages[-1]["content"]
                if len(assistant_content) < MIN_RESPONSE_LENGTH:
                    continue

                formatted_messages.append(messages)

            except (KeyError, TypeError, AttributeError) as e:
                # Malformed example — skip silently
                logger.debug(f"Skipping malformed CVE example: {e}")
                continue

        return {"messages": formatted_messages}

    formatted_dataset = raw_dataset.map(
        format_cve_example,
        batched=True,
        batch_size=500,
        remove_columns=raw_dataset.column_names,
        desc="Formatting CVE dataset",
    )

    formatted_dataset = formatted_dataset.filter(
        lambda x: x["messages"] is not None and len(x["messages"]) >= 3
    )

    logger.info(f"CVE formatted: {len(formatted_dataset)} examples after filtering")
    return formatted_dataset


# =============================================================================
# Apply chat template and tokenize
# =============================================================================

def apply_chat_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """
    Convert the "messages" column into a single formatted text string using
    the tokenizer's chat template.

    After this step, each example has a "text" column containing the full
    ChatML-formatted sequence like:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a cybersecurity analyst...<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        What is a CVE?<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        A CVE (Common Vulnerabilities and Exposures)...<|eot_id|>

    The SFTTrainer from trl will then:
    1. Tokenize these text strings
    2. Identify the "assistant" portions and compute loss only on those tokens
    3. Pack multiple short sequences into one context window (if packing=True)

    Args:
        dataset: Dataset with "messages" column
        tokenizer: Llama-3.1-8B-Instruct tokenizer

    Returns:
        Dataset with "text" column containing formatted strings
    """
    def apply_template(examples: dict[str, list]) -> dict[str, list]:
        texts = []
        for messages in examples["messages"]:
            # apply_chat_template handles all the special token insertion.
            # tokenize=False because SFTTrainer handles tokenization internally.
            # add_generation_prompt=False because we're including the full
            # response (not generating at inference time).
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(
        apply_template,
        batched=True,
        batch_size=1000,
        desc="Applying chat template",
    )

    return dataset


# =============================================================================
# Main entry point
# =============================================================================

def build_dataset(
    config: Any,  # OmegaConf DictConfig
    tokenizer: PreTrainedTokenizerBase,
    cache_dir: Optional[str] = None,
) -> tuple[Dataset, Dataset]:
    """
    Build the complete training and validation datasets for Activity 0 SFT.

    This is the main function called by the training script. It:
    1. Loads and formats the Trendyol primary dataset
    2. Loads and formats the CVE secondary dataset
    3. Concatenates them into a single combined dataset
    4. Applies the chat template to produce formatted text strings
    5. Splits into train (95%) and validation (5%) sets
    6. Optionally saves to disk cache for future runs

    Args:
        config: OmegaConf config loaded from config.yaml
        tokenizer: Initialized tokenizer (must be Llama-3.1-8B-Instruct)
        cache_dir: Override cache directory (uses config value if None)

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    cache_dir = cache_dir or config.datasets.get("cache_dir", "/tmp/loraforge_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load primary dataset (Trendyol)
    # ------------------------------------------------------------------
    primary_cfg = config.datasets.primary
    trendyol_ds = load_trendyol_dataset(
        hf_dataset_id=primary_cfg.hf_dataset_id,
        split=primary_cfg.split,
        instruction_column=primary_cfg.instruction_column,
        response_column=primary_cfg.response_column,
        system_prompt=primary_cfg.get("system_prompt", DEFAULT_CYBERSEC_SYSTEM_PROMPT),
        cache_dir=cache_dir,
    )

    # ------------------------------------------------------------------
    # Step 2: Load secondary dataset (CVE records)
    # ------------------------------------------------------------------
    secondary_cfg = config.datasets.secondary
    cve_ds = load_cve_dataset(
        hf_dataset_id=secondary_cfg.hf_dataset_id,
        split=secondary_cfg.split,
        messages_column=secondary_cfg.messages_column,
        sample_n=secondary_cfg.get("sample_n", None),
        sample_seed=secondary_cfg.get("sample_seed", 42),
        cache_dir=cache_dir,
    )

    # ------------------------------------------------------------------
    # Step 3: Combine datasets
    # ------------------------------------------------------------------
    # concatenate_datasets merges two datasets with the same schema.
    # After combining, we shuffle to interleave examples from both sources.
    # This prevents the model from learning in "phases" (all Trendyol first,
    # then all CVE) which can cause instability.
    combined_ds = concatenate_datasets([trendyol_ds, cve_ds])
    combined_ds = combined_ds.shuffle(seed=42)

    logger.info(
        f"Combined dataset: {len(combined_ds)} total examples "
        f"({len(trendyol_ds)} Trendyol + {len(cve_ds)} CVE)"
    )

    # ------------------------------------------------------------------
    # Step 4: Apply chat template
    # ------------------------------------------------------------------
    combined_ds = apply_chat_template(combined_ds, tokenizer)

    # ------------------------------------------------------------------
    # Step 5: Train/validation split
    # ------------------------------------------------------------------
    val_split = config.datasets.get("validation_split", 0.05)
    split_dataset = combined_ds.train_test_split(
        test_size=val_split,
        seed=42,
        shuffle=True,
    )
    train_ds = split_dataset["train"]
    eval_ds = split_dataset["test"]

    logger.info(
        f"Final split — Train: {len(train_ds)} examples, "
        f"Eval: {len(eval_ds)} examples "
        f"({val_split*100:.0f}% held out)"
    )

    # ------------------------------------------------------------------
    # Step 6: Log a sample for manual inspection
    # ------------------------------------------------------------------
    _log_sample(train_ds)

    return train_ds, eval_ds


def _log_sample(dataset: Dataset, n: int = 2) -> None:
    """
    Log a few formatted examples so you can visually verify the data format.

    This is one of the most useful debugging steps — actually look at what
    you're training on. Many bugs (wrong format, missing tokens, truncated
    responses) are immediately visible here.
    """
    logger.info("=" * 60)
    logger.info(f"Sample training examples (first {n}):")
    for i in range(min(n, len(dataset))):
        text = dataset[i]["text"]
        logger.info(f"\n--- Example {i+1} ---")
        logger.info(text[:800] + "..." if len(text) > 800 else text)
    logger.info("=" * 60)


def get_dataset_statistics(dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> dict:
    """
    Compute statistics about the dataset: token length distribution.

    Call this after build_dataset() to understand your data:
    - What is the average sequence length? (affects VRAM usage)
    - What fraction of sequences exceed max_seq_length? (those get truncated)
    - Are there any sequences that are suspiciously long or short?

    Args:
        dataset: Formatted dataset with "text" column
        tokenizer: Initialized tokenizer

    Returns:
        Dict with statistics: mean_length, p50, p90, p99, max_length, etc.
    """
    import numpy as np

    logger.info("Computing dataset token length statistics...")

    # Sample at most 1000 examples for speed (full tokenization takes minutes)
    sample_size = min(1000, len(dataset))
    sample = dataset.select(range(sample_size))

    lengths = []
    for example in sample:
        tokens = tokenizer(example["text"], add_special_tokens=False)
        lengths.append(len(tokens["input_ids"]))

    lengths = np.array(lengths)
    stats = {
        "num_examples": len(dataset),
        "sample_size": sample_size,
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(np.min(lengths)),
        "p50_length": int(np.percentile(lengths, 50)),
        "p90_length": int(np.percentile(lengths, 90)),
        "p99_length": int(np.percentile(lengths, 99)),
        "max_length": int(np.max(lengths)),
    }

    logger.info(
        f"Token length stats (n={sample_size}): "
        f"mean={stats['mean_length']:.0f}, "
        f"p50={stats['p50_length']}, "
        f"p90={stats['p90_length']}, "
        f"p99={stats['p99_length']}, "
        f"max={stats['max_length']}"
    )

    return stats


if __name__ == "__main__":
    """
    Quick test: run this script directly to verify dataset loading works
    without launching the full training pipeline.

    Usage:
        python -m activity0_sft.data.dataset_builder

    This will load 100 examples from each dataset and print samples.
    Useful for debugging data format issues without waiting for training to start.
    """
    import sys
    from transformers import AutoTokenizer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("Loading tokenizer for format test...")
    # Use a local model if HF_TOKEN is not set, or provide the real model
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        token=None,  # Set HF_TOKEN environment variable
    )

    print("\nTesting Trendyol dataset formatting (100 examples)...")
    ds = load_trendyol_dataset(
        hf_dataset_id="Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
        split="train[:100]",  # Only 100 examples for the test
        instruction_column="instruction",
        response_column="output",
        system_prompt=DEFAULT_CYBERSEC_SYSTEM_PROMPT,
    )
    ds = apply_chat_template(ds, tokenizer)
    print(f"✓ Trendyol: {len(ds)} examples formatted")
    print("\nFirst example:")
    print(ds[0]["text"][:600])
    print("...")
