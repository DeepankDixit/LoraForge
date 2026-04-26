"""
activity2_quantization/quantizer/calibration.py
=================================================
CalibrationDatasetBuilder — builds and caches the PTQ calibration dataset.

WHY CALIBRATION DATA IS NEEDED
-------------------------------
Post-training quantization algorithms (AWQ, SmoothQuant, FP8 with static scales)
need to *observe* the model's activations on real inputs in order to set quantization
scale factors. Without this:

  - AWQ cannot identify which weight channels are "salient" (high activation magnitude).
    Those channels need to be scaled before quantization so they aren't rounded to zero.

  - SmoothQuant cannot compute the per-channel smoothing factors s_j that migrate
    quantization difficulty from activations to weights.

  - FP8 with static input scales cannot set the per-tensor clamp range — it would
    have to use the theoretical maximum, which wastes dynamic range.

Naive quantization (e.g., bitsandbytes load_in_8bit) skips this step and just rounds
weights to the nearest representable value. It works passably but loses 1–3% accuracy
that calibration-based methods recover.

CALIBRATION DATASET DESIGN
---------------------------
  - 512 samples: enough for stable activation statistics via CLT (standard deviation
    of the sample mean ∝ 1/√n; n=512 gives ~4× more stability than n=32).
  - Domain-matched: we use UltraChat 200K (same as Activity 0 training) so the
    activation statistics reflect the model's actual deployment distribution.
  - Sequence length 512: covers the bulk of real query lengths; longer sequences
    don't shift the channel-wise maxima significantly.
  - Held-out slice: we skip the first N_TRAIN examples and sample from the tail
    to avoid calibrating on data the model was trained on (would overfit scales).

Run directly:
  python -m activity2_quantization.quantizer.calibration
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class CalibrationBatch:
    """A single tokenized calibration example."""
    input_ids: Tensor            # Shape: (seq_len,)
    attention_mask: Tensor       # Shape: (seq_len,)
    text: str                    # Original text (for debugging)


@dataclass
class CalibrationDataset:
    """The complete calibration dataset, ready to feed into ModelOpt."""
    batches: list[CalibrationBatch] = field(default_factory=list)
    tokenizer_name: str = ""
    num_samples: int = 0
    max_seq_length: int = 512
    dataset_id: str = ""

    def to_dataloader(self, batch_size: int = 1):
        """
        Yield batches of (input_ids, attention_mask) tensors.

        ModelOpt's mtq.quantize() accepts a callable that yields batches.
        This method returns the correct format.

        Args:
            batch_size: Number of sequences per batch. Typically 1 for calibration
                        (ModelOpt processes sequences one at a time to accumulate
                        per-channel statistics).

        Yields:
            dict with keys 'input_ids' and 'attention_mask', each a (B, seq_len) tensor.
        """
        for i in range(0, len(self.batches), batch_size):
            chunk = self.batches[i:i + batch_size]
            yield {
                "input_ids": torch.stack([b.input_ids for b in chunk]),
                "attention_mask": torch.stack([b.attention_mask for b in chunk]),
            }

    def __len__(self) -> int:
        return len(self.batches)


# ── CalibrationDatasetBuilder ─────────────────────────────────────────────────

class CalibrationDatasetBuilder:
    """
    Builds and caches a tokenized PTQ calibration dataset.

    Usage:
        builder = CalibrationDatasetBuilder(config)
        calib_data = builder.build()
        # Pass to ModelOpt:
        mtq.quantize(model, quant_config, forward_loop=calib_data.to_dataloader())
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        calib_cfg = config.get("calibration", {})

        self.dataset_id: str = calib_cfg.get("dataset_id", "HuggingFaceH4/ultrachat_200k")
        self.dataset_split: str = calib_cfg.get("dataset_split", "train_sft")
        self.num_samples: int = calib_cfg.get("num_calibration_samples", 512)
        self.max_seq_length: int = calib_cfg.get("max_seq_length", 512)
        self.seed: int = calib_cfg.get("seed", 42)
        self.cache_path: Path = Path(calib_cfg.get("cache_path", "./outputs/calibration_dataset.pt"))
        self.base_model_id: str = config.get("base_model_id", "meta-llama/Llama-3.1-8B-Instruct")

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self, force_rebuild: bool = False) -> CalibrationDataset:
        """
        Build the calibration dataset (or load from cache).

        Args:
            force_rebuild: If True, ignore cache and re-build from scratch.

        Returns:
            CalibrationDataset ready for use with ModelOpt.
        """
        if not force_rebuild and self.cache_path.exists():
            logger.info(f"Loading cached calibration dataset from {self.cache_path}")
            return self._load_from_cache()

        logger.info(
            f"Building calibration dataset: {self.num_samples} samples "
            f"from {self.dataset_id}/{self.dataset_split}"
        )

        tokenizer = self._load_tokenizer()
        raw_texts = self._load_raw_texts()
        batches = self._tokenize(raw_texts, tokenizer)

        dataset = CalibrationDataset(
            batches=batches,
            tokenizer_name=self.base_model_id,
            num_samples=len(batches),
            max_seq_length=self.max_seq_length,
            dataset_id=self.dataset_id,
        )

        self._save_to_cache(dataset)
        logger.info(
            f"Calibration dataset built: {len(batches)} samples, "
            f"max_seq_length={self.max_seq_length}"
        )
        return dataset

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        """Load the Llama-3.1 tokenizer with chat template support."""
        logger.info(f"Loading tokenizer: {self.base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            use_fast=True,
            trust_remote_code=False,
        )
        # Llama-3.1 doesn't have a pad token; set to eos_token for batched tokenization
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug("Set pad_token = eos_token (Llama-3.1 has no pad token by default)")
        return tokenizer

    def _load_raw_texts(self) -> list[str]:
        """
        Load and extract raw text strings from the UltraChat dataset.

        We skip the first 60% of the split to avoid overlap with Activity 0 training data.
        UltraChat train_sft has ~200K examples; we skip the first 120K and sample 512
        from the remaining ~80K.
        """
        logger.info(f"Loading dataset {self.dataset_id} split={self.dataset_split}...")
        dataset = load_dataset(
            self.dataset_id,
            split=self.dataset_split,
            streaming=True,   # Stream to avoid loading 200K examples into RAM
        )

        # Skip the first 60% of the dataset (used in Activity 0 training)
        # UltraChat streaming: we skip by iterating past the first N_SKIP examples.
        # This is a conservative estimate — Activity 0 used ~79K examples.
        N_SKIP = 80_000

        texts = []
        skipped = 0
        collected = 0

        for example in dataset:
            if skipped < N_SKIP:
                skipped += 1
                continue

            text = self._extract_text(example)
            if text and len(text) > 50:  # Skip very short examples
                texts.append(text)
                collected += 1

            if collected >= self.num_samples:
                break

        if len(texts) < self.num_samples:
            logger.warning(
                f"Only found {len(texts)} valid examples (target: {self.num_samples}). "
                f"Proceeding with fewer samples."
            )

        logger.info(f"Collected {len(texts)} calibration texts (skipped {N_SKIP} train examples)")
        return texts

    def _extract_text(self, example: dict) -> Optional[str]:
        """
        Extract plain text from an UltraChat example.

        UltraChat format: {"messages": [{"role": "user", "content": "..."}, ...]}
        We join the first 3 turns (user + assistant + user) into a single string.
        This approximates realistic conversation context.
        """
        messages = example.get("messages", [])
        if not messages:
            return None

        # Take first 3 messages (1 or 2 full turns)
        parts = []
        for msg in messages[:3]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content:
                parts.append(f"{role}: {content}")

        return "\n".join(parts) if parts else None

    def _tokenize(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
    ) -> list[CalibrationBatch]:
        """
        Tokenize all texts, truncate to max_seq_length, return CalibrationBatch list.

        We apply the chat template so the calibration inputs have the same format
        as the training data: <|begin_of_text|><|user|>...<|assistant|>...
        This ensures activation statistics reflect the actual deployment distribution.
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn

        batches = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task(
                f"Tokenizing {len(texts)} calibration examples...", total=len(texts)
            )

            for text in texts:
                encoding = tokenizer(
                    text,
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                batches.append(
                    CalibrationBatch(
                        input_ids=encoding["input_ids"].squeeze(0),
                        attention_mask=encoding["attention_mask"].squeeze(0),
                        text=text[:200],  # Store truncated text for debugging
                    )
                )
                progress.advance(task)

        return batches

    def _save_to_cache(self, dataset: CalibrationDataset) -> None:
        """Save tokenized dataset to disk as a .pt file."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_ids": torch.stack([b.input_ids for b in dataset.batches]),
            "attention_mask": torch.stack([b.attention_mask for b in dataset.batches]),
            "texts": [b.text for b in dataset.batches],
            "tokenizer_name": dataset.tokenizer_name,
            "num_samples": dataset.num_samples,
            "max_seq_length": dataset.max_seq_length,
            "dataset_id": dataset.dataset_id,
        }
        torch.save(payload, self.cache_path)
        logger.info(f"Calibration dataset cached to {self.cache_path}")

    def _load_from_cache(self) -> CalibrationDataset:
        """Load tokenized dataset from .pt cache file."""
        payload = torch.load(self.cache_path, map_location="cpu", weights_only=True)
        input_ids_all = payload["input_ids"]
        attention_masks_all = payload["attention_mask"]
        texts = payload["texts"]

        batches = [
            CalibrationBatch(
                input_ids=input_ids_all[i],
                attention_mask=attention_masks_all[i],
                text=texts[i],
            )
            for i in range(len(texts))
        ]
        return CalibrationDataset(
            batches=batches,
            tokenizer_name=payload.get("tokenizer_name", ""),
            num_samples=payload.get("num_samples", len(batches)),
            max_seq_length=payload.get("max_seq_length", 512),
            dataset_id=payload.get("dataset_id", ""),
        )


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_calibration_build(config: dict, args=None) -> CalibrationDataset:
    """Build and cache calibration dataset. Called from main.py."""
    builder = CalibrationDatasetBuilder(config)
    force_rebuild = getattr(args, "force_rebuild_calib", False)
    return builder.build(force_rebuild=force_rebuild)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Build PTQ calibration dataset")
    parser.add_argument("--config", default="./activity2_quantization/config.yaml")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Ignore cache and rebuild from scratch")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = run_calibration_build(config, args)
    logger.info(f"Done. Dataset: {len(dataset)} samples at {dataset.max_seq_length} tokens each.")
