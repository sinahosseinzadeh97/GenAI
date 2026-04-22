"""Colab-ready LoRA/QLoRA fine-tuner for LexReview.

This module is a **standalone script** — it is NOT imported by the FastAPI
application.  Run it directly in Google Colab or on a GPU machine:

    python -m src.lexreview.finetune.trainer \\
        --data data/sft_train.jsonl \\
        --model qwen   \\   # or 'mistral'
        --output ./lora_checkpoints

Requirements (install via [project.optional-dependencies] finetune):
    pip install ragforge-enterprise[finetune]

Target models
-------------
- ``qwen``    → ``Qwen/Qwen2-0.5B``   (fast, Colab-T4 friendly)
- ``mistral`` → ``mistralai/Mistral-7B-v0.1`` (best quality, A100 recommended)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, str] = {
    "qwen": "Qwen/Qwen2-0.5B",
    "mistral": "mistralai/Mistral-7B-v0.1",
}

LORA_TARGET_MODULES: dict[str, list[str]] = {
    "qwen": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


class LoraFinetuner:
    """QLoRA fine-tuner using PEFT + TRL SFTTrainer.

    Supports Qwen-0.5B and Mistral-7B with 4-bit quantisation via
    ``bitsandbytes``.  All heavy imports are deferred to :meth:`train`
    to keep module import time fast.

    Args:
        model_key:       ``"qwen"`` or ``"mistral"``.
        data_path:       Path to the SFT JSONL file.
        output_dir:      Directory to save LoRA adapter checkpoints.
        lora_r:          LoRA rank (default 16).
        lora_alpha:      LoRA alpha scaling (default 32).
        lora_dropout:    LoRA dropout (default 0.05).
        use_4bit:        Enable 4-bit QLoRA quantisation (default True).
        epochs:          Training epochs (default 3).
        batch_size:      Per-device training batch size (default 2).
        grad_accum:      Gradient accumulation steps (default 8).
        lr:              Learning rate (default 2e-4).
        max_seq_length:  Maximum sequence length in tokens (default 2048).

    Example::

        # In Google Colab:
        finetuner = LoraFinetuner(
            model_key="qwen",
            data_path=Path("data/sft_train.jsonl"),
            output_dir=Path("./lora_out"),
        )
        finetuner.train()
    """

    def __init__(
        self,
        model_key: str = "qwen",
        data_path: Path = Path("data/sft_train.jsonl"),
        output_dir: Path = Path("./lora_checkpoints"),
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_4bit: bool = True,
        epochs: int = 3,
        batch_size: int = 2,
        grad_accum: int = 8,
        lr: float = 2e-4,
        max_seq_length: int = 2048,
    ) -> None:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_key '{model_key}'. Choose from: {list(MODEL_REGISTRY)}"
            )
        self.model_key = model_key
        self.model_id = MODEL_REGISTRY[model_key]
        self.data_path = data_path
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.lr = lr
        self.max_seq_length = max_seq_length

    def _load_dataset(self) -> any:  # type: ignore[return]  # noqa: ANN401
        """Load JSONL training data into a HuggingFace Dataset.

        Returns:
            A ``datasets.Dataset`` with a ``"text"`` column containing
            concatenated prompt+completion strings.

        Raises:
            FileNotFoundError: When *data_path* does not exist.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training data not found at: {self.data_path}")

        from datasets import Dataset  # type: ignore[import-untyped]

        records = []
        with self.data_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    # SFTTrainer expects a "text" field = prompt + completion
                    records.append({"text": obj["prompt"] + obj["completion"]})

        log.info("Loaded %d training samples from %s", len(records), self.data_path)
        return Dataset.from_list(records)

    def train(self) -> None:
        """Run the full LoRA/QLoRA fine-tuning loop.

        Steps:
        1. Load model + tokenizer (with optional 4-bit quantisation)
        2. Apply LoRA adapter via PEFT
        3. Load and format training dataset
        4. Run SFTTrainer
        5. Save adapter to ``output_dir``

        Raises:
            ImportError: When optional ``[finetune]`` dependencies are missing.
            FileNotFoundError: When data file is missing.
        """
        try:
            import torch  # type: ignore[import-untyped]
            import transformers  # type: ignore[import-untyped]
            from peft import (  # type: ignore[import-untyped]
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
            from trl import SFTTrainer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Fine-tuning dependencies missing. "
                "Run: pip install 'ragforge-enterprise[finetune]'"
            ) from exc

        log.info("LoraFinetuner: loading model '%s'", self.model_id)

        # ── Quantisation config (4-bit QLoRA) ─────────────────────────────────
        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # ── Load model + tokenizer ─────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )

        if self.use_4bit:
            model = prepare_model_for_kbit_training(model)

        # ── LoRA adapter ───────────────────────────────────────────────────────
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=LORA_TARGET_MODULES[self.model_key],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ── Dataset ────────────────────────────────────────────────────────────
        train_dataset = self._load_dataset()

        # ── Training arguments ─────────────────────────────────────────────────
        training_args = transformers.TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            learning_rate=self.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            fp16=not self.use_4bit,
            bf16=self.use_4bit,
            logging_steps=10,
            save_strategy="epoch",
            optim="paged_adamw_8bit" if self.use_4bit else "adamw_torch",
            report_to="none",
        )

        # ── SFTTrainer ─────────────────────────────────────────────────────────
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            args=training_args,
        )

        log.info("LoraFinetuner: starting training")
        trainer.train()

        # ── Save adapter ───────────────────────────────────────────────────────
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        log.info("LoraFinetuner: saved adapter to '%s'", self.output_dir)


# ── CLI entry point ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LexReview LoRA/QLoRA fine-tuner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/sft_train.jsonl"), help="SFT JSONL file path."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY),
        default="qwen",
        help="Model to fine-tune: 'qwen' (Qwen2-0.5B) or 'mistral' (Mistral-7B).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("./lora_checkpoints"), help="Checkpoint output dir."
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantisation.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _parse_args()
    finetuner = LoraFinetuner(
        model_key=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_4bit=not args.no_4bit,
    )
    finetuner.train()
