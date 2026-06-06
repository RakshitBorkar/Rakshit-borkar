"""
training/sft_train.py
─────────────────────
Supervised Fine-Tuning using LoRA + QLoRA.
Trains on instruction-following data in ShareGPT / Alpaca format.

Run:
    python training/sft_train.py --config configs/sft_config.yaml
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from loguru import logger


# ── Config ───────────────────────────────────────────────────────────────────
@dataclass
class SFTConfig:
    # Model
    model_name:       str   = "Qwen/Qwen2.5-3B-Instruct"
    output_dir:       str   = "./checkpoints/sft"

    # Data
    train_file:       str   = "./data/train.jsonl"
    val_file:         Optional[str] = None
    max_seq_length:   int   = 2048

    # LoRA
    lora_r:           int   = 16
    lora_alpha:       int   = 32
    lora_dropout:     float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training
    epochs:           int   = 3
    batch_size:       int   = 4
    grad_accum:       int   = 4
    lr:               float = 2e-4
    warmup_ratio:     float = 0.05
    weight_decay:     float = 0.01
    max_grad_norm:    float = 1.0
    save_steps:       int   = 100
    eval_steps:       int   = 100
    logging_steps:    int   = 10

    # Quantization
    use_4bit:         bool  = True

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.get("sft", {}).items()
                      if k in cls.__dataclass_fields__})


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> Dataset:
    """
    Expects JSONL with either:
      {"instruction": "...", "input": "...", "output": "..."}  ← Alpaca format
      {"conversations": [{"role": "user", ...}, {"role": "assistant", ...}]}  ← ShareGPT
    """
    import json
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_alpaca(example: Dict, tokenizer) -> Dict:
    """Format Alpaca-style data into chat template."""
    instruction = example.get("instruction", "")
    inp         = example.get("input", "")
    output      = example.get("output", "")

    user_content = instruction
    if inp:
        user_content += f"\n\n{inp}"

    messages = [
        {"role": "system",    "content": "You are a helpful AI assistant."},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": output},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def format_sharegpt(example: Dict, tokenizer) -> Dict:
    """Format ShareGPT conversations."""
    convs = example.get("conversations", [])
    messages = []
    for turn in convs:
        role    = turn.get("from", turn.get("role", "user"))
        content = turn.get("value", turn.get("content", ""))
        if role in ("human", "user"):
            messages.append({"role": "user",      "content": content})
        elif role in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": content})
        elif role == "system":
            messages.insert(0, {"role": "system", "content": content})
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def detect_format(dataset: Dataset) -> str:
    sample = dataset[0]
    if "conversations" in sample:
        return "sharegpt"
    return "alpaca"


# ── Trainer ───────────────────────────────────────────────────────────────────
class SFTTrainerWrapper:

    def __init__(self, config: SFTConfig):
        self.config = config
        self.tokenizer = None
        self.model     = None

    def setup(self):
        cfg = self.config

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Quantization
        bnb_config = None
        if cfg.use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.config.use_cache = False

        logger.success("Model and tokenizer loaded.")

    def prepare_dataset(self):
        cfg  = self.config
        tok  = self.tokenizer

        train_ds = load_jsonl(cfg.train_file)
        fmt      = detect_format(train_ds)
        formatter = format_sharegpt if fmt == "sharegpt" else format_alpaca

        logger.info(f"Detected format: {fmt}. Train samples: {len(train_ds)}")

        train_ds = train_ds.map(lambda x: formatter(x, tok), remove_columns=train_ds.column_names)

        val_ds = None
        if cfg.val_file and Path(cfg.val_file).exists():
            val_ds = load_jsonl(cfg.val_file)
            val_ds = val_ds.map(lambda x: formatter(x, tok), remove_columns=val_ds.column_names)

        return train_ds, val_ds

    def train(self):
        self.setup()
        train_ds, val_ds = self.prepare_dataset()
        cfg = self.config

        # LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            bias="none",
        )

        # Training args
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.lr,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            fp16=False,
            bf16=torch.cuda.is_available(),
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            eval_strategy="steps" if val_ds else "no",
            eval_steps=cfg.eval_steps if val_ds else None,
            save_total_limit=3,
            load_best_model_at_end=bool(val_ds),
            report_to="none",
            dataloader_num_workers=2,
            group_by_length=True,
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if val_ds else []

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
            args=training_args,
            callbacks=callbacks,
        )

        logger.info("Starting SFT training...")
        trainer.train()

        # Save final adapter
        final_path = Path(cfg.output_dir) / "final_adapter"
        trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        logger.success(f"Training complete! Adapter saved to: {final_path}")

        return str(final_path)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    args = parser.parse_args()

    if Path(args.config).exists():
        config = SFTConfig.from_yaml(args.config)
    else:
        logger.warning("No config found — using defaults.")
        config = SFTConfig()

    trainer = SFTTrainerWrapper(config)
    trainer.train()