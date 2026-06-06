"""
core/model.py
─────────────
Model loading, quantization, and inference engine.
Supports Qwen2.5, Mistral, LLaMA families.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from threading import Thread
from loguru import logger
from typing import Optional, Generator
import gc


# ── Supported base models ──────────────────────────────────────────────────
SUPPORTED_MODELS = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b":   "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b":   "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3.2-3b":  "meta-llama/Llama-3.2-3B-Instruct",
    "tinyllama":    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


class SLMEngine:
    """
    Main inference engine for your SLM.
    Handles model loading, LoRA adapters, quantization, and generation.
    """

    def __init__(
        self,
        model_name_or_path: str,
        lora_adapter_path: Optional[str] = None,
        quantize: bool = True,          # 4-bit quantization for low VRAM
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_name_or_path = model_name_or_path
        self.lora_adapter_path  = lora_adapter_path
        self.quantize           = quantize
        self.device             = device
        self.max_new_tokens     = max_new_tokens
        self.temperature        = temperature
        self.top_p              = top_p

        self.model     = None
        self.tokenizer = None
        self._load()

    # ── Loading ─────────────────────────────────────────────────────────────
    def _load(self):
        logger.info(f"Loading model: {self.model_name_or_path}")

        # Resolve shorthand names
        resolved = SUPPORTED_MODELS.get(self.model_name_or_path, self.model_name_or_path)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config (saves 60-70% VRAM)
        bnb_config = None
        if self.quantize and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Base model
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved,
            quantization_config=bnb_config,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not self.quantize else None,
        )

        # Load LoRA adapter if provided (fine-tuned weights)
        if self.lora_adapter_path:
            logger.info(f"Loading LoRA adapter from: {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
            self.model = self.model.merge_and_unload()  # merge for faster inference

        self.model.eval()
        logger.success("Model loaded successfully.")

    # ── Generation ──────────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful, accurate, and concise AI assistant.",
        stream: bool = False,
    ) -> str | Generator:
        """Generate a response given a prompt."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]

        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if stream:
            return self._stream(gen_kwargs, input_ids.shape[1])

        with torch.no_grad():
            output = self.model.generate(**gen_kwargs)

        # Decode only the new tokens
        new_tokens = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _stream(self, gen_kwargs: dict, prompt_len: int) -> Generator:
        """Streaming generation (token by token)."""
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token

    # ── LoRA Setup (for training) ────────────────────────────────────────────
    def attach_lora(self, r: int = 16, lora_alpha: int = 32, dropout: float = 0.05):
        """Attach LoRA adapters for fine-tuning."""
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()
        return self.model

    def free_memory(self):
        """Release GPU memory."""
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("GPU memory freed.")