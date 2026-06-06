"""
rl/grpo_train.py
────────────────
GRPO (Group Relative Policy Optimization) — the same RL algorithm used in
DeepSeek-R1 and similar frontier reasoning models.

How GRPO works:
  1. For each prompt, sample G responses from the current policy
  2. Score each response with a reward function
  3. Compute relative advantages: A_i = (r_i - mean(r)) / std(r)
  4. Update policy to increase probability of high-advantage responses
  5. KL penalty keeps policy from drifting too far from reference

Run:
    python rl/grpo_train.py --config configs/rl_config.yaml
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
from loguru import logger
import numpy as np


# ── Config ───────────────────────────────────────────────────────────────────
@dataclass
class GRPOConfig:
    model_name:          str   = "Qwen/Qwen2.5-3B-Instruct"
    sft_adapter_path:    Optional[str] = None          # start from SFT checkpoint
    output_dir:          str   = "./checkpoints/rl"
    train_file:          str   = "./data/rl_prompts.jsonl"

    # GRPO hyperparams
    group_size:          int   = 4       # G: responses sampled per prompt
    epochs:              int   = 2
    batch_size:          int   = 2       # prompts per batch
    lr:                  float = 1e-5
    kl_coef:             float = 0.05   # KL penalty weight
    max_new_tokens:      int   = 256
    temperature:         float = 0.8
    clip_ratio:          float = 0.2    # PPO-style clipping

    # LoRA
    lora_r:              int   = 8
    lora_alpha:          int   = 16
    use_4bit:            bool  = True

    logging_steps:       int   = 10
    save_steps:          int   = 50

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.get("rl", {}).items()
                      if k in cls.__dataclass_fields__})


# ── Reward Functions ──────────────────────────────────────────────────────────
class RewardFunctions:
    """
    Modular reward functions. Combine them as needed.
    Each function takes (prompt, response) and returns a float score.
    """

    @staticmethod
    def length_penalty(prompt: str, response: str,
                       ideal_min: int = 50, ideal_max: int = 300) -> float:
        """Penalize very short or very long responses."""
        length = len(response.split())
        if length < ideal_min:
            return -0.5 + (length / ideal_min) * 0.5
        elif length > ideal_max:
            return max(-0.3, -0.3 * ((length - ideal_max) / ideal_max))
        return 0.2

    @staticmethod
    def format_reward(prompt: str, response: str) -> float:
        """Reward well-structured responses."""
        score = 0.0
        if len(response) > 20:       score += 0.1
        if not response.isupper():   score += 0.1
        if response[0].isupper():    score += 0.05
        # Penalize repetition
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            score += unique_ratio * 0.2
        return score

    @staticmethod
    def helpfulness_heuristic(prompt: str, response: str) -> float:
        """Simple heuristic: does the response address the question?"""
        import re
        score = 0.0
        q_words = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
        r_words = set(re.findall(r'\b\w{4,}\b', response.lower()))
        overlap = len(q_words & r_words) / max(len(q_words), 1)
        score += overlap * 0.3
        # Reward answers that contain numbers/facts
        if re.search(r'\d+', response):
            score += 0.1
        return score

    @staticmethod
    def no_refusal_reward(prompt: str, response: str) -> float:
        """Penalize unnecessary refusals."""
        refusal_phrases = [
            "i cannot", "i can't", "i'm not able", "i won't",
            "i'm unable", "as an ai", "i don't have the ability"
        ]
        resp_lower = response.lower()
        for phrase in refusal_phrases:
            if phrase in resp_lower:
                return -0.5
        return 0.1

    @staticmethod
    def reasoning_reward(prompt: str, response: str) -> float:
        """Reward step-by-step reasoning (chain-of-thought)."""
        cot_indicators = [
            "first,", "second,", "third,", "step 1", "step 2",
            "therefore", "because", "since", "as a result",
            "let me think", "let's consider"
        ]
        resp_lower = response.lower()
        count = sum(1 for ind in cot_indicators if ind in resp_lower)
        return min(0.4, count * 0.1)

    @classmethod
    def combined_reward(cls, prompt: str, response: str) -> float:
        """Combine all heuristics into one scalar reward."""
        return (
            cls.format_reward(prompt, response) +
            cls.helpfulness_heuristic(prompt, response) +
            cls.no_refusal_reward(prompt, response) +
            cls.length_penalty(prompt, response) +
            cls.reasoning_reward(prompt, response)
        )


# ── Custom Reward (plug in your own) ─────────────────────────────────────────
class CustomRewardModel:
    """
    Use a trained reward model (DeBERTa-based) for more accurate scoring.
    Falls back to heuristics if not available.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path and Path(model_path).exists():
            from transformers import pipeline
            self.model = pipeline(
                "text-classification",
                model=model_path,
                device=0 if torch.cuda.is_available() else -1,
            )
            logger.info(f"Reward model loaded from {model_path}")
        else:
            logger.info("No reward model found — using heuristic rewards.")

    def score(self, prompt: str, response: str) -> float:
        if self.model:
            text  = f"Question: {prompt}\nAnswer: {response}"
            result = self.model(text[:512])[0]
            # Assumes label POSITIVE=good, NEGATIVE=bad
            score = result["score"] if result["label"] == "POSITIVE" else -result["score"]
            return float(score)
        return RewardFunctions.combined_reward(prompt, response)


# ── GRPO Trainer ─────────────────────────────────────────────────────────────
class GRPOTrainer:

    def __init__(self, config: GRPOConfig, reward_fn: Optional[Callable] = None):
        self.config    = config
        self.reward_fn = reward_fn or RewardFunctions.combined_reward
        self.tokenizer = None
        self.policy    = None     # trainable model
        self.reference = None     # frozen reference model (for KL)

    def setup(self):
        cfg = self.config
        logger.info(f"Setting up GRPO trainer — base: {cfg.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"   # important for generation

        bnb_config = None
        if cfg.use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Policy model gets LoRA
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        self.policy = get_peft_model(base, lora_cfg)
        self.policy.print_trainable_parameters()

        # Reference model is frozen (used for KL divergence)
        self.reference = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        for param in self.reference.parameters():
            param.requires_grad = False
        self.reference.eval()

        logger.success("GRPO setup complete.")

    def load_prompts(self) -> List[str]:
        prompts = []
        with open(self.config.train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt") or obj.get("instruction") or obj.get("question", "")
                if prompt:
                    prompts.append(prompt)
        logger.info(f"Loaded {len(prompts)} training prompts.")
        return prompts

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        messages = [
            {"role": "system",  "content": "You are a helpful AI assistant."},
            {"role": "user",    "content": prompt},
        ]
        ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        return ids.to(self.policy.device)

    @torch.no_grad()
    def _sample_responses(self, input_ids: torch.Tensor) -> List[str]:
        """Sample G responses from the policy."""
        cfg = self.config
        outputs = self.policy.generate(
            input_ids.repeat(cfg.group_size, 1),
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        responses = []
        for out in outputs:
            new_tokens = out[input_ids.shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(text)
        return responses

    def _compute_log_probs(self, model, input_ids: torch.Tensor,
                           response_ids: torch.Tensor) -> torch.Tensor:
        """Compute token log-probabilities for a response."""
        full_ids = torch.cat([input_ids, response_ids], dim=-1)
        with torch.no_grad() if model is self.reference else torch.enable_grad():
            logits = model(full_ids).logits
        # Shift: predict next token
        shift_logits = logits[0, input_ids.shape[1]-1:-1]
        log_probs    = F.log_softmax(shift_logits, dim=-1)
        token_lp     = log_probs.gather(-1, response_ids[0].unsqueeze(-1)).squeeze(-1)
        return token_lp.sum()

    def _grpo_step(self, prompt: str, optimizer: torch.optim.Optimizer) -> Dict:
        """One GRPO update step for a single prompt."""
        cfg      = self.config
        input_ids = self._tokenize_prompt(prompt)

        # 1. Sample G responses
        responses = self._sample_responses(input_ids)

        # 2. Score responses
        rewards = np.array([self.reward_fn(prompt, r) for r in responses], dtype=np.float32)

        # 3. Compute relative advantages
        mean_r = rewards.mean()
        std_r  = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # 4. Policy gradient update
        total_loss = torch.tensor(0.0, device=self.policy.device, requires_grad=True)

        for resp, adv in zip(responses, advantages):
            if not resp.strip():
                continue
            resp_ids = self.tokenizer(
                resp, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.policy.device)

            if resp_ids.shape[1] == 0:
                continue

            # Policy log-prob
            policy_lp = self._compute_log_probs(self.policy, input_ids, resp_ids)

            # Reference log-prob (for KL)
            with torch.no_grad():
                ref_lp = self._compute_log_probs(self.reference, input_ids, resp_ids)

            kl      = policy_lp - ref_lp
            pg_loss = -torch.tensor(adv, device=self.policy.device) * policy_lp
            loss    = pg_loss + cfg.kl_coef * kl
            total_loss = total_loss + loss

        total_loss = total_loss / cfg.group_size
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        optimizer.step()

        return {
            "loss":       total_loss.item(),
            "mean_reward": mean_r,
            "best_reward": rewards.max(),
            "best_response": responses[rewards.argmax()],
        }

    def train(self):
        self.setup()
        prompts   = self.load_prompts()
        cfg       = self.config
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.policy.parameters()),
            lr=cfg.lr,
        )

        step = 0
        for epoch in range(cfg.epochs):
            np.random.shuffle(prompts)
            for i in range(0, len(prompts), cfg.batch_size):
                batch = prompts[i : i + cfg.batch_size]
                for prompt in batch:
                    metrics = self._grpo_step(prompt, optimizer)
                    step   += 1

                    if step % cfg.logging_steps == 0:
                        logger.info(
                            f"Step {step} | Loss: {metrics['loss']:.4f} | "
                            f"Mean Reward: {metrics['mean_reward']:.3f} | "
                            f"Best Reward: {metrics['best_reward']:.3f}"
                        )

                    if step % cfg.save_steps == 0:
                        save_path = Path(cfg.output_dir) / f"step_{step}"
                        self.policy.save_pretrained(str(save_path))
                        logger.info(f"Checkpoint saved: {save_path}")

        # Final save
        final_path = Path(cfg.output_dir) / "final_rl_adapter"
        self.policy.save_pretrained(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        logger.success(f"RL training complete! Saved to: {final_path}")
        return str(final_path)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rl_config.yaml")
    args = parser.parse_args()

    config  = GRPOConfig.from_yaml(args.config) if Path(args.config).exists() else GRPOConfig()
    trainer = GRPOTrainer(config)
    trainer.train()