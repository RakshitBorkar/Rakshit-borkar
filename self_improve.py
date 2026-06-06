"""
scripts/self_improve.py
────────────────────────
Self-improvement loop: the model generates its own training data,
scores it, filters the best, and retrains itself.

This is the core idea behind:
- STaR (Self-Taught Reasoner)
- ReST (Reinforced Self-Training)
- Constitutional AI self-critique
- DeepSeek-R1 rejection sampling

Pipeline per round:
  1. Generate multiple responses per seed prompt
  2. Score with reward function
  3. Keep only high-quality (top-k or above threshold)
  4. Add to training pool
  5. SFT fine-tune on new data
  6. Evaluate improvement
  7. Repeat

Run:
    python scripts/self_improve.py --rounds 5 --model ./checkpoints/sft/final_adapter
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model import SLMEngine
from rl.grpo_train import RewardFunctions


# ── Config ───────────────────────────────────────────────────────────────────
@dataclass
class SelfImprovementConfig:
    model_name:         str   = "Qwen/Qwen2.5-3B-Instruct"
    adapter_path:       Optional[str] = None
    seed_prompts_file:  str   = "./data/seed_prompts.jsonl"
    output_dir:         str   = "./checkpoints/self_improve"

    rounds:             int   = 5
    samples_per_prompt: int   = 4       # how many responses to generate
    quality_threshold:  float = 0.5    # minimum reward to keep a sample
    max_new_tokens:     int   = 256
    temperature:        float = 0.8

    # Retrain after collecting this many new samples
    retrain_every:      int   = 200
    sft_epochs:         int   = 1      # quick SFT per round


# ── Synthetic Data Generator ──────────────────────────────────────────────────
class SyntheticDataGenerator:

    def __init__(self, engine: SLMEngine, config: SelfImprovementConfig):
        self.engine = engine
        self.config = config
        self.reward = RewardFunctions.combined_reward

    def generate_samples(self, prompt: str) -> List[Dict]:
        """Generate multiple responses and score them."""
        samples = []
        for _ in range(self.config.samples_per_prompt):
            try:
                response = self.engine.generate(
                    prompt=prompt,
                    system_prompt="You are a helpful, accurate, and thoughtful AI assistant.",
                )
                score = self.reward(prompt, response)
                samples.append({
                    "prompt":   prompt,
                    "response": response,
                    "reward":   score,
                })
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
        return samples

    def filter_samples(self, samples: List[Dict]) -> List[Dict]:
        """Keep only high-quality samples."""
        filtered = [s for s in samples if s["reward"] >= self.config.quality_threshold]
        # Also keep the single best even if below threshold
        if not filtered and samples:
            best = max(samples, key=lambda x: x["reward"])
            filtered = [best]
        filtered.sort(key=lambda x: x["reward"], reverse=True)
        return filtered

    def generate_critique_prompt(self, prompt: str, response: str) -> str:
        """Ask the model to critique and improve its own response."""
        return (
            f"Here is a question and an answer. "
            f"Critique the answer and then provide an improved version.\n\n"
            f"Question: {prompt}\n\n"
            f"Original Answer: {response}\n\n"
            f"Critique and Improved Answer:"
        )

    def self_critique_sample(self, prompt: str, response: str) -> Optional[Dict]:
        """Generate a self-critique and improved response."""
        critique_prompt = self.generate_critique_prompt(prompt, response)
        try:
            improved = self.engine.generate(prompt=critique_prompt)
            score    = self.reward(prompt, improved)
            return {"prompt": prompt, "response": improved, "reward": score, "type": "critique"}
        except Exception as e:
            logger.warning(f"Critique generation failed: {e}")
            return None


# ── Seed Prompt Generator (if no file provided) ───────────────────────────────
DEFAULT_SEED_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "What are the key differences between supervised and unsupervised learning?",
    "How does attention mechanism work in transformers?",
    "Describe the steps to solve a system of linear equations.",
    "What is the difference between RAM and ROM?",
    "Explain what recursion is in programming with an example.",
    "How does a neural network learn from data?",
    "What are the main principles of object-oriented programming?",
    "Describe how the internet works at a high level.",
    "What is the difference between correlation and causation?",
    "Explain gradient descent in simple terms.",
    "What is tokenization in NLP?",
    "How does HTTPS make web connections secure?",
    "What are the advantages of using version control systems?",
    "Explain what a REST API is and how it works.",
]


def load_seed_prompts(path: str) -> List[str]:
    if not Path(path).exists():
        logger.warning(f"Seed prompts file not found: {path}. Using defaults.")
        return DEFAULT_SEED_PROMPTS

    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("instruction") or obj.get("question", "")
            if prompt:
                prompts.append(prompt)

    logger.info(f"Loaded {len(prompts)} seed prompts.")
    return prompts


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(engine: SLMEngine, eval_prompts: List[str]) -> Dict:
    """Quick evaluation: average reward on held-out prompts."""
    rewards = []
    for prompt in eval_prompts[:20]:   # cap at 20 for speed
        try:
            response = engine.generate(prompt)
            reward   = RewardFunctions.combined_reward(prompt, response)
            rewards.append(reward)
        except Exception:
            pass

    if not rewards:
        return {"mean_reward": 0.0, "n_samples": 0}

    import numpy as np
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "n_samples":   len(rewards),
    }


# ── Self-Improvement Loop ─────────────────────────────────────────────────────
class SelfImprovementLoop:

    def __init__(self, config: SelfImprovementConfig):
        self.config    = config
        self.data_pool: List[Dict] = []
        self.metrics_history: List[Dict] = []

    def run(self):
        cfg = self.config
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Load seed prompts
        seed_prompts = load_seed_prompts(cfg.seed_prompts_file)
        eval_prompts = seed_prompts[:10]   # hold out for evaluation

        logger.info(f"Starting self-improvement loop: {cfg.rounds} rounds")

        current_adapter = cfg.adapter_path
        best_reward     = -999.0

        for round_num in range(1, cfg.rounds + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"ROUND {round_num} / {cfg.rounds}")
            logger.info(f"{'='*60}")

            # Load current model
            engine = SLMEngine(
                model_name_or_path=cfg.model_name,
                lora_adapter_path=current_adapter,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
            )
            generator = SyntheticDataGenerator(engine, cfg)

            # Evaluate before
            eval_before = evaluate_model(engine, eval_prompts)
            logger.info(f"Eval BEFORE training: {eval_before}")

            # Generate synthetic data
            new_samples = []
            for prompt in seed_prompts:
                samples  = generator.generate_samples(prompt)
                filtered = generator.filter_samples(samples)
                new_samples.extend(filtered)

                # Self-critique the best response
                if filtered:
                    best = filtered[0]
                    critique = generator.self_critique_sample(best["prompt"], best["response"])
                    if critique and critique["reward"] > best["reward"]:
                        new_samples.append(critique)

            logger.info(f"Round {round_num}: collected {len(new_samples)} new samples")

            # Save new samples
            round_data_path = Path(cfg.output_dir) / f"round_{round_num}_data.jsonl"
            with open(round_data_path, "w") as f:
                for sample in new_samples:
                    # Convert to training format
                    record = {
                        "instruction": sample["prompt"],
                        "output":      sample["response"],
                        "reward":      sample["reward"],
                    }
                    f.write(json.dumps(record) + "\n")

            self.data_pool.extend(new_samples)

            # Free model memory before retraining
            engine.free_memory()
            del engine, generator

            # Retrain if enough data
            if len(self.data_pool) >= cfg.retrain_every or round_num == cfg.rounds:
                logger.info(f"Retraining on {len(self.data_pool)} samples...")
                new_adapter = self._retrain(round_num, round_data_path, current_adapter)
                current_adapter = new_adapter

                # Evaluate after
                engine_new = SLMEngine(
                    model_name_or_path=cfg.model_name,
                    lora_adapter_path=current_adapter,
                    max_new_tokens=cfg.max_new_tokens,
                )
                eval_after = evaluate_model(engine_new, eval_prompts)
                engine_new.free_memory()

                logger.info(f"Eval AFTER training: {eval_after}")

                improvement = eval_after["mean_reward"] - eval_before["mean_reward"]
                logger.info(f"Improvement: {improvement:+.4f}")

                if eval_after["mean_reward"] > best_reward:
                    best_reward = eval_after["mean_reward"]
                    best_path   = Path(cfg.output_dir) / "best_adapter"
                    import shutil
                    if Path(current_adapter).exists():
                        shutil.copytree(current_adapter, best_path, dirs_exist_ok=True)
                    logger.success(f"New best model! Reward: {best_reward:.4f}")

                self.metrics_history.append({
                    "round": round_num,
                    "before": eval_before,
                    "after":  eval_after,
                    "improvement": improvement,
                    "new_samples": len(new_samples),
                })

        # Save metrics
        metrics_path = Path(cfg.output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        logger.success(f"\nSelf-improvement complete!")
        logger.success(f"Best reward: {best_reward:.4f}")
        logger.success(f"Best model: {Path(cfg.output_dir) / 'best_adapter'}")
        logger.success(f"Metrics saved: {metrics_path}")

    def _retrain(self, round_num: int, data_path: Path, current_adapter: Optional[str]) -> str:
        """Quick SFT fine-tune on new data."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from training.sft_train import SFTConfig, SFTTrainerWrapper

        output = Path(self.config.output_dir) / f"adapter_round_{round_num}"
        cfg    = SFTConfig(
            model_name=self.config.model_name,
            train_file=str(data_path),
            output_dir=str(output),
            epochs=self.config.sft_epochs,
            batch_size=2,
            grad_accum=8,
            lr=1e-4,
        )
        trainer = SFTTrainerWrapper(cfg)
        return trainer.train()


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds",  type=int, default=5)
    parser.add_argument("--model",   type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--output",  type=str, default="./checkpoints/self_improve")
    args = parser.parse_args()

    config = SelfImprovementConfig(
        model_name=args.model,
        adapter_path=args.adapter,
        output_dir=args.output,
        rounds=args.rounds,
    )
    loop = SelfImprovementLoop(config)
    loop.run()