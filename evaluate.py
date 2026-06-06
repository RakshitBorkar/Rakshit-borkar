"""
scripts/evaluate.py
────────────────────
Evaluate your SLM against benchmarks and compare versions.

Metrics:
  - ROUGE-L score (vs reference answers)
  - Reward model score
  - Response length distribution
  - Refusal rate
  - Latency (tokens/sec)

Run:
    python scripts/evaluate.py --model Qwen/Qwen2.5-3B-Instruct
    python scripts/evaluate.py --adapter ./checkpoints/sft/final_adapter --compare base
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model import SLMEngine
from rl.grpo_train import RewardFunctions


# ── Built-in eval prompts ─────────────────────────────────────────────────────
EVAL_PROMPTS = [
    {"prompt": "Explain what a transformer neural network is.",
     "ref":    "A transformer is a deep learning model that uses self-attention mechanisms to process sequential data."},
    {"prompt": "What is the difference between TCP and UDP?",
     "ref":    "TCP is connection-oriented and reliable; UDP is connectionless and faster but without guaranteed delivery."},
    {"prompt": "Write a Python function to reverse a string.",
     "ref":    "def reverse_string(s): return s[::-1]"},
    {"prompt": "What causes inflation?",
     "ref":    "Inflation is caused by increased money supply, demand exceeding supply, or rising production costs."},
    {"prompt": "Explain recursion with a simple example.",
     "ref":    "Recursion is a function calling itself. Example: factorial(n) = n * factorial(n-1)."},
    {"prompt": "What is gradient descent?",
     "ref":    "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function."},
    {"prompt": "How does HTTPS work?",
     "ref":    "HTTPS uses TLS/SSL to encrypt communication between browser and server using public-key cryptography."},
    {"prompt": "What is overfitting in machine learning?",
     "ref":    "Overfitting occurs when a model learns training data too specifically, performing poorly on new data."},
]


# ── Metrics ───────────────────────────────────────────────────────────────────
def rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 score."""
    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()

    if not h_tokens or not r_tokens:
        return 0.0

    # LCS length
    m, n = len(h_tokens), len(r_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if h_tokens[i-1] == r_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs = dp[m][n]
    precision = lcs / m if m > 0 else 0
    recall    = lcs / n if n > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else 0)
    return round(f1, 4)


def compute_metrics(prompt: str, response: str, reference: str) -> Dict:
    return {
        "rouge_l": rouge_l(response, reference),
        "reward":  RewardFunctions.combined_reward(prompt, response),
        "length":  len(response.split()),
        "refused": any(p in response.lower() for p in
                       ["i cannot", "i can't", "i'm unable", "as an ai"]),
    }


# ── Evaluator ─────────────────────────────────────────────────────────────────
class ModelEvaluator:

    def __init__(self, model_name: str, adapter_path: Optional[str] = None):
        self.engine = SLMEngine(
            model_name_or_path=model_name,
            lora_adapter_path=adapter_path,
            quantize=True,
        )

    def run(self, eval_set: List[Dict] = None) -> Dict:
        if eval_set is None:
            eval_set = EVAL_PROMPTS

        all_metrics = []
        latencies   = []

        logger.info(f"Evaluating on {len(eval_set)} prompts...")

        for item in eval_set:
            prompt = item["prompt"]
            ref    = item.get("ref", "")

            t0       = time.time()
            response = self.engine.generate(prompt)
            elapsed  = time.time() - t0
            tok_sec  = len(response.split()) / max(elapsed, 0.01)

            metrics = compute_metrics(prompt, response, ref)
            metrics["tokens_per_sec"] = round(tok_sec, 1)
            all_metrics.append(metrics)
            latencies.append(elapsed)

            logger.info(
                f"  ROUGE-L: {metrics['rouge_l']:.3f} | "
                f"Reward: {metrics['reward']:.3f} | "
                f"Length: {metrics['length']} | "
                f"Speed: {tok_sec:.0f} tok/s"
            )

        # Aggregate
        summary = {
            "mean_rouge_l":  round(float(np.mean([m["rouge_l"]  for m in all_metrics])), 4),
            "mean_reward":   round(float(np.mean([m["reward"]   for m in all_metrics])), 4),
            "mean_length":   round(float(np.mean([m["length"]   for m in all_metrics])), 1),
            "refusal_rate":  round(float(np.mean([m["refused"]  for m in all_metrics])), 3),
            "mean_latency":  round(float(np.mean(latencies)), 3),
            "tok_per_sec":   round(float(np.mean([m["tokens_per_sec"] for m in all_metrics])), 1),
            "n_samples":     len(all_metrics),
        }

        return summary, all_metrics

    def compare(self, other_engine: "ModelEvaluator", eval_set=None):
        """Compare two model versions."""
        logger.info("Evaluating Model A (this)...")
        summary_a, _ = self.run(eval_set)

        logger.info("Evaluating Model B (other)...")
        summary_b, _ = other_engine.run(eval_set)

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"{'Metric':<20} {'Model A':>12} {'Model B':>12} {'Delta':>12}")
        print("-" * 60)

        for key in summary_a:
            a = summary_a[key]
            b = summary_b[key]
            if isinstance(a, (int, float)):
                delta = b - a
                sign  = "+" if delta > 0 else ""
                print(f"{key:<20} {a:>12.4f} {b:>12.4f} {sign}{delta:>11.4f}")

        print("=" * 60)
        return summary_a, summary_b


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--compare", type=str, default=None,
                        help="Second adapter to compare against")
    parser.add_argument("--output",  type=str, default="./eval_results.json")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model, args.adapter)

    if args.compare:
        other = ModelEvaluator(args.model, args.compare)
        summary_a, summary_b = evaluator.compare(other)
        results = {"model_a": summary_a, "model_b": summary_b}
    else:
        summary, details = evaluator.run()
        results = {"summary": summary}
        print("\n📊 EVALUATION RESULTS:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Results saved to: {args.output}")