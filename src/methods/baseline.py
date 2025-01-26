#!/usr/bin/env python3

"""
baseline.py

Run zero-shot or few-shot evaluations on tasks from BIG-Bench-Hard using a vLLM-based LLM.
In addition to printing results, we also save them to a specified JSON file.

Supports optional majority-vote inference in the few-shot setting.
"""

import os
import json
import time
import random
import argparse
from tqdm import tqdm
from vllm import LLM
from collections import Counter

# Local imports
from src.tasks import TASKS
from src.utils import (
    compute_accuracy,
    inference_vllm
)

def majority_vote(answers):
    """
    Given a list of answers (strings), pick the one that appears most frequently.
    If there's a tie, pick whichever candidate is lexicographically smallest.
    """
    counter = Counter(answers)
    max_count = max(counter.values())
    # All candidates that match the max frequency:
    candidates = [ans for ans, cnt in counter.items() if cnt == max_count]
    # Tie-break by lex order:
    return sorted(candidates)[0]

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run zero-shot or few-shot evaluations on tasks from BIG-Bench-Hard")
    parser.add_argument("--exp_name", type=str, default="baseline", help="Experiment name for logging.")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to the LLM model directory.")
    parser.add_argument("--output_file", type=str, default="baseline_results.json", help="File to save the results.")
    parser.add_argument("--task_start", type=int, default=0, help="Start index for the range of tasks to evaluate.")
    parser.add_argument("--task_end", type=int, default=None, help="End index (exclusive) for the range of tasks to evaluate.")
    parser.add_argument("--k", type=int, default=0, help="Number of few-shot examples. Use k=0 for zero-shot.")
    parser.add_argument("--majority_vote", type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="If True, do multiple permutations of the few-shot examples and pick the majority answer.")
    parser.add_argument("--vote_permutations", type=int, default=5,
                        help="Number of random permutations for majority-vote inference (only if --majority_vote=True).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling examples.")

    args = parser.parse_args()

    # Validate task_end
    if args.task_end is None:
        args.task_end = len(TASKS)

    # Initialize the model with vLLM
    llm = LLM(model=args.model_dir, max_model_len=8192)

    results = []

    # Select task range
    tasks_subset = list(TASKS.items())[args.task_start:args.task_end]

    for task, info in tqdm(tasks_subset, desc="Evaluating tasks"):
        generation_length = info["generation_length"]
        answer_format = info["answer_format"]
        task_prompt = info["task_prompt"]

        # Load the JSON for that task
        task_file = f"external/BIG-Bench-Hard/bbh/{task}.json"
        if not os.path.exists(task_file):
            print(f"Skipping {task} because {task_file} not found.")
            continue

        try:
            with open(task_file, "r") as f:
                question_data = json.load(f)
        except Exception as e:
            print(f"Error loading {task}: {e}")
            continue

        all_examples = question_data.get("examples", [])
        if len(all_examples) == 0:
            print(f"Skipping {task}, no examples found.")
            continue

        # Shuffle examples for consistency
        random.seed(args.seed)
        random.shuffle(all_examples)

        # Prepare questions/targets
        questions = [ex["input"] for ex in all_examples]
        targets = [ex["target"] for ex in all_examples]

        # Determine evaluation examples based on k
        if args.k == 0:
            # Zero-shot
            few_shot_prompt_prefix = ""
            eval_questions = questions
            eval_targets = targets
            method = "Zero-shot"
        else:
            # Few-shot
            if len(all_examples) <= args.k:
                print(f"Skipping {task} for k={args.k}, not enough examples.")
                continue

            # First k examples for in-context
            few_shot_examples = all_examples[:args.k]
            # Remaining are for evaluation
            evaluation_examples = all_examples[args.k:]

            eval_questions = [ex["input"] for ex in evaluation_examples]
            eval_targets = [ex["target"] for ex in evaluation_examples]
            method = f"Few-shot (k={args.k})"

        print(f"=== Task: {task} - {method} ===")
        task_start_time = time.perf_counter()

        # If zero-shot or if few-shot but majority_vote=False, we just do a single pass
        if args.k == 0 or not args.majority_vote:
            # Build the few-shot prefix (if k>0 and majority_vote=False)
            if args.k > 0:
                # Use the first k examples in the same order
                few_shot_prompt_prefix = ""
                for ex in few_shot_examples:
                    q, a = ex["input"], ex["target"]
                    few_shot_prompt_prefix += f"Q: {q}\nA: {a}\n\n"
            else:
                few_shot_prompt_prefix = ""

            # Generate predictions (single pass)
            preds = inference_vllm(
                llm,
                eval_questions,
                max_new_tokens=generation_length,
                task_prompt=task_prompt,
                answer_format=answer_format,
                few_shot_prompt_prefix=few_shot_prompt_prefix
            )
        else:
            # Few-shot with majority vote enabled
            # We'll do multiple permutations of the k examples for each batch inference
            vote_preds = []
            for _ in range(args.vote_permutations):
                # Shuffle the k in-context examples
                perm_examples = few_shot_examples[:]
                random.shuffle(perm_examples)

                # Build a prefix from this permutation
                perm_prefix = ""
                for ex in perm_examples:
                    q, a = ex["input"], ex["target"]
                    perm_prefix += f"Q: {q}\nA: {a}\n\n"

                # Inference for this permutation
                batch_preds = inference_vllm(
                    llm,
                    eval_questions,
                    max_new_tokens=generation_length,
                    task_prompt=task_prompt,
                    answer_format=answer_format,
                    few_shot_prompt_prefix=perm_prefix
                )
                vote_preds.append(batch_preds)

            # For each question, pick the majority answer
            preds = []
            for q_idx in range(len(eval_questions)):
                # Gather the q_idx-th prediction from each permutation run
                candidate_answers = [vote_preds[p][q_idx] for p in range(args.vote_permutations)]
                preds.append(majority_vote(candidate_answers))

        task_end_time = time.perf_counter()
        eval_duration = task_end_time - task_start_time

        # Compute accuracy
        accuracy = compute_accuracy(preds, eval_targets)
        print(f"{method} Accuracy: {accuracy:.2f}%")
        print(f"Evaluation Time: {eval_duration:.2f} seconds\n")

        # Prepare examples with predictions (show up to 5)
        examples = [
            {
                "question": eval_questions[i],
                "prediction": preds[i],
                "true_answer": eval_targets[i].lower()
            }
            for i in range(min(5, len(eval_questions)))
        ]

        # Compile task results
        task_results = {
            "task": task,
            f"{args.exp_name}_accuracy": round(accuracy, 4),
            "eval_time": round(eval_duration, 4),
            "examples": examples
        }

        results.append(task_results)

    # Summaries
    print("\nResults Summary:\n")
    for r in results:
        print(json.dumps(r, indent=2))

    # Save to disk
    try:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved baseline results to {args.output_file}.")
    except Exception as e:
        print(f"Error saving results to {args.output_file}: {e}")


if __name__ == "__main__":
    main()
