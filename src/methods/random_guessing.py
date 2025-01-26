#!/usr/bin/env python3

"""
random_guessing.py

Run evaluations on tasks from BIG-Bench-Hard using a random guessing baseline.
In addition to printing results, we also save them to a specified JSON file.
"""

import os
import json
import argparse
from tqdm import tqdm
from src.tasks import TASKS

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run random guessing baseline evaluations on tasks from BIG-Bench-Hard")
    parser.add_argument("--exp_name", type=str, default="random_guessing", help="Experiment name for logging.")
    parser.add_argument("--output_file", type=str, default="random_guessing_results.json", help="File to save the results.")
    parser.add_argument("--task_start", type=int, default=0, help="Start index for the range of tasks to evaluate.")
    parser.add_argument("--task_end", type=int, default=None, help="End index (exclusive) for the range of tasks to evaluate.")
    args = parser.parse_args()

    # Validate task_end
    if args.task_end is None:
        args.task_end = len(TASKS)

    results = []

    # Select task range
    tasks_subset = list(TASKS.items())[args.task_start:args.task_end]

    for task, info in tqdm(tasks_subset, desc="Evaluating tasks"):
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

        # Get the number of choices for the task
        choices = info.get("choices", 1)

        # Calculate random guessing accuracy
        random_guessing_accuracy = round(100 / choices, 4) if choices else 0

        print(f"=== Task: {task} ===")
        print(f"Random Guessing Accuracy: {random_guessing_accuracy:.2f}%\n")

        # Compile task results
        task_results = {
            "task": task,
            f"{args.exp_name}_accuracy": random_guessing_accuracy,
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
        print(f"\nSaved random guessing results to {args.output_file}.")
    except Exception as e:
        print(f"Error saving results to {args.output_file}: {e}")


if __name__ == "__main__":
    main()