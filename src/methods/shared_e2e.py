#!/usr/bin/env python3

"""
shared_e2e.py

Perform LoRA-based finetuning (E2E learning) on multiple BBH tasks simultaneously
using a single shared LoRA adapter. We combine few-shot examples from each task
into one dataset, train once, then evaluate each task on its remaining examples.

Usage example:
    python3 shared_e2e.py --task_start 0 --task_end 27 \
        --model_dir /path/to/base_llm_dir \
        --batch_size 5 --epochs 4 --lr 1e-4 --lora_rank 64 \
        --lora_alpha 16 --k 10
"""

import os
import sys
import time
import random
import json
import shutil
import gc
import argparse

import torch
import yaml
from vllm import LLM
from vllm.lora.request import LoRARequest

# Local imports
from src.tasks import TASKS
from src.utils import (
    compute_accuracy,
    inference_vllm
)

CHECKPOINT_FILES = [
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
]


def create_torchtune_config(
    MODEL_DIR,
    dataset_filename,
    output_dir,
    batch_size,
    epochs,
    lr,
    lora_rank,
    lora_alpha,
    lora_dropout,
    k,
    tasks_in_dataset
):
    """
    Create a YAML config for Torchtune finetuning, using a combined dataset.
    """
    tokenizer_path = os.path.join(MODEL_DIR, "original", "tokenizer.model")
    checkpoint_dir = MODEL_DIR

    # We guess an approximate number of total steps to set for scheduler/warmup
    # Let each task contribute 'k' examples if it has >= k examples.
    total_samples = len(tasks_in_dataset) * k
    # For simplicity, assume steps_per_epoch = (total_samples / batch_size)
    # If total_samples < batch_size, we'll just do 1 step per epoch
    steps_per_epoch = max(1, total_samples // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)

    config = {
        "tokenizer": {
            "_component_": "torchtune.models.llama3.llama3_tokenizer",
            "path": tokenizer_path,
            "max_seq_len": None,
        },
        "dataset": {
            "_component_": "torchtune.datasets.masked_text_completion_dataset",
            "source": "json",
            "data_files": dataset_filename,
            "column": "text",
            "split": "train",
        },
        "model": {
            "_component_": "torchtune.models.llama3_1.lora_llama3_1_8b",
            "lora_attn_modules": ["q_proj", "v_proj", "output_proj"],
            "apply_lora_to_mlp": True,
            "apply_lora_to_output": False,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
        "checkpointer": {
            "_component_": "torchtune.training.FullModelHFCheckpointer",
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_files": CHECKPOINT_FILES,
            "output_dir": output_dir,
            "recipe_checkpoint": None,
            "model_type": "LLAMA3",
        },
        "save_adapter_weights_only": True,
        "resume_from_checkpoint": False,
        "metric_logger": {
            "_component_": "torchtune.training.metric_logging.DiskLogger",
            "log_dir": output_dir,
        },
        "log_every_n_steps": 1,
        "seed": 42,
        "shuffle": True,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_steps_per_epoch": None,
        "optimizer": {
            "_component_": "torch.optim.AdamW",
            "lr": lr,
            "eps": 1e-8,
        },
        "lr_scheduler": {
            "_component_": "torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup",
            "num_warmup_steps": warmup_steps,
            "num_training_steps": total_steps,
        },
        "loss": {"_component_": "torch.nn.CrossEntropyLoss"},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "bf16",
        "output_dir": output_dir,
        "enable_activation_checkpointing": True,
        "enable_activation_offloading": False,
        "gradient_accumulation_steps": 1,
        "optimizer_in_bwd": True,
        "compile": False,
    }

    config_filename = os.path.join(output_dir, f"shared_config.yaml")
    with open(config_filename, "w") as f:
        yaml.dump(config, f)

    return config_filename


def finetune_with_torchtune(config_filename):
    """
    Run the 'tune' command from torchtune.  
    If 'tune' is not on your PATH, update this command accordingly.
    """
    cmd = f"tune run lora_finetune_single_device --config {config_filename}"
    print("Running torchtune command:", cmd)
    ret = os.system(cmd)
    if ret != 0:
        print(f"Torchtune command failed for config: {config_filename}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Perform LoRA-based finetuning on multiple BBH tasks using a single shared adapter."
    )
    parser.add_argument("--exp_name", type=str, default="shared_finetune",
                        help="Experiment name for logging.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the LLM model directory.")
    parser.add_argument("--output_file", type=str,
                        default="shared_finetune_results.json",
                        help="File to save the overall experiment results.")
    parser.add_argument("--task_start", type=int, default=0,
                        help="Start index (inclusive) for tasks to finetune/evaluate.")
    parser.add_argument("--task_end", type=int, default=27,
                        help="End index (exclusive) for tasks to finetune/evaluate.")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of few-shot examples per task (if available).")
    # Hyperparameter arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for finetuning.")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of epochs for finetuning.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling examples.")
    args = parser.parse_args()

    # Select only the tasks in the specified range
    all_tasks = list(TASKS.items())
    tasks_subset = all_tasks[args.task_start:args.task_end]
    if not tasks_subset:
        print("No tasks in the specified range. Exiting.")
        sys.exit(0)

    print("\n=== Finetuning with the following hyperparameters ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"k (few-shot examples per task): {args.k}\n")

    # 1) Build one combined dataset
    combined_data = []
    tasks_in_dataset = []

    for task, info in tasks_subset:
        task_file = f"external/BIG-Bench-Hard/bbh/{task}.json"
        if not os.path.exists(task_file):
            print(f"Skipping {task}, JSON not found.")
            continue

        try:
            with open(task_file, "r") as f:
                question_data = json.load(f)
        except Exception as exc:
            print(f"Error loading {task}: {exc}")
            continue

        all_examples = question_data.get("examples", [])
        if len(all_examples) < args.k:
            # Not enough data for k-shot
            print(f"Skipping {task}, not enough examples (required: {args.k}, found: {len(all_examples)}).")
            continue

        # Shuffle and pick first k
        random.seed(args.seed)
        random.shuffle(all_examples)
        few_shot_examples = all_examples[:args.k]

        for ex in few_shot_examples:
            # Same prompt logic as in e2e.py
            text = f"{info['task_prompt']} {info['answer_format']}\n\nQ: {ex['input']}\nA: {ex['target']}"
            combined_data.append({"text": text})

        tasks_in_dataset.append(task)

    if not tasks_in_dataset:
        print("No tasks qualified (all have < k examples or not found). Exiting.")
        sys.exit(0)

    # Write the combined dataset
    dataset_filename = f"/tmp/shared_{args.exp_name}_dataset.json"
    with open(dataset_filename, "w") as fjson:
        json.dump(combined_data, fjson)
    print(f"Combined dataset created with {len(combined_data)} total samples across {len(tasks_in_dataset)} tasks.")

    # Prepare output_dir for the single shared LoRA adapter
    output_dir = f"/tmp/shared_{args.exp_name}_adapter"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 2) Create a single torchtune config and finetune
    config_filename = create_torchtune_config(
        MODEL_DIR=args.model_dir,
        dataset_filename=dataset_filename,
        output_dir=output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        k=args.k,
        tasks_in_dataset=tasks_in_dataset,
    )

    print("\n=== Starting finetuning for all tasks combined... ===")
    ft_start_time = time.perf_counter()
    finetune_with_torchtune(config_filename)
    ft_end_time = time.perf_counter()
    ft_duration = ft_end_time - ft_start_time
    print(f"Finetuning done in {ft_duration:.2f}s.")

    # Cleanup dataset
    if os.path.exists(dataset_filename):
        try:
            os.remove(dataset_filename)
            print(f"Deleted combined dataset file: {dataset_filename}")
        except Exception as exc:
            print(f"Could not remove dataset file {dataset_filename}: {exc}")

    # Free up memory from the python side (especially if running in the same process)
    gc.collect()
    torch.cuda.empty_cache()

    # 3) Load base model once with enable_lora=True, evaluate each task
    eval_start_time = time.perf_counter()
    print("\n=== Loading base LLM with LoRA enabled for evaluation... ===")

    llm = LLM(
        model=args.model_dir,
        enable_lora=True,
        max_model_len=1024,
        max_lora_rank=args.lora_rank
    )

    evaluation_results = []
    # We will load the single adapter once, using LoRARequest
    lora_request = LoRARequest(
        lora_name="shared_adapter",
        lora_int_id=42,       # any unique integer ID
        lora_path=output_dir, # path to the shared LoRA weights
    )

    # Evaluate each task that we actually used for training
    for task in tasks_in_dataset:
        task_file = f"external/BIG-Bench-Hard/bbh/{task}.json"
        if not os.path.exists(task_file):
            print(f"Skipping evaluation for task: {task}, JSON not found.")
            continue

        with open(task_file, "r") as f:
            question_data = json.load(f)

        all_examples = question_data.get("examples", [])
        if len(all_examples) < args.k:
            print(f"Skipping {task}, not enough examples for evaluation.")
            continue

        # Evaluate only on the data beyond k
        random.seed(args.seed)
        random.shuffle(all_examples)  # ensure consistent ordering with the training selection
        evaluation_examples = all_examples[args.k:]
        questions = [ex["input"] for ex in evaluation_examples]
        targets = [ex["target"] for ex in evaluation_examples]

        # Prepare prompt info
        info = TASKS[task]
        task_prompt = info["task_prompt"]
        answer_format = info["answer_format"]
        generation_length = info["generation_length"]

        if not questions:
            # No data left to evaluate after k
            print(f"No leftover evaluation examples for task: {task}")
            continue

        # Run inference with the single shared LoRA
        t0 = time.perf_counter()
        ft_preds = inference_vllm(
            llm,
            questions,
            max_new_tokens=generation_length,
            task_prompt=task_prompt,
            answer_format=answer_format,
            lora_request=lora_request
        )
        t1 = time.perf_counter()
        eval_time = t1 - t0

        ft_accuracy = compute_accuracy(ft_preds, targets)

        # Store a few example predictions
        examples = []
        for i in range(min(5, len(questions))):
            examples.append({
                "question": questions[i],
                "prediction": ft_preds[i],
                "true_answer": targets[i].lower()
            })

        # Collect result for this task
        evaluation_results.append({
            "task": task,
            f"{args.exp_name}_accuracy": round(ft_accuracy, 4),
            "eval_time": round(eval_time, 4),
            "examples": examples
        })

        print(f"Task: {task}, accuracy={ft_accuracy:.2f}%, eval_time={eval_time:.2f}s")

    eval_end_time = time.perf_counter()
    eval_duration = eval_end_time - eval_start_time
    total_time = ft_duration + eval_duration
    print(f"\nEvaluation finished in {eval_duration:.2f}s.")
    print(f"Total time (finetune + eval): {total_time:.2f}s")

    # 4) Save results
    with open(args.output_file, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"Saved overall experiment results to {args.output_file}.")

    # 5) Cleanup the shared LoRA adapter directory
    print("\n=== Cleaning up the shared finetuned adapter directory ===")
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"Deleted adapter directory: {output_dir}")
        except Exception as exc:
            print(f"Could not remove adapter directory {output_dir}: {exc}")
    print("Cleanup completed.")


if __name__ == "__main__":
    main()
