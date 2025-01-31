#!/usr/bin/env python3

"""
shared_ttt.py

Implements "Shared TTT": instead of training a separate LoRA adapter for each task,
aggregate all tasks' few-shot examples in a single dataset. Specifically, for each
task, you create `num_training_steps` random permutations of that task's k-shot
examples (one permutation per training sample). Then combine them across tasks.

Finally, train exactly ONE LoRA adapter on this big dataset, and evaluate it on
each task's leftover evaluation set.

Usage example:
    python shared_ttt.py --model_dir /path/to/model \
        --k 10 --num_training_steps 40 --shuffle True \
        --batch_size 1 --epochs 2 --lr 5e-5 --lora_rank 64 --lora_alpha 32
"""

import os
import sys
import json
import time
import random
import shutil
import argparse
import gc
from collections import Counter

import torch
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

def majority_vote(answers):
    """
    Given a list of answers (strings), pick the one that appears most frequently.
    If there's a tie, pick whichever candidate is lexicographically smallest.
    """
    counter = Counter(answers)
    max_count = max(counter.values())
    winners = [ans for ans, cnt in counter.items() if cnt == max_count]
    winners.sort()
    return winners[0]

def build_inference_prompt(
    correct_examples: list,
    leave_one_out: bool = False,
    shuffle_examples: bool = False
) -> str:
    """
    Build a single few-shot prompt prefix from the list of correct examples (Q, A).
    If leave_one_out=True, we exclude the *last* example in 'correct_examples'.
    If shuffle_examples=True, we shuffle them first.

    Returns a string such as:
       Q: question_1
       A: answer_1

       Q: question_2
       A: answer_2
       ...
    """
    if not correct_examples:
        return ""

    examples_to_use = correct_examples[:]
    if shuffle_examples:
        random.shuffle(examples_to_use)

    if leave_one_out and len(examples_to_use) > 1:
        # Exclude the last example as "hold-out"
        examples_to_use = examples_to_use[:-1]

    prefix = ""
    for (q, a) in examples_to_use:
        prefix += f"Q: {q}\nA: {a}\n\n"
    return prefix

def create_combined_dataset(
    task_dict,
    k: int,
    num_training_steps: int,
    dataset_filename: str,
    shuffle_examples: bool
):
    """
    Build one large dataset that has:
      - For each task, `num_training_steps` training samples.
      - Each training sample is a random permutation of that task's k examples.

    So total = (#tasks) * (num_training_steps) samples.

    Each sample's text format:
       <task_prompt> <answer_format>

       Q: ...
       A: ...

       Q: ...
       A: ...
       ...
    """
    data_samples = []

    for task_name, task_info in task_dict.items():
        correct_examples = task_info["correct_examples"]  # List of (q, a)
        task_prompt = task_info["task_prompt"]
        answer_format = task_info["answer_format"]

        if not correct_examples:
            continue

        # For each task, repeat num_training_steps times
        for _ in range(num_training_steps):
            perm = correct_examples[:]
            if shuffle_examples:
                random.shuffle(perm)

            # Build text
            text = f"{task_prompt} {answer_format}\n\n"
            for (q, a) in perm:
                text += f"Q: {q}\nA: {a}\n\n"

            data_samples.append({"text": text})

    with open(dataset_filename, "w") as f:
        json.dump(data_samples, f)

def create_torchtune_config(
    model_dir: str,
    dataset_filename: str,
    output_dir: str,
    batch_size: int,
    epochs: int,
    lr: float,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    dataset_type: str = "text_completion_dataset"
) -> str:
    """
    Create a YAML config for Torchtune finetuning (LoRA).
    We'll do multiple samples => steps_per_epoch = (#samples / batch_size).
    """
    import yaml

    tokenizer_path = os.path.join(model_dir, "original", "tokenizer.model")
    checkpoint_dir = model_dir

    # Count samples in dataset
    try:
        with open(dataset_filename, "r") as f:
            data = json.load(f)
        n_samples = len(data)
    except Exception:
        n_samples = 0

    steps_per_epoch = max(1, n_samples // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)

    config = {
        "tokenizer": {
            "_component_": "torchtune.models.llama3.llama3_tokenizer",
            "path": tokenizer_path,
            "max_seq_len": None,
        },
        "dataset": {
            "_component_": f"torchtune.datasets.{dataset_type}",
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
        "loss": {
            "_component_": "torch.nn.CrossEntropyLoss"
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "bf16",
        "output_dir": output_dir,
        "enable_activation_checkpointing": True,
        "enable_activation_offloading": False,
        "gradient_accumulation_steps": 1,
        "optimizer_in_bwd": True,
        "compile": False,
    }

    config_filename = os.path.join(output_dir, "shared_ttt_config.yaml")
    with open(config_filename, "w") as f:
        yaml.dump(config, f)

    return config_filename

def finetune_with_torchtune(config_filename: str):
    """
    Run the torchtune finetuning command.
    """
    cmd = f"tune run lora_finetune_single_device --config {config_filename}"
    print("Running command:", cmd)
    ret = os.system(cmd)
    if ret != 0:
        print(f"[Shared TTT] Torchtune command failed with config: {config_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Shared TTT: Combine k-shot from all tasks into a big dataset (num_training_steps permutations per task), train one LoRA, then evaluate per task."
    )
    parser.add_argument("--exp_name", type=str, default="shared_ttt",
                        help="Experiment name.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the model directory (base model).")
    parser.add_argument("--output_file", type=str, default="shared_ttt_results.json",
                        help="Where to store final JSON results.")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of few-shot examples per task.")
    parser.add_argument("--dataset_type", type=str, default="text_completion_dataset",
                        help="Dataset type for Torchtune finetuning.")
    parser.add_argument("--num_training_steps", type=int, default=5,
                        help="Number of random shuffles per task. If each of the 27 tasks has k examples, we make num_training_steps samples per task => total #samples = 27 * num_training_steps.")
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == "true"), default=True,
                        help="Whether to shuffle the few-shot examples for each training sample, and whether to shuffle them at inference time as well.")
    parser.add_argument("--majority_vote", type=lambda x: (str(x).lower() == "true"), default=False,
                        help="If True, do multiple permutations at inference and pick the majority answer.")
    parser.add_argument("--vote_permutations", type=int, default=5,
                        help="Number of random permutations for majority-vote inference.")
    parser.add_argument("--leave_one_out", type=lambda x: (str(x).lower() == "true"), default=False,
                        help="If True, exclude the last example from the prefix at inference time.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="LoRA training batch size.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs for LoRA training.")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for LoRA training.")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling examples.")
    args = parser.parse_args()

    # Load tasks, gather k-shot, keep leftover for eval
    task_dict = {}

    # For each task in TASKS, we read the JSON and store:
    #   correct_examples -> list of (q, a) for training
    #   eval_questions, eval_targets for leftover
    for task_name, info in TASKS.items():
        task_file = f"external/BIG-Bench-Hard/bbh/{task_name}.json"
        if not os.path.exists(task_file):
            print(f"[Shared TTT] Skipping {task_name}, file not found: {task_file}")
            continue

        # Load JSON
        try:
            with open(task_file, "r") as f:
                task_json = json.load(f)
        except Exception as e:
            print(f"[Shared TTT] Error loading {task_file}: {e}")
            continue

        all_examples = task_json.get("examples", [])
        if not all_examples:
            print(f"[Shared TTT] Skipping {task_name}, no examples in JSON.")
            continue

        random.seed(args.seed)
        random.shuffle(all_examples)

        if len(all_examples) <= args.k:
            print(f"[Shared TTT] Skipping {task_name}, not enough examples for k={args.k}.")
            continue

        # k-shot for "correct_examples", rest for eval
        train_data = all_examples[:args.k]
        eval_data = all_examples[args.k:]

        correct_examples = []
        for ex in train_data:
            q = ex["input"]
            a = ex["target"]
            correct_examples.append((q, a))

        eval_questions = [e["input"] for e in eval_data]
        eval_targets = [e["target"] for e in eval_data]

        task_dict[task_name] = {
            "correct_examples": correct_examples,
            "eval_questions": eval_questions,
            "eval_targets": eval_targets,
            "task_prompt": info.get("task_prompt", ""),
            "answer_format": info.get("answer_format", ""),
            "generation_length": info.get("generation_length", 128),
        }

    if not task_dict:
        print("\n[Shared TTT] No tasks to process. Exiting.")
        sys.exit(0)

    print(
        f"\n=== Shared TTT: Loaded {len(task_dict)} tasks with k={args.k}. "
        f"num_training_steps={args.num_training_steps}. ==="
    )

    # Prepare single output_dir for the LoRA adapter
    shared_output_dir = f"/tmp/{args.exp_name}_adapter"
    if os.path.exists(shared_output_dir):
        shutil.rmtree(shared_output_dir)
    os.makedirs(shared_output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # PHASE 2: Build combined dataset & Finetune
    # -------------------------------------------------------------------------
    ft_time = 0.0
    if args.num_training_steps == 0:
        print("=== PHASE 2: Skipping finetuning (num_training_steps=0) ===")
    else:
        dataset_filename = os.path.join(shared_output_dir, "shared_ttt_dataset.json")

        # Build combined dataset:
        create_combined_dataset(
            task_dict=task_dict,
            k=args.k,
            num_training_steps=args.num_training_steps,
            dataset_filename=dataset_filename,
            shuffle_examples=args.shuffle
        )

        # Create config & finetune
        config_filename = create_torchtune_config(
            model_dir=args.model_dir,
            dataset_filename=dataset_filename,
            output_dir=shared_output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            dataset_type=args.dataset_type
        )

        print("\n=== PHASE 2: Single LoRA Finetuning for combined tasks ===")
        ft_start = time.perf_counter()
        finetune_with_torchtune(config_filename)
        ft_end = time.perf_counter()
        ft_time = ft_end - ft_start
        print(f"[Shared TTT] Finetuning completed in {ft_time:.2f}s")

        # Cleanup intermediate dataset file
        if os.path.exists(dataset_filename):
            try:
                os.remove(dataset_filename)
            except Exception as exc:
                print(f"[Shared TTT] Could not remove dataset file {dataset_filename}: {exc}")

    # -------------------------------------------------------------------------
    # PHASE 3: Evaluate the single adapter on each task
    # -------------------------------------------------------------------------
    print("\n=== PHASE 3: Evaluation with vLLM + Shared LoRA ===")
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize LLM for evaluation with LoRA enabled
    llm_eval = LLM(
        model=args.model_dir,
        enable_lora=True,
        max_model_len=4096,
        max_lora_rank=args.lora_rank,
    )

    # Prepare LoRARequest if we actually finetuned
    lora_request = None
    if args.num_training_steps > 0 and os.path.isdir(shared_output_dir):
        lora_request = LoRARequest(
            lora_name="shared_adapter",
            lora_int_id=1,
            lora_path=shared_output_dir
        )

    results = []

    for task_name, data_dict in task_dict.items():
        gen_len = data_dict["generation_length"]
        task_prompt = data_dict["task_prompt"]
        answer_format = data_dict["answer_format"]
        correct_examples = data_dict["correct_examples"]
        eval_questions = data_dict["eval_questions"]
        eval_targets = data_dict["eval_targets"]

        if not eval_questions:
            print(f"[PHASE 3] {task_name}: No eval data found, skipping.")
            continue

        # Evaluate
        eval_start = time.perf_counter()

        if not args.majority_vote:
            prefix = build_inference_prompt(
                correct_examples,
                leave_one_out=args.leave_one_out,
                shuffle_examples=args.shuffle
            )
            eval_outputs = inference_vllm(
                llm=llm_eval,
                prompts=eval_questions,
                max_new_tokens=gen_len,
                task_prompt=task_prompt,
                answer_format=answer_format,
                few_shot_prompt_prefix=prefix,
                lora_request=lora_request
            )
            preds = eval_outputs if eval_outputs else []
        else:
            # Majority vote
            vote_preds = []
            for _ in range(args.vote_permutations):
                prefix = build_inference_prompt(
                    correct_examples,
                    leave_one_out=args.leave_one_out,
                    shuffle_examples=args.shuffle
                )
                batch_outputs = inference_vllm(
                    llm=llm_eval,
                    prompts=eval_questions,
                    max_new_tokens=gen_len,
                    task_prompt=task_prompt,
                    answer_format=answer_format,
                    few_shot_prompt_prefix=prefix,
                    lora_request=lora_request
                )
                vote_preds.append(batch_outputs if batch_outputs else [])

            final_preds = []
            for q_idx in range(len(eval_questions)):
                candidate_answers = [
                    vote_preds[p][q_idx] for p in range(args.vote_permutations)
                ]
                final_preds.append(majority_vote(candidate_answers))
            preds = final_preds

        eval_time = time.perf_counter() - eval_start
        acc = compute_accuracy(preds, eval_targets)

        examples = [
            {
                "question": eval_questions[i],
                "prediction": preds[i],
                "true_answer": eval_targets[i].lower(),
            }
            for i in range(min(5, len(eval_questions)))
        ]

        results.append({
            "task": task_name,
            f"{args.exp_name}_accuracy": round(acc, 4),
            "ft_time": round(ft_time, 4),
            "eval_time": round(eval_time, 4),
            "examples": examples
        })

        print(
            f"[PHASE 3] Task={task_name}, accuracy={acc:.2f}%, "
            f"ft_time={ft_time:.2f}s, eval_time={eval_time:.2f}s"
        )

    # Cleanup
    del llm_eval
    gc.collect()
    torch.cuda.empty_cache()

    # Save results
    try:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Shared TTT] Results saved to {args.output_file}.")
    except Exception as e:
        print(f"[Shared TTT] Error saving results: {e}")

    # Remove LoRA adapter directory if you like
    if os.path.exists(shared_output_dir):
        try:
            shutil.rmtree(shared_output_dir)
        except Exception as exc:
            print(f"[Shared TTT] Warning: could not remove adapter dir {shared_output_dir}: {exc}")

    print("[Shared TTT] Done. Goodbye!")


if __name__ == "__main__":
    main()
