#!/usr/bin/env python3

"""
ttt.py

Implements the main TTT (Test-Time Training) method.
1. Load tasks from the BIG-Bench-Hard (BBH) dataset.
2. Creating a multi-sample training dataset for LoRA finetuning, using few-shot examples.
3. Fine-tune the language model with the Torchtune library and LoRA adapters.
4. Performing inference and evaluation with the fine-tuned model, optionally using majority-vote.
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
    'model-00001-of-00004.safetensors',
    'model-00002-of-00004.safetensors',
    'model-00003-of-00004.safetensors',
    'model-00004-of-00004.safetensors'
]


def create_ttt_dataset(
    prefix: str,
    correct_examples: list,
    num_training_steps: int,
    dataset_filename: str,
    shuffle_examples: bool
):
    """
    Create a multi-sample JSON dataset for Torchtune finetuning, with no chain-of-thought.

    - If shuffle_examples=True, each item will be a random permutation of correct_examples.
    - If shuffle_examples=False, each item uses the same order of correct_examples.

    We prepend `prefix` to each sample. Then, for each (Q, A) pair, we append:
       Q: ...
       A: ...
    """
    data_samples = []
    for _ in range(num_training_steps):
        if shuffle_examples:
            perm = correct_examples[:]
            random.shuffle(perm)
            ex_list = perm
        else:
            ex_list = correct_examples

        text = prefix
        for (q, a) in ex_list:
            # Each example has question/answer only (no chain-of-thought)
            text += f"Q: {q}\nA: {a}\n\n"
        data_samples.append({"text": text})

    with open(dataset_filename, 'w') as f:
        json.dump(data_samples, f)


def build_inference_prompt(
    correct_examples: list,
    leave_one_out: bool = False,
    shuffle_examples: bool = False
) -> str:
    """
    Build a single few-shot prompt prefix from the list of correct examples (Q, A).

    If leave_one_out=True, we exclude the *last* example in 'correct_examples'.
    If shuffle_examples=True, we shuffle them first.
    """
    if not correct_examples:
        return ""

    examples_to_use = correct_examples[:]
    if shuffle_examples:
        random.shuffle(examples_to_use)

    if leave_one_out and len(examples_to_use) > 1:
        examples_to_use = examples_to_use[:-1]

    prefix = ""
    for (q, a) in examples_to_use:
        prefix += f"Q: {q}\nA: {a}\n\n"
    return prefix


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


def create_torchtune_config(
    model_dir: str,
    dataset_type: str,
    task: str,
    dataset_filename: str,
    output_dir: str,
    batch_size: int,
    epochs: int,
    lr: float,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float
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
        with open(dataset_filename, 'r') as f:
            data = json.load(f)
        n_samples = len(data)
    except Exception:
        n_samples = 0

    steps_per_epoch = max(1, n_samples // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)

    config = {
        'tokenizer': {
            '_component_': 'torchtune.models.llama3.llama3_tokenizer',
            'path': tokenizer_path,
            'max_seq_len': None
        },
        'dataset': {
            '_component_': f'torchtune.datasets.{dataset_type}',
            'source': 'json',
            'data_files': dataset_filename,
            'column': 'text',
            'split': 'train'
        },
        'model': {
            '_component_': 'torchtune.models.llama3_1.lora_llama3_1_8b',
            'lora_attn_modules': ['q_proj', 'v_proj', 'output_proj'],
            'apply_lora_to_mlp': True,
            'apply_lora_to_output': False,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout
        },
        'checkpointer': {
            '_component_': 'torchtune.training.FullModelHFCheckpointer',
            'checkpoint_dir': checkpoint_dir,
            'checkpoint_files': CHECKPOINT_FILES,
            'output_dir': output_dir,
            'recipe_checkpoint': None,
            'model_type': 'LLAMA3'
        },
        'save_adapter_weights_only': True,
        'resume_from_checkpoint': False,
        'metric_logger': {
            '_component_': 'torchtune.training.metric_logging.DiskLogger',
            'log_dir': output_dir
        },
        'log_every_n_steps': 1,
        'seed': 42,
        'shuffle': True,
        'batch_size': batch_size,
        'epochs': epochs,
        'max_steps_per_epoch': None,
        'optimizer': {
            '_component_': 'torch.optim.AdamW',
            'lr': lr,
            'eps': 1e-8
        },
        'lr_scheduler': {
            '_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup',
            'num_warmup_steps': warmup_steps,
            'num_training_steps': total_steps
        },
        'loss': {
            '_component_': 'torch.nn.CrossEntropyLoss'
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': 'bf16',
        'output_dir': output_dir,
        'enable_activation_checkpointing': True,
        'enable_activation_offloading': False,
        'gradient_accumulation_steps': 1,
        'optimizer_in_bwd': True,
        'compile': False
    }

    config_filename = os.path.join(output_dir, f"{task}_config.yaml")
    with open(config_filename, 'w') as f:
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
        print(f"[TTT] Torchtune command failed with config: {config_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="TTT: Test-Time Training with multi-sample training & optional majority-vote inference, loading data from tasks_k{k}.json."
    )
    parser.add_argument("--exp_name", type=str, default="ttt",
                        help="Experiment name.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the model directory (base model).")
    parser.add_argument("--output_file", type=str, default="ttt_results.json",
                        help="Where to store final JSON results.")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of few-shot examples.")
    parser.add_argument("--dataset_type", type=str, default="text_completion_dataset",
                        help="Dataset type for Torchtune finetuning.")
    parser.add_argument("--num_training_steps", type=int, default=5,
                        help="Number of random shuffles to create for the training dataset. If 0, skip finetuning.")
    parser.add_argument("--shuffle", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to shuffle the few-shot examples when building each training sample and/or each inference prefix.")
    parser.add_argument("--majority_vote", type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="If True, we do multiple permutations at inference and pick the majority answer.")
    parser.add_argument("--vote_permutations", type=int, default=5,
                        help="Number of random permutations for majority-vote inference.")
    parser.add_argument("--leave_one_out", type=lambda x: (str(x).lower() == 'true'), default=False,
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

    # -------------------------------------------------------------------------
    # PHASE 1: Load tasks from external/BIG-Bench-Hard/bbh/<task>.json
    # -------------------------------------------------------------------------
    task_dict = {}

    # Set the random seed for reproducible shuffling

    # Loop over the tasks defined in src.tasks.TASKS
    for task_name, info in TASKS.items():
        # Path to the BBH JSON
        task_file = f"external/BIG-Bench-Hard/bbh/{task_name}.json"
        if not os.path.exists(task_file):
            print(f"[TTT] Skipping {task_name} because file not found: {task_file}")
            continue

        # Load the BBH JSON
        try:
            with open(task_file, "r") as f:
                task_json = json.load(f)
        except Exception as e:
            print(f"[TTT] Error loading {task_file}: {e}")
            continue

        all_examples = task_json.get("examples", [])
        if not all_examples:
            print(f"[TTT] Skipping {task_name} because no examples found in JSON.")
            continue

        # Shuffle all examples with the chosen seed
        random.seed(args.seed)
        random.shuffle(all_examples)

        # If we don't have enough examples for k
        if len(all_examples) <= args.k:
            print(f"[TTT] Skipping {task_name} for k={args.k}, not enough examples.")
            continue

        # The first k become the training set ("correct_examples")
        train_data = []
        for ex in all_examples[:args.k]:
            q = ex["input"]
            a = ex["target"]
            # We'll store these for finetuning in the same format as "train" was before
            train_data.append({"question": q, "answer": a})

        # The rest become the eval set
        eval_data = []
        for ex in all_examples[args.k:]:
            q = ex["input"]
            a = ex["target"]
            eval_data.append({"question": q, "answer": a})

        # Gather info from TASKS
        generation_length = info.get("generation_length", 128)
        answer_format     = info.get("answer_format", "")
        task_prompt       = info.get("task_prompt", "")

        # Convert train data to the "correct_examples" format: List of (q, a)
        correct_examples = []
        for ex in train_data:
            q = ex["question"]
            a = ex["answer"]
            correct_examples.append((q, a))

        # Convert eval data to separate question list and target list
        eval_questions = [e["question"] for e in eval_data]
        eval_targets   = [e["answer"]   for e in eval_data]

        # Create output_dir for potential LoRA adapters
        output_dir = f"/tmp/{task_name}_{args.exp_name}_adapter"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Populate our task_dict
        task_dict[task_name] = {
            "correct_examples": correct_examples,
            "eval_questions": eval_questions,
            "eval_targets": eval_targets,
            "info": info,
            "output_dir": output_dir,
            "generation_length": generation_length,
            "answer_format": answer_format,
            "task_prompt": task_prompt
        }

    if not task_dict:
        print("\n[TTT] No tasks to process. Exiting.")
        sys.exit(0)

    print(f"\n=== TTT: Loaded {len(task_dict)} tasks from BBH with k={args.k} ===")
    print(f"Finetuning steps: {args.num_training_steps} (0 => skip), majority_vote={args.majority_vote}, leave_one_out={args.leave_one_out}\n")

    # -------------------------------------------------------------------------
    # PHASE 2: LoRA Finetuning on multi-sample dataset (unless num_training_steps=0)
    # -------------------------------------------------------------------------
    ft_times = {}
    if args.num_training_steps == 0:
        print("=== PHASE 2: Skipping finetuning because num_training_steps=0 ===")
        for tname in task_dict:
            ft_times[tname] = 0.0
    else:
        print("=== PHASE 2: Multi-sample LoRA Finetuning ===")
        for task_name, data_dict in task_dict.items():
            correct_examples = data_dict["correct_examples"]
            if not correct_examples:
                print(f"[PHASE 2] No correct examples for {task_name}. Skipping finetune.")
                ft_times[task_name] = 0.0
                continue

            # Construct an optional prefix (task instruction, etc.)
            prefix = (
                f"{data_dict['task_prompt']} "
                f"{data_dict['answer_format']}\n\n"
            )

            dataset_filename = os.path.join(data_dict["output_dir"], f"{task_name}_ttt_dataset.json")

            # Build the multi-sample dataset with num_training_steps random shuffles
            create_ttt_dataset(
                prefix=prefix,
                correct_examples=correct_examples,
                num_training_steps=args.num_training_steps,
                dataset_filename=dataset_filename,
                shuffle_examples=args.shuffle
            )

            # Create Torchtune config
            config_filename = create_torchtune_config(
                model_dir=args.model_dir,
                dataset_type=args.dataset_type,
                task=task_name,
                dataset_filename=dataset_filename,
                output_dir=data_dict["output_dir"],
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout
            )

            # Finetune
            ft_start = time.perf_counter()
            finetune_with_torchtune(config_filename)
            ft_end = time.perf_counter()
            ft_time = ft_end - ft_start
            ft_times[task_name] = ft_time

            # Cleanup
            if os.path.exists(dataset_filename):
                try:
                    os.remove(dataset_filename)
                except Exception as exc:
                    print(f"[TTT] Could not remove dataset file {dataset_filename}: {exc}")

            print(f"[PHASE 2] Finished finetuning for {task_name} in {ft_time:.2f}s")

    # -------------------------------------------------------------------------
    # PHASE 3: Evaluate with vLLM + LoRA (if present) + optional majority vote
    # -------------------------------------------------------------------------
    print("\n=== PHASE 3: Evaluation with vLLM + LoRA ===")

    gc.collect()
    torch.cuda.empty_cache()

    # Initialize LLM for evaluation with LoRA enabled
    llm_eval = LLM(
        model=args.model_dir,
        enable_lora=True,
        max_model_len=4096,
        max_lora_rank=args.lora_rank,
    )

    ttt_results = []
    lora_id = 1

    for task_name, data_dict in task_dict.items():
        generation_length = data_dict["generation_length"]
        answer_format = data_dict["answer_format"]
        task_prompt = data_dict["task_prompt"]
        correct_examples = data_dict["correct_examples"]
        eval_questions = data_dict["eval_questions"]
        eval_targets = data_dict["eval_targets"]
        output_dir = data_dict["output_dir"]

        # If no correct examples, there's no adapter for this task => evaluate base model
        if not correct_examples:
            print(f"[PHASE 3] {task_name}: No correct examples -> evaluating base model (no adapter).")
            eval_start = time.perf_counter()
            eval_outputs = inference_vllm(
                llm=llm_eval,
                prompts=eval_questions,
                max_new_tokens=generation_length,
                task_prompt=task_prompt,
                answer_format=answer_format,
                few_shot_prompt_prefix="",
                lora_request=None
            )
            preds = eval_outputs if eval_outputs else []
            eval_time = time.perf_counter() - eval_start
            acc = compute_accuracy(preds, eval_targets)
            ttt_results.append({
                "task": task_name,
                f"{args.exp_name}_accuracy": round(acc, 4),
                "ft_time": 0.0,
                "eval_time": round(eval_time, 4),
                "examples": []
            })
            continue

        # Otherwise, load the LoRA adapter (if we actually finetuned)
        lora_request = None
        if args.num_training_steps > 0 and os.path.isdir(output_dir):
            lora_request = LoRARequest(
                lora_name=f"{task_name}_adapter_{lora_id}",
                lora_int_id=lora_id,
                lora_path=output_dir
            )
            lora_id += 1

        eval_start = time.perf_counter()
        if not args.majority_vote:
            # Single permutation
            prefix = build_inference_prompt(
                correct_examples,
                leave_one_out=args.leave_one_out,
                shuffle_examples=args.shuffle
            )
            eval_outputs = inference_vllm(
                llm=llm_eval,
                prompts=eval_questions,
                max_new_tokens=generation_length,
                task_prompt=task_prompt,
                answer_format=answer_format,
                few_shot_prompt_prefix=prefix,
                lora_request=lora_request
            )
            preds = eval_outputs if eval_outputs else []
        else:
            # Majority vote with multiple permutations
            vote_preds = []
            for _ in range(args.vote_permutations):
                prefix = build_inference_prompt(
                    correct_examples,
                    leave_one_out=args.leave_one_out,
                    shuffle_examples=True
                )
                batch_outputs = inference_vllm(
                    llm=llm_eval,
                    prompts=eval_questions,
                    max_new_tokens=generation_length,
                    task_prompt=task_prompt,
                    answer_format=answer_format,
                    few_shot_prompt_prefix=prefix,
                    lora_request=lora_request
                )
                vote_preds.append(batch_outputs if batch_outputs else [])

            # For each question, pick the majority answer
            final_preds = []
            for q_idx in range(len(eval_questions)):
                candidate_answers = [vote_preds[p][q_idx] for p in range(args.vote_permutations)]
                final_preds.append(majority_vote(candidate_answers))
            preds = final_preds

        eval_time = time.perf_counter() - eval_start
        # Compute accuracy
        acc = compute_accuracy(preds, eval_targets)

        # Prepare examples for logging (show only up to 5)
        examples = [
            {
                "question": eval_questions[i],
                "prediction": preds[i],
                "true_answer": eval_targets[i].lower()
            }
            for i in range(min(5, len(eval_questions)))
        ]

        ttt_results.append({
            "task": task_name,
            f"{args.exp_name}_accuracy": round(acc, 4),
            "ft_time": round(ft_times.get(task_name, 0.0), 4),
            "eval_time": round(eval_time, 4),
            "examples": examples
        })

        print(f"[PHASE 3] Task={task_name}, accuracy={acc:.2f}%, ft_time={ft_times.get(task_name, 0.0):.2f}s, eval_time={eval_time:.2f}s")

    # Cleanup the LLM
    del llm_eval
    gc.collect()
    torch.cuda.empty_cache()

    # Save results
    try:
        with open(args.output_file, 'w') as f:
            json.dump(ttt_results, f, indent=2)
        print(f"\n[TTT] Results saved to {args.output_file}.")
    except Exception as e:
        print(f"[TTT] Error saving results: {e}")

    # Remove LoRA adapter directories if any
    for task_name, data_dict in task_dict.items():
        out_dir = data_dict["output_dir"]
        if os.path.exists(out_dir):
            try:
                shutil.rmtree(out_dir)
            except Exception as exc:
                print(f"[TTT] Warning: could not remove adapter dir {out_dir}: {exc}")

    print("[TTT] Done. Goodbye!")


if __name__ == "__main__":
    main()
