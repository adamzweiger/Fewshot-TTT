# graphResultsICLvTTT.py

import matplotlib.pyplot as plt
import json
import os

# Load the data file
data_file = "../logs/archive/averages/allResults.json"
output_dir = "../plots"
os.makedirs(output_dir, exist_ok=True)

with open(data_file, "r") as f:
    data = json.load(f)

tasks = data["per_task_results"]
aggregated = data["aggregated_statistics"]

# We only want these tasks
tasks_to_plot = [
    "date_understanding",
    "dyck_languages",
    "geometric_shapes",
    "hyperbaton",
    "movie_recommendation",
    "ruin_names",
    "snarks",
    "temporal_sequences",
]

# Filter the tasks
filtered_tasks = [t for t in tasks if t["task"] in tasks_to_plot]

# Define methods and their keys (only ICL and TTT)
methods = [
    ("ICL", "FSL_10_accuracy"),
    ("TTT", "ICFT_main_10_masked_inputs_text_completion_dataset_40_True_False_5_False_5_1_1e-4_64_64_0.05_accuracy"),
]

# Use only the third and fifth colors
colors = ["#598392", "#01161e"]

def format_task_name(task_name):
    # Replace underscores with spaces and capitalize each word
    return task_name.replace("_", " ").title()

# --- Create task-specific plots for the filtered tasks ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
axes = axes.flatten()

for idx, task in enumerate(filtered_tasks):
    ax = axes[idx]
    method_names = [m[0] for m in methods]
    method_values = [task[m[1]] for m in methods]

    ax.bar(method_names, method_values, color=colors)
    ax.set_title(format_task_name(task["task"]), fontsize=20)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha="right", fontsize=16)
    ax.set_ylim(0, 100)

# Hide any unused subplots if the number of tasks is less than the available axes
for i in range(len(filtered_tasks), len(axes)):
    axes[i].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(output_dir, "tasks_results_loss_ablations_filtered.png"))
plt.close()

# --- Create a plot for average results (across ALL tasks) ---
avg_results = [aggregated[m[1]]["average"] for m in methods]
method_names = [m[0] for m in methods]

plt.figure(figsize=(10, 6))
plt.bar(method_names, avg_results, color=colors)
plt.xticks(rotation=45, ha="right", fontsize=14)
plt.ylabel("Accuracy", fontsize=18)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "average_results_loss_ablations.png"))
plt.close()

# --- Create a plot for all tasks in a 3-row by 9-column layout ---
fig, axes = plt.subplots(3, 9, figsize=(36, 12))
fig.subplots_adjust(hspace=0.6, wspace=0.6)
axes = axes.flatten()

for idx, task in enumerate(tasks):
    ax = axes[idx]
    method_names = [m[0] for m in methods]
    method_values = [task[m[1]] for m in methods]

    ax.bar(method_names, method_values, color=colors)
    ax.set_title(format_task_name(task["task"]), fontsize=13)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 100)

# Hide any unused subplots if the number of tasks is less than the available axes
for i in range(len(tasks), len(axes)):
    axes[i].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(output_dir, "all_tasks_results.png"))
plt.close()

print(f"Plots saved in {output_dir}")
