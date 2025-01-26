# graphResultsAblations.py

import matplotlib.pyplot as plt
import json
import os

data_file = "../logs/archive/averages/allResults.json"
output_dir = "../plots"
os.makedirs(output_dir, exist_ok=True)

# Load the data file
with open(data_file, "r") as f:
    data = json.load(f)

tasks = data["per_task_results"]
aggregated = data["aggregated_statistics"]

# Tasks to plot (filtered subset)
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
filtered_tasks = [t for t in tasks if t["task"] in tasks_to_plot]

# Updated method list with new labels + reduced spacing
#   x-position,           Display Name,                JSON Key,                                                                                Bar Color
method_configs = [
    (0, "ICL",                  "FSL_10_accuracy",                                                                                   "#598392"),
    (1, "TTT",                  "ICFT_main_10_masked_inputs_text_completion_dataset_40_True_False_5_False_5_1_1e-4_64_64_0.05_accuracy", "#01161e"),
    # first dashed line at x=1.5
    (2, "ICL (Majority Vote)",  "FSL_10_maj_accuracy",                                                                               "#8ecae6"),
    (3, "TTT (Majority Vote)",  "ICFT_maj_10_masked_inputs_text_completion_dataset_40_True_True_5_False_5_1_1e-4_64_64_accuracy",      "#219ebc"),
    # second dashed line at x=3.5
    (4, "No Demonstration Loss","ICFT_ex_trainLast_10_masked_text_completion_dataset_40_True_False_5_False_5_1_1e-4_64_64_accuracy",   "#467a69"),
    (5, "Loss on Inputs and Outputs","ICFT_ex_trainAll_10_text_completion_dataset_40_True_False_5_False_5_1_1e-4_64_64_0.05_accuracy", "#315c4f"),
]

# Dashed lines after the 2nd bar (x=1.5) and after the 4th bar (x=3.5)
vertical_lines = [1.5, 3.5]

# Precompute x-positions & labels
x_positions   = [cfg[0] for cfg in method_configs]
method_labels = [cfg[1] for cfg in method_configs]

def format_task_name(task_name):
    """Replace underscores with spaces and capitalize each word."""
    return task_name.replace("_", " ").title()

def plot_bar_group(ax, task_or_agg, is_task=True):
    """
    Plots a bar chart for either:
      - a single task's data (is_task=True), or
      - the aggregated data (is_task=False).
    """
    # Retrieve accuracy values in the specified order
    values = []
    for (x, label, key, color) in method_configs:
        if is_task:
            # per-task dictionary: key -> numeric accuracy
            y = task_or_agg[key]
        else:
            # aggregated dictionary: key -> { "average": ... }
            y = task_or_agg[key]["average"]
        values.append((x, y, color))

    # Plot each bar at the correct x-position
    for (x_val, y_val, color) in values:
        ax.bar(x_val, y_val, color=color, width=0.8)

    # Draw vertical dashed lines to separate groups
    for vline in vertical_lines:
        ax.axvline(x=vline, color="black", linestyle="--", linewidth=1)

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=10)

    # Increase font size specifically for "ICL" and "TTT" labels
    for tick_label in ax.get_xticklabels():
        text = tick_label.get_text()
        if text in ["ICL", "TTT"]:
            tick_label.set_fontsize(14)  # bigger font for ICL/TTT

    # Generally want accuracy range 0% to 100%
    # ax.set_ylim(0, 100)


fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
axes = axes.flatten()

for idx, task in enumerate(filtered_tasks):
    ax = axes[idx]
    plot_bar_group(ax, task, is_task=True)
    ax.set_title(format_task_name(task["task"]), fontsize=16)

# Hide any unused subplots
for i in range(len(filtered_tasks), len(axes)):
    axes[i].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(output_dir, "tasks_results_custom_order.png"))
plt.close()

print(f"Saved filtered tasks plot -> {os.path.join(output_dir, 'tasks_results_custom_order.png')}")

plt.figure(figsize=(10, 6))
ax = plt.gca()
plot_bar_group(ax, aggregated, is_task=False)

plt.ylabel("Accuracy", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "average_results_custom_order.png"))
plt.close()

print(f"Saved average results plot -> {os.path.join(output_dir, 'average_results_custom_order.png')}")

fig, axes = plt.subplots(3, 9, figsize=(36, 12))
fig.subplots_adjust(hspace=0.6, wspace=0.6)
axes = axes.flatten()

for idx, task in enumerate(tasks):
    ax = axes[idx]
    plot_bar_group(ax, task, is_task=True)
    ax.set_title(format_task_name(task["task"]), fontsize=13)

# Hide any unused subplots
for i in range(len(tasks), len(axes)):
    axes[i].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(output_dir, "all_tasks_results_custom_order.png"))
plt.close()

print(f"Saved all tasks plot -> {os.path.join(output_dir, 'all_tasks_results_custom_order.png')}")
print("All plots saved successfully!")
