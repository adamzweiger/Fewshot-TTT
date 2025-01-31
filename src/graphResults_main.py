import matplotlib.pyplot as plt
import json
import os
import numpy as np
import textwrap

# Load the data file
data_file = "../logs/archive/averages/allResults.json"
output_dir = "../plots"
os.makedirs(output_dir, exist_ok=True)

with open(data_file, "r") as f:
    data = json.load(f)

tasks = data["per_task_results"]
aggregated = data["aggregated_statistics"]
tasks_to_plot = ["dyck_languages", "ruin_names", "movie_recommendation", "hyperbaton","boolean_expressions"]

# Filter and manually reorder the tasks
filtered_tasks = sorted([t for t in tasks if t["task"] in tasks_to_plot], 
                                 key=lambda t: tasks_to_plot.index(t["task"]))



# Define methods and their keys
methods = [
    ("Zero-Shot", "ZSL_accuracy"),
    ("ICL", "FSL_10_accuracy"),
    ("TTT", "ICFT_main_10_masked_inputs_text_completion_dataset_40_True_False_5_False_5_1_1e-4_64_64_0.05_accuracy"),
]

# Generate a color palette for the 3 methods
colors = [
    "#EFECCA",
    "#598392",
    "#01161e"
]

def format_task_name(task_name):
    # Replace underscores with spaces and capitalize each word
    return task_name.replace("_", " ").title()

def wrap_label(label, width=10):
    """
    Wrap text into a maximum of two lines if it exceeds 'width' characters.
    """
    wrapped = textwrap.fill(label, width=width)
    lines = wrapped.split('\n')
    # If more than two lines result, truncate to just two
    if len(lines) > 2:
        lines = lines[:2]
    return "\n".join(lines)

plt.rcParams['font.family'] = 'serif'

# ------------------------------------------------------------------------------
# 1) Single Plot with 4 Groups (Each Group = 1 Task, 3 Bars = 3 Methods)
# ------------------------------------------------------------------------------
# Prepare data for grouped bar chart
x = np.arange(len(filtered_tasks))  # positions for each task
bar_width = 0.25

# Calculate a suitable y-limit
all_scores = []
for task in filtered_tasks:
    for _, method_key in methods:
        all_scores.append(task[method_key])
max_score = max(all_scores)
ylim_top = max(100, max_score * 1.15)  # a bit of padding above the tallest bar

plt.figure(figsize=(12, 8))

for i, (method_name, method_key) in enumerate(methods):
    scores = [t[method_key] for t in filtered_tasks]
    
    # Plot bars for this method
    bar_positions = x + i * bar_width
    bars = plt.bar(
        bar_positions,
        scores,
        width=bar_width,
        color=colors[i],
        label=method_name,
        alpha=0.9
    )
    
    # Add values on top of each bar
    for pos, score in zip(bar_positions, scores):
        plt.text(
            pos,
            score, #+ (0.015 * max_score),  # small offset above the bar
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontsize=16
        )

sep_x = x[3] + 3*bar_width # Position after the 4th task group
plt.axvline(x=sep_x, linestyle="dashed", color="gray", linewidth=1.5)

# Configure x-axis
plt.xticks(
    x + bar_width,  # center tick label under the group
    [wrap_label(format_task_name(t["task"])) for t in filtered_tasks],
    # [format_task_name(t["task"]) for t in filtered_tasks],
    # rotation=45,
    ha="center",
    fontsize=18
)

plt.ylim(0, ylim_top)
plt.ylabel("Accuracy (%)", fontsize=26)
plt.tick_params(axis='y', labelsize=16)
plt.tick_params(axis='x', direction='in', length=0, labelbottom=True, pad=4)
# Remove the top and right spines for a cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add legend
plt.legend(fontsize=16, frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "BBH_tasks_main.pdf"))
plt.close()

# ------------------------------------------------------------------------------
# 2) Plot for Average Results (Across ALL Tasks)
# ------------------------------------------------------------------------------
avg_results = [aggregated[m[1]]["average"] for m in methods]
method_names = [m[0] for m in methods]

plt.figure(figsize=(10, 6))
bars = plt.bar(method_names, avg_results, color=colors)

# Add values on top of the bars
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        bar.get_height() + 0.5, 
        f"{bar.get_height():.1f}", 
        ha="center", 
        fontsize=22
    )

plt.xticks(rotation=45, ha="right", fontsize=20)
plt.yticks(fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=26)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "BBH_overall_main.pdf"))
plt.close()


# ------------------------------------------------------------------------------
# 3) Plot for All Tasks (27 Tasks in 3 Rows of 9)
# ------------------------------------------------------------------------------
# Determine the layout of the grid (3 rows, 9 columns)
rows = 3
cols = 9
task_count = len(tasks)
tasks_per_row = cols

# Ensure all tasks are included in the plot
assert task_count <= rows * cols, "Too many tasks to fit into the grid layout."

# Create a figure for the grid plot
fig, axes = plt.subplots(rows, cols, figsize=(24, 12), sharey=True)
axes = axes.flatten()

# Sort tasks alphabetically for consistent order
tasks_sorted = sorted(tasks, key=lambda t: t["task"])

# Loop through each task and plot its data in the respective subplot
for i, task in enumerate(tasks_sorted):
    ax = axes[i]
    
    # Extract data for the task
    task_name = format_task_name(task["task"])
    scores = [task[m[1]] for m in methods]
    
    # Create a bar plot for the task
    bar_positions = np.arange(len(methods))
    bars = ax.bar(
        bar_positions,
        scores,
        width=0.6,
        color=colors,
        alpha=0.9
    )
    
    # Add the task name as the subplot title
    ax.set_title(wrap_label(task_name, width=20), fontsize=14, pad=10)
    
    # Set x-axis labels for the methods
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([m[0] for m in methods], fontsize=10, rotation=45)
    
    # Add accuracy values on top of each bar
    for pos, score in zip(bar_positions, scores):
        ax.text(
            pos,
            score + 0.5,  # small offset above the bar
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontsize=10
        )
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust y-axis limits
    ax.set_ylim(0, ylim_top)

# Hide any empty subplots if task count < rows * cols
for i in range(task_count, rows * cols):
    axes[i].axis('off')

# Add a global y-axis label and adjust layout
fig.text(0.04, 0.5, "Accuracy (%)", va='center', rotation='vertical', fontsize=18)
fig.tight_layout(rect=[0.04, 0.03, 1, 0.95])

# Save the grid plot
output_file = os.path.join(output_dir, "BBH_all_main.pdf")
plt.savefig(output_file)
plt.close()


print(f"All plots saved in {output_dir}")
