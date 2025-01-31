import matplotlib.pyplot as plt
import json
import os
import textwrap  

data_file = "../logs/archive/averages/allResults.json"
output_dir = "../plots"
os.makedirs(output_dir, exist_ok=True)

# Load the data file
with open(data_file, "r") as f:
    data = json.load(f)

tasks = data["per_task_results"]
aggregated = data["aggregated_statistics"]

# Updated method list & colors
method_configs = [
    (0, "Zero-Shot", "ZSL_accuracy", "#EFECCA"),
    (1, "ICL", "FSL_10_accuracy", "#598392"),
    (2, "TTT", "ICFT_main_10_masked_inputs_text_completion_dataset_40_True_False_5_False_5_1_1e-4_64_64_0.05_accuracy", "#01161e"),
    (3, "No Example Permutations", "TTT_noshuffle_10_masked_inputs_text_completion_dataset_1_False_False_5_False_1_4_1e-4_64_64_accuracy", "#004C6D"),
    (4, "Direct I/O", "FT_10_5_4_1e-4_64_64_0.05_accuracy", "#80a4c4"),
    (5, "Shared TTT", "SHARED_TTT_exp_10_masked_inputs_text_completion_dataset_40_True_False_5_False_5_1_5e-5_64_64_0.05_accuracy", "#bc5191"),
    (6, "No Demo Loss", "ICFT_ex_trainLast_10_masked_text_completion_dataset_40_True_False_5_False_5_1_1e-4_64_64_accuracy", "#ff6361"),
    (7, "Loss on Inputs and Outputs", "ICFT_ex_trainAll_10_text_completion_dataset_40_True_False_5_False_5_1_1e-4_64_64_0.05_accuracy", "#FFB20F"),
]

vertical_lines = [2.5, 4.5]

def wrap_label(label, width=12):
    return "\n".join(textwrap.fill(label, width=width).split('\n'))

def plot_bar_group(ax, task_or_agg, is_task=True, small_text=False):
    values = [(x, task_or_agg[key]["average"] if not is_task else task_or_agg[key], color) for (x, _, key, color) in method_configs]
    max_score = max(y for (_, y, _) in values)
    ylim_top = max(100, max_score + (0.07 * max_score)) if is_task else 70
    ax.set_ylim(0, ylim_top)
    if not is_task:
        ax.text(3.5, ylim_top + 2, "Data Ablations", ha="center", fontsize=16)
        ax.text(6, ylim_top + 2, "Optimization Ablations", ha="center", fontsize=16)
    
    for (x_val, y_val, color) in values:
        bar_container = ax.bar(x_val, y_val, color=color, width=0.6)
        rect = bar_container[0]
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + .7,
            f"{y_val:.1f}",
            ha="center",
            fontsize=10 if small_text else 17
        )
    
    for vline in vertical_lines:
        ax.axvline(x=vline, color="black", linestyle="--", linewidth=1)

    x_positions = [cfg[0] for cfg in method_configs]
    method_labels = [wrap_label(cfg[1], width=14) for cfg in method_configs]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=10 if small_text else 14)
    ax.tick_params(axis='x', direction='in', length=2, labelbottom=True, pad=4)
    ax.tick_params(axis='y', labelsize=14)
plt.rcParams['font.family'] = 'serif'
# --- Average Results Plot ---
plt.figure(figsize=(10, 6))
ax = plt.gca()
plot_bar_group(ax, aggregated, is_task=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylabel("Accuracy (%)", fontsize=22)
plt.tight_layout()
out_path = os.path.join(output_dir, "BBH_overall_ablations.pdf")
plt.savefig(out_path)
plt.close()
print(f"Saved average results plot -> {out_path}")

# --- All Tasks Plot ---
fig, axes = plt.subplots(3, 9, figsize=(36, 12))
fig.subplots_adjust(hspace=0.6, wspace=0.6)
axes = axes.flatten()

for idx, task in enumerate(tasks):
    ax = axes[idx]
    plot_bar_group(ax, task, is_task=True, small_text=True)
    ax.set_title(task["task"].replace("_", " ").title(), fontsize=12)

for i in range(len(tasks), len(axes)):
    axes[i].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = os.path.join(output_dir, "BBH_all_ablations.pdf")
plt.savefig(out_path)
plt.close()
print(f"Saved all tasks plot -> {out_path}")
print("All ablation plots updated and saved successfully!")
