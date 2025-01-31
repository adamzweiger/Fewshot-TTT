import os
import matplotlib.pyplot as plt

output_dir = "../plots"

# Data
labels = ['FT', 'FT + TTT', 'Base', 'TTT']
accuracy = [17.5, 45.0, 50.5, 57.8]  # in percent
heights = [val / 100.0 for val in accuracy]

# Set up the figure (increasing the width from 5 to 7)
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(range(len(labels)), heights, color='white', edgecolor='black')

# Add hatching to the second, fourth, and sixth bars
# bars[1].set_hatch('//')
# bars[3].set_hatch('//')
# bars[5].set_hatch('//')
# bars[5].set_edgecolor('gray')  # Add this line for lighter hatching


for i in [1, 3]:  # Indices of the bars to add hatches to
    ax.bar(
        i,
        heights[i],
        color='none',
        edgecolor='black',    # Set light gray hatches
        hatch='//',
        linewidth=0,
        alpha=0.5                 # Adjust transparency for a lighter effect
    )



# Add numeric labels above each bar
for i, val in enumerate(accuracy):
    ax.text(i, heights[i], f'{val}%', ha='center', va='bottom', fontsize=14)

# Dashed vertical line after the 4th bar
ax.axvline(x=1.5, color='black', linestyle='--')

# ---------------------------------------------------------
# Add a horizontal line for the zero-shot BBH baseline
# to the right of x=3.5 (i.e., in the BBH region only).
zero_shot_bbh = 40.9
ax.hlines(
    y=zero_shot_bbh / 100.0,    # Convert % to fraction
    xmin=1.5,                   # Start just to the right of the vertical dashed line
    xmax=len(labels) - 0.5,     # Extend to near the last bar (bar indices go from 0 to 5)
    color='black',
    linestyle='--',
    linewidth=1
)

# Optionally, label this partial horizontal line
# ax.text(
#     3.64,                       # x-position near the middle of the BBH region
#     (zero_shot_bbh / 100.0) + 0.005,
#     'Zero-Shot Baseline',
#     color='black',
#     fontsize=12
# )
# ---------------------------------------------------------

# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Label each region
ax.text(0.5, 0.65, 'ARC', ha='center', va='bottom', fontsize=16)
ax.text(2.5, 0.65, 'BBH', ha='center', va='bottom', fontsize=16)

# X-axis ticks and labels
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=12)

# Y-axis label and limit
ax.set_ylabel("Accuracy", fontsize=20)
ax.set_ylim([0, 0.7])
plt.tick_params(axis='y', labelsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(output_dir, "ARC+BBH.pdf"))
# plt.show()
