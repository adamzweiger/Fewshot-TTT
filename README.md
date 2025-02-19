# Fewshot-TTT

This repository is an implementation of test-time training for few-shot learning using **vLLM** for inference and **Torchtune** for **LoRA** finetuning.

## Repository Structure

```
TTT/
├── external/
│   ├── BIG-Bench-Hard/          # Submodule for BIG-Bench Hard tasks
│   └── torchtune/               # Submodule for adamzweiger's Torchtune
│
├── logs/                        # Logs
│   ├── archive/                 # Archived logs
│   └── current/                 # Current logs
│
├── scripts/                     # SLURM scripts for running experiments or utilities
│
├── src/                         # Source code for the project
│   ├── tasks.py                 # Defines BIG-Bench Hard tasks
│   ├── utils.py                 # Common utilities (e.g., inference_vllm, compute_accuracy)
│   └── methods/
│       ├── baseline.py          # Zero-/few-shot baseline
│       ├── e2e.py               # Direct I/O finetuning without ICL
│       └── ttt_sweep.py         # Main TTT method (in-context fine-tuning)
│
├── README.md                    # Project documentation
└── requirements.txt             # List of dependencies
```

---

## Installation

### **1. Clone the Repository with Submodules**

To ensure that the external submodules are included, clone the repository using the `--recurse-submodules` flag:

```bash
git clone --recurse-submodules https://github.com/adamzweiger/Fewshot-TTT.git
cd TTT
```

**Alternatively**, if you've already cloned the repository without submodules, initialize and update them manually:

```bash
git submodule update --init --recursive
```

### **2. Install Dependencies**

#### **A. Create and Activate a Virtual Environment**

Using `conda`:

```bash
conda create -n tttenv python=3.12
conda activate tttenv
```

Using `venv`:

```bash
python3.12 -m venv tttenv
source tttenv/bin/activate
```

#### **B. Install Project Dependencies**

These dependencies cover everything needed in the submodules as well.

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Direct evaluation
Run the respective evaluation script to test the method on tasks in BIG-Bench Hard.

```bash
python src/methods/baseline.py --model_dir Llama-3.1-8B-Instruct --output_file results.json
```

### 2. Example Bash Script
Use the bash scripts in `scripts/` for a complete end-to-end evaluation workflow:

```bash
sbatch scripts/baseline.sh
```
