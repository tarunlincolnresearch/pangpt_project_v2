# PanGPT Pipeline — Step-by-Step Guide

## What this project does

It runs the full PanGPT workflow end-to-end while keeping James McInerney's
panGPT code completely untouched. All improvements (windowing, preprocessing
logging, hyperparameter search, inference) live in this project only.

---

## STEP-BY-STEP DATA SETUP GUIDE

### Where to put your input data

Your input file is `gene-order.gz` (the compressed gene-order file).
You need to place it here:

```
pangpt_project/
└── data/
    └── gene-order.gz        ← PUT YOUR FILE HERE
```

**Exact commands to do this:**

```bash
# If you are on the HPC cluster:
cp /path/to/your/gene-order.gz  ~/pangpt_project/data/gene-order.gz

# If the file is already uncompressed (just called gene-order):
cp /path/to/your/gene-order     ~/pangpt_project/data/gene-order.gz
# Then edit config.py line:   DATA_RAW = DATA_DIR / "gene-order"   (remove .gz)
```

**If you downloaded it from Google Drive** (as in your notebook):
```bash
# Download to the correct location directly
pip install gdown
python -c "
import gdown
gdown.download('https://drive.google.com/uc?id=1OatYnj24lHKaZ4cT8SrFTK0cDAmVQPYO',
               'data/gene-order.gz', quiet=False)
"
```

---

## STEP-BY-STEP SETUP

### Step 1 — Clone panGPT (do NOT modify it)

```bash
git clone https://github.com/mol-evol/panGPT.git ~/panGPT
```

This puts panGPT at `~/panGPT/panGPT.py` — which is what `config.py` expects.

If you want to put it somewhere else, edit `config.py`:
```python
PANGPT_DIR = Path("/your/custom/path/to/panGPT")
```

### Step 2 — Install dependencies

```bash
pip install torch tokenizers scikit-learn numpy tqdm
```

On the HPC cluster you may need:
```bash
module load gcc nvidia/cuda
conda activate your_env
```

### Step 3 — Edit config.py (only 2 lines to change)

Open `config.py` and set:
```python
PROJECT_DIR = Path("/work/users/tgangil/pangpt_project")   # where this project lives
PANGPT_DIR  = Path("/home/users/tgangil/panGPT")           # where you cloned panGPT
```

Everything else (data paths, output paths) is computed automatically.

### Step 4 — Run the pipeline

```bash
cd ~/pangpt_project

# Full run (all steps):
bash run_pipeline.sh

# Skip hyperparameter search (faster, uses sensible defaults):
bash run_pipeline.sh --skip-search

# Restart from a specific step (e.g. if step 1 already finished):
bash run_pipeline.sh --from 2
```

### Step 5 — On GPU cluster (SLURM)

```bash
# Training job (step 4):
sbatch scripts/slurm/train.slurm

# Hyperparameter search (step 3):
sbatch scripts/slurm/hyperparam.slurm

# Check job status:
squeue --me
```

---

## FULL PROJECT STRUCTURE

```
pangpt_project/
│
├── config.py                       ← ONLY file you need to edit (2 lines)
├── run_pipeline.sh                 ← runs everything
├── README.md
│
├── data/                           ← all data lives here
│   ├── gene-order.gz               ← YOUR INPUT FILE goes here
│   ├── pangpt_training_sequences.txt       (after step 1)
│   ├── pangpt_train_windows.txt            (after step 2)  → fed to panGPT
│   ├── pangpt_val_windows.txt              (after step 2)
│   ├── pangpt_test_windows.txt             (after step 2)
│   ├── preprocessing_stats.json            (step 1 QC report)
│   └── split_and_window_stats.json         (step 2 split report)
│
├── logs/                           ← training logs
│   ├── 01_preprocess.log
│   ├── 02_split_and_window.log
│   ├── training.log                        (panGPT output)
│   └── ...
│
├── checkpoints/
│   └── model_checkpoint.pth        ← saved model (after training)
│
├── results/
│   ├── best_hyperparam_config.json (after step 3)
│   ├── training_summary.json       (after step 4)
│   ├── inference_results.json      (after step 5)
│   └── anomaly_results.json        (after step 6)
│
└── scripts/
    ├── 01_preprocess.py            ← parse gene-order, log * removals
    ├── 02_split_and_window.py      ← split genomes THEN window train only
    ├── 03_hyperparam_search.py     ← find best training config
    ├── 04_train.py                 ← call panGPT.py as-is
    ├── 05_inference.py             ← gene prediction evaluation
    ├── 06_anomaly.py               ← perplexity-based detection
    └── slurm/
        ├── train.slurm
        └── hyperparam.slurm
```

---

## THE EXACT FLOW (what each script does)

```
gene-order.gz
    │
    ▼  01_preprocess.py
    │  • Reads gene-order.gz (or plain gene-order)
    │  • Removes * entries — logs EXACTLY how many per genome
    │  • Writes: data/pangpt_training_sequences.txt
    │            (one genome per line, space-separated genes)
    │
    ▼  02_split_and_window.py    ← KEY SCRIPT — corrected flow
    │
    │  PART A: Genome-level split (mirrors James's panGPT.py exactly)
    │  ┌─────────────────────────────────────────────────────────────┐
    │  │  from sklearn.model_selection import train_test_split       │
    │  │  train_genomes, temp  = train_test_split(all, 80%, seed=42) │
    │  │  val_genomes, test    = train_test_split(temp, 50%, seed=42)│
    │  │  → 80% train  /  10% val  /  10% test                      │
    │  └─────────────────────────────────────────────────────────────┘
    │
    │  PART B: Moving window — ONLY on train genomes
    │  ┌─────────────────────────────────────────────────────────────┐
    │  │  Uses James's exact movingSplits.py logic:                  │
    │  │  [genome[i:i+W] for i in range(0, len-W+1, S)]             │
    │  │                                                             │
    │  │  Run 5 times with increasing overlap:                       │
    │  │   W=1024, S=1024  →  0%  overlap  (no overlap)             │
    │  │   W=1024, S=768   →  25% overlap                           │
    │  │   W=1024, S=512   →  50% overlap  (James's default)        │
    │  │   W=1024, S=256   →  75% overlap                           │
    │  │   W=1024, S=128   →  87% overlap  (maximum density)        │
    │  │                                                             │
    │  │  All windows combined + shuffled → train_windows.txt        │
    │  └─────────────────────────────────────────────────────────────┘
    │
    │  PART C: Val and test — single fixed window (no augmentation)
    │  → val_windows.txt   (W=1024, S=1024)
    │  → test_windows.txt  (W=1024, S=1024)
    │
    ▼  03_hyperparam_search.py  (optional)
    │  • Runs 12 short trials of panGPT
    │  • Saves best config → results/best_hyperparam_config.json
    │
    ▼  04_train.py
    │  • Reads best config (or uses config.py defaults)
    │  • Calls panGPT.py as-is via subprocess
    │  • Input: data/pangpt_train_windows.txt
    │
    ▼  05_inference.py
    │  • Loads trained model + tokenizer
    │  • Evaluates gene predictions with greedy decoding
    │  • Prompt length: 200 tokens (was 50)
    │
    ▼  06_anomaly.py
       • Builds perplexity baseline from normal genomes
       • Tests synthetic insertions/deletions/substitutions
       • Reports detection rates
```

---

## FREQUENTLY ASKED QUESTIONS

**Q: I have the gene-order file, not gene-order.gz. What do I do?**

Just put it in `data/gene-order` (without .gz), then edit one line in `config.py`:
```python
DATA_RAW = DATA_DIR / "gene-order"    # plain text, no .gz
```

**Q: Where exactly does my data go on the HPC cluster?**

Assuming your WORK directory is `/work/users/tgangil/`:
```
/work/users/tgangil/pangpt_project/data/gene-order.gz
```
And in `config.py`:
```python
PROJECT_DIR = Path("/work/users/tgangil/pangpt_project")
PANGPT_DIR  = Path("/home/users/tgangil/panGPT")
```

**Q: Do I need to change anything in panGPT?**

No. panGPT is used completely as-is. Nothing inside `~/panGPT/` is modified.

**Q: Why does 04_train.py pass train_windows.txt to panGPT?**

Because we already did the genome-level split in step 2. The train_windows.txt
contains ONLY training data. panGPT will do its own internal split on top of
this (its default is train=80%, val=10%), which is fine — it just means panGPT
re-splits within the training windows we provide.
