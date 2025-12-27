# ASEL — Adaptive Selector with Efficient Learning for Vision Transformers

This repository contains the reference code used in the ASEL paper: an adaptive patch selector built on top of a Vision Transformer (ViT) that learns to keep only the most informative image patches at inference time for faster and cheaper inference while retaining accuracy. The notebooks and saved artifacts included provide training, transfer learning and benchmarking for CIFAR‑10 and several remote sensing datasets (AID, EuroSAT, RSSCN7).

## Repository structure

- `AgentViT.ipynb` — related notebook (experimental / auxiliary agent model notebook).
- `ASEL.ipynb` — main pipeline notebook: training warmup on CIFAR‑10, transfer to remote sensing datasets, and the benchmarking suite.
- `saved_models/` — pretrained/fine-tuned model checkpoints (examples included):
  - `agentvit_aid.pth`, `agentvit_eurosat.pth`, `agentvit_rsscn7.pth`
  - `aid_finetuned.pth`, `eurosat_finetuned.pth`, `rsscn7_finetuned.pth`
  - `cifar10_warmup.pth` (CIFAR‑10 warmup checkpoint)
- `benchmarks_results/` and `benchmarks_results_agentvit/` — output plots produced by the benchmarking routines.

> Note: The notebook files contain self-contained code (models, training loops, deterministic dataset handling, and benchmarking). You can run them interactively or convert to scripts if preferred.

## Highlights / Features

- ASEL model: ViT backbone (`vit_tiny_patch16_224` from `timm`) plus a learned patch selector (small MLP) that scores each patch w.r.t. a global patch feature.
- Two-phase pipeline implemented in the notebook:
  1. CIFAR‑10 warmup (train selector + backbone with special learning rate schedule and sparsity loss).
  2. Transfer learning to remote sensing datasets (AID, EuroSAT, RSSCN7) using CIFAR‑10 weights (excluding head).
- Deterministic splits and DataLoader seeding for reproducible experiments.
- Benchmarking suite that measures accuracy (for multiple policies), GFLOPs (via `fvcore` if available or a heuristic), latency and throughput.

## Requirements

This project was developed with Python and PyTorch. Minimum recommended dependencies:

- Python 3.8+ (tested with 3.8–3.11)
- torch
- torchvision
- timm
- numpy
- matplotlib
- tqdm
- fvcore (optional, used for accurate FLOPs counting)

Install with pip (example):

```fish
# create and activate a virtual environment (fish syntax)
python -m venv .venv
source .venv/bin/activate.fish

# Install core requirements
pip install torch torchvision timm numpy matplotlib tqdm

# Optional: accurate FLOPs counting
pip install fvcore
```

Adjust the `torch` install line according to your CUDA version (see https://pytorch.org/).

## How to run

There are two primary ways to run the reference code:

1) Open the main notebook `ASEL.ipynb` in JupyterLab / Jupyter Notebook and run the cells top-to-bottom. The notebook contains the full pipeline with caching logic (saves checkpoints to `./saved_models`) and benchmarking (saves plots to `./benchmarks_results`).

2) Convert the notebook to a script (if you prefer non-interactive runs) and execute it in a Python environment. Example (optional):

```fish
# Convert to script (optional)
jupyter nbconvert --to script ASEL.ipynb --stdout > run_asel.py
python run_asel.py
```

Important runtime notes:
- The notebook's `CONFIG` dictionary (inside `ASEL.ipynb`) controls device selection (`cuda` if available), batch size, number of workers, random seed, and save paths. Adjust these values before running large experiments.
- The code uses deterministic seeds and worker init functions so that dataset splits and shuffling are reproducible.

## Datasets and paths

- CIFAR‑10: downloaded automatically by torchvision when running the notebook (stored under `./data`).
- EuroSAT: also available via torchvision and downloaded to `./data` automatically.
- AID, UCMerced and RSSCN7: the notebook expects these as local folders (see `DATASET_PATHS` in `ASEL.ipynb`). Edit `DATASET_PATHS` or place the datasets in the repository root with the expected folder names:

  - `AID-data/`
  - `UCMerced_LandUse/Images/` (named `ucmerced` in the code)
  - `./RSSCN7/`

If a path is missing the notebook will print a message and skip that dataset (so you can run the rest of the pipeline).

## Saved models and caching

- CIFAR‑10 warmup checkpoint: `./saved_models/cifar10_warmup.pth`. If present, the notebook will skip CIFAR‑10 training and proceed to benchmarking and transfer learning.
- Transfer/fine-tuned checkpoints for target datasets are saved under `./saved_models/{ds_name}_finetuned.pth`. If present, the notebook will load them and skip training for that dataset.

## Benchmarks and output

- All benchmarking plots are saved in `./benchmarks_results` by default. Filenames follow the pattern `{dataset}_{index}_{metric}.png` (see notebook for exact names).
- The Benchmarking routine evaluates three selection policies at multiple keep ratios:
  - `learned` (top-k patches by selector scores)
  - `random` (random subset)
  - `central` (most central patches in patch grid)

Metrics collected: accuracy, GFLOPs (fvcore if available), latency (batch timing using CUDA events), throughput.

## Reproducibility

- The notebook uses `set_seed(...)` and a `torch.Generator` instance `GEN` for deterministic splits and reproducible DataLoader shuffles.
- CuDNN deterministic mode is enabled by default in the notebook; this can slow down runs but increases reproducibility.

## Quick tips

- If you only want to run the benchmarking (no training), provide the relevant checkpoint(s) in `./saved_models` (e.g. `cifar10_warmup.pth` and `{ds_name}_finetuned.pth`). The notebook will detect them and skip training.
- If you encounter memory/OOM issues, reduce `CONFIG['batch_size']` or run on CPU by setting `CONFIG['device']='cpu'`.

## Citation

If you use this code in your research, please cite the ASEL paper. Example BibTeX (replace with actual paper details):

```bibtex
@inproceedings{author2025asel,
  title={ASEL: Adaptive Selector with Efficient Learning for Vision Transformers},
  author={Author, A. and Author, B.},
  year={2025},
  booktitle={Proceedings of ...}
}
```

## Contributing

Contributions, bug reports, and improvements are welcome. Open an issue or submit a pull request. Please include a short description and, if relevant, a small reproducible example.

## License

This repository does not include an explicit license file. Please check with the project owner for license details before using the code in production or redistributing it.

---

If you'd like, I can also:

- generate a `requirements.txt` pinned to the versions used when I tested the code,
- add a small runnable script that extracts the notebook `run()` function and runs it as a script,
- or create a minimal `README` badge and CI instructions for running a smoke test on CI.

Tell me which of those you'd like me to add next.
