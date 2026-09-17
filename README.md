# RobustnessMultimodal

Adversarial robustness evaluation pipeline for the **Themis** multimodal fake-news classifier. The model fuses CLIP vision embeddings with TinyLlama text representations via cross-attention.

Evaluated model variants:
- **Text-only** and **Image-only** (unimodal baselines)
- **Feature fusion** (Themis joint model)
- **Late fusion** (min / mean / max score aggregation of unimodal outputs)

Attack types: PGD (image), TREPAT (text), PGD+TREPAT (multimodal).

---

## Requirements

- Python 3.10, CUDA 11.8, PyTorch 2.1.2, NumPy 1.26.4
- Remaining dependencies in `requirements.txt`

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate multimodal
```

---

## Configuration

All pipeline settings live in `config.yaml`:

| Setting | Description |
|---|---|
| `train_pipeline` | Enable training stage (0/1) |
| `clean_pipeline` | Enable clean evaluation stage (0/1) |
| `ablation_pipeline` | Enable missing-modality ablation stage (0/1) |
| `adversarial_pipeline` | Enable adversarial attack stage (0/1) |
| `datasets` | Active dataset(s) for the pipeline sweep |
| `dataset_configs` | Per-dataset GPU devices, subset sampling, and checkpoint filenames |
| `pipeline_late_fusion_modes` | Late-fusion modes to evaluate |
| `pipeline_attacks` | Attack types to run |

Subset sampling is configured per-dataset inside `dataset_configs` (`subset_size`, `balanced_subset`). Training hyperparameters, attack parameters, and model settings are also in `config.yaml`. Python code reads them via `configuration_files/configuration.py`.

---

## Running the full pipeline

```bash
bash run_scripts.sh
```

Toggle which stages run by setting `train_pipeline`, `clean_pipeline`, `ablation_pipeline`, and `adversarial_pipeline` to `0` or `1` in `config.yaml`. For each dataset listed in `datasets`:

0. **Training** — text, image, and feature-fusion models
1. **Clean evaluation** — text, image, feature-fusion, and all late-fusion modes
2. **Missing-modality ablation** — feature-fusion and late-fusion (metrics computed inline)
3. **Adversarial attacks** — all attack types against all fusion methods, followed by metrics

Results are written to `results/<dataset>/<subset_size>/<seed>/{clean,ablation,perturbed}/<model>/`.

---

## Printing result tables

```bash
python3 -m scripts.make_tables --dataset Recovery
python3 -m scripts.make_tables --dataset Fakeddit --table adversarial
```

Tables: `clean`, `adversarial`, `ablation`, or `all` (default). Metrics (AUC, F1, Acc, TPR, TNR, ASR) are computed directly from per-sample CSVs with Real as the positive class.

---

## Repository layout

```
RobustnessMultimodal/
├── config.yaml                          # All tunable settings
├── run_scripts.sh                       # Full pipeline entry point
├── configuration_files/
│   ├── configuration.py                 # Loads config.yaml into Python constants
│   └── paths.py                         # Result directory path helpers
├── data_loading/
│   └── my_datasets.py                   # Dataset classes (Recovery, Fakeddit)
├── models/
│   ├── themis_model.py                  # Themis nn.Module
│   └── fusion.py                        # Late-fusion heads
├── scripts/
│   ├── make_tables.py                   # Print result tables to stdout
│   ├── main_scripts/
│   │   ├── run_clean.py                 # Clean eval orchestrator
│   │   ├── run_ablation.py              # Ablation orchestrator
│   │   └── run_multimodal_attacks.py    # Attack orchestrator
│   ├── utils/
│   │   ├── eval.py                      # Single-model clean inference
│   │   ├── metrics.py                   # Metric computation
│   │   ├── modality_ablation.py         # Missing-modality probe
│   │   └── utils.py                     # Shared utilities
│   ├── train_scripts/
│   │   └── train.py                     # Model training
│   └── plot/                            # Plotting scripts
├── attacks/
│   ├── unimodal/
│   │   ├── image/attack.py              # PGD on image model
│   │   └── text/attack.py               # TREPAT on text model
│   ├── multimodal/
│   │   ├── sum/attack.py                # PGD+TREPAT sum attack
│   │   └── joint/attack.py              # Joint HotFlip+PGD attack
│   └── attack_algorithms/               # Core attack implementations
├── data/<Dataset>/                      # Dataset files (images + annotations)
├── checkpoints/<Dataset>/               # Model weights
└── results/<Dataset>/                   # Output CSVs and metrics
```

---

## Adding a new dataset

1. Place data in `data/<DatasetName>/` with `images/` and a `test.*` annotation file.
2. Add `<DatasetName>_Dataset` class and `<datasetname>_load_annotations_file` function to `data_loading/my_datasets.py`.
3. Add a `dataset_configs` entry in `config.yaml` with devices, subset settings, and checkpoint filenames.
4. Add the dataset name to `datasets` in `config.yaml`.

The dataset is discovered automatically — no registry edits needed.

---

## Training

Set `train_pipeline: 1` in `config.yaml`, or run manually:

```bash
python3 -m scripts.train_scripts.train --dataset Recovery --model text
python3 -m scripts.train_scripts.train --dataset Recovery --model image
python3 -m scripts.train_scripts.train --dataset Recovery --model feature-fusion
```

Training parameters (epochs, learning rate, LoRA settings) are configured in `config.yaml`. Late fusion does not require separate training — it combines unimodal model outputs.

---

## Result directory structure

```
results/<Dataset>/<subset_size>/<seed>/
├── clean/<model>/
│   ├── results.csv
│   └── parameters.json
├── ablation/<model>/
│   └── modality_ablation_metrics.csv
└── perturbed/<attack>/<model>/
    ├── perturbed_results.csv
    └── parameters.json
```

`<subset_size>` is `full` when no subset is used, or the integer count (e.g. `200`). `<seed>` is the random seed from config.yaml (e.g. `42`).

---

## License

Distributed under the MIT License.
