# RobustnessMultimodal

Official repository for evaluating the robustness of the Themis model and its unimodal components under adversarial perturbations applied to text, images, or both modalities.

The project is organized as a modular pipeline. Clean inference, adversarial attacks, late-fusion construction, metric computation, and ROC-curve generation are executed through separate scripts.

---

## Quick start

The main entry point is `run_pipeline.py`. Select a dataset and a model type; checkpoint paths are resolved automatically from `models/weights/<dataset>/`.

```bash
python run_pipeline.py --dataset Recovery --model feature-fusion
```

Supported model selections are:

- `text`;
- `image`;
- `feature-fusion`;
- `late-fusion` with `--fusion mean|min|max`.

Examples:

```bash
python run_pipeline.py --dataset Recovery --model text
python run_pipeline.py --dataset Recovery --model image
python run_pipeline.py --dataset Recovery --model late-fusion --fusion max
```

Use `--list` to display compatible experiments, `--only` to select experiments, and `--dry-run` to preview the workflow without running models:

```bash
python run_pipeline.py --dataset Recovery --model feature-fusion --list
python run_pipeline.py --dataset Recovery --model feature-fusion --only clean pgd
python run_pipeline.py --dataset Recovery --model late-fusion --fusion mean --dry-run
```

Technical subprocess commands are hidden by default. Add `--show-commands` when debugging. Logs are stored under `logs/pipeline/<dataset>/<model>/`, and results under `results/<dataset>/classification_results/`.

## Requirements

Reference environment:

- Python 3.10
- CUDA 11.8
- PyTorch 2.1.2
- Torchvision 0.16.2
- NumPy 1.26.4

The remaining dependencies are listed in `requirements.txt`.

---

## 1. Clone the repository

```bash
git clone https://github.com/Davi2082/RobustnessMultimodal.git
cd RobustnessMultimodal
```

---

## 2. Create a Python 3.10 environment

### Linux and macOS

```bash
pyenv install 3.10
pyenv local 3.10

python -m venv venv
source venv/bin/activate
```

### Windows PowerShell

```powershell
pyenv install 3.10
pyenv local 3.10

python -m venv venv
venv\Scripts\activate
```

Verify the Python version:

```bash
python --version
```

The output should be Python 3.10.x.

---

## 3. Install the dependencies

Install the packages declared by the repository:

```bash
pip install -r requirements.txt
```

Install the version for your CUDA builds of PyTorch and Torchvision:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

Install the NumPy version used by the reference environment:

```bash
pip install numpy==1.26.4
```

---

## 4. Configure the project

Shared model, device, and attack defaults are defined in `configuration_files/configuration.py`. Derived filesystem paths are defined in `configuration_files/paths.py`. Command-line values supplied to `run_pipeline.py` take precedence for dataset and model selection.

For quick checks, set `SUBSET_SIZE` to a small integer. Use `None` for the complete test set. CUDA devices are controlled by `DEVICE`, `DEVICE_EVAL`, and `DEVICE_MLM`.

## 5. Prepare or add a dataset

Datasets live below `data_loading/`:

```text
data_loading/
├── my_datasets.py
├── Recovery/
│   ├── images/
│   ├── train_augmented.csv
│   ├── val_augmented.csv
│   └── test.csv
└── NewDataset/
    ├── images/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Training accepts `train_augmented.csv`, `train_augmented.tsv`, `train.csv`, or `train.tsv`, and the equivalent `val...` names. Evaluation requires exactly one `test.*` annotation file.

To integrate a new dataset, add these two objects to `data_loading/my_datasets.py`:

```python
class NewDataset_Dataset(torch.utils.data.Dataset):
    ...

def newdataset_load_annotations_file(file_path):
    ...
```

The names must follow `<DatasetName>_Dataset` and `<datasetname>_load_annotations_file`. Each dataset item must return:

```python
image, label, tokenized_text, image_path, sample_index
```

After adding the directory and implementation, the dataset is discovered automatically by both `scripts/train.py` and the evaluation utilities. No hard-coded training registry needs to be edited.

## 6. Train models

Use the same trainer for all base model types:

```bash
python -m scripts.train --dataset NewDataset --model text
python -m scripts.train --dataset NewDataset --model image
python -m scripts.train --dataset NewDataset --model feature-fusion
```

Common options include:

| Argument | Description |
| --- | --- |
| `--dataset` | Dataset directory below `data_loading/` |
| `--model` | `text`, `image`, or `feature-fusion` |
| `--epochs` | Number of training epochs |
| `--batch-size` | Training and validation batch size |
| `--learning-rate` | AdamW learning rate |
| `--device` | PyTorch device, for example `cuda:0` or `cpu` |
| `--name-llm` | Hugging Face language-model identifier |
| `--name-img-embed` | Image encoder identifier |
| `--use-lora` / `--no-use-lora` | Enable or disable LoRA |
| `--n-tokens` | Maximum tokenized text length |

Run `python -m scripts.train --help` for the complete list. The best validation-F1 checkpoint is saved using a stable pipeline-compatible name:

```text
models/weights/<dataset>/best_text_only.pt
models/weights/<dataset>/best_img_only.pt
models/weights/<dataset>/best_feature_fusion.pt

```

Each checkpoint is accompanied by a `.json` metadata file containing the encoder and LoRA architecture settings. `run_pipeline.py` reads this sidecar automatically, so non-default training options are preserved during evaluation.

Training curves are saved under `results/<dataset>/training/<model>/training_history.png`.

To train every model needed for all fusion experiments on a new dataset:

```bash
python -m scripts.train --dataset NewDataset --model text
python -m scripts.train --dataset NewDataset --model image
python -m scripts.train --dataset NewDataset --model feature-fusion
```

Then run the pipeline without specifying checkpoint paths:

```bash
python run_pipeline.py --dataset NewDataset --model feature-fusion
python run_pipeline.py --dataset NewDataset --model late-fusion --fusion max
```

Late fusion is not trained as a separate neural checkpoint. It combines the trained text-only and image-only models.

The pipeline prefers the stable checkpoint names generated by `scripts.train`. For backward compatibility, it can also discover one unambiguous legacy checkpoint matching `*_txt_only.pt`, `*_img_only.pt`, or the feature-fusion pattern. If several legacy candidates exist and no stable checkpoint exists, it stops and reports the ambiguity.

## 7. Result-directory structure

The metric and plotting scripts expect the following structure:

```text
results/
└── Recovery/
    └── classification_results/
        ├── clean/
        │   ├── feature-fusion/
        │   │   ├── results.csv
        │   │   └── parameters.json
        │   ├── text/
        │   │   ├── results.csv
        │   │   └── parameters.json
        │   ├── image/
        │   │   ├── results.csv
        │   │   └── parameters.json
        │   └── late-fusion/
        │       ├── mean/
        │       ├── min/
        │       └── max/
        └── perturbed/
            ├── feature-fusion/
            │   ├── perturbed_results.csv
            │   ├── parameters.json
            │   ├── text-perturbed/
            │   │   └── perturbed_results.csv
            │   └── image-perturbed/
            │       └── perturbed_results.csv
            ├── text/
            │   ├── perturbed_results.csv
            │   └── parameters.json
            ├── image/
            │   ├── perturbed_results.csv
            │   └── parameters.json
            └── late-fusion/
                ├── mean/
                ├── min/
                └── max/
```

### Path consistency

All path constants are centralised in `configuration_files/paths.py` (`RESULT_PATH`, `CLEAN_BASE`, `PERT_BASE`, and specific CSV / parameter paths). The `--results_path` CLI argument defaults to `RESULT_PATH`.

The commands below assume:

```text
results/Recovery/classification_results
```

as the result root (the value of `RESULT_PATH` in `configuration_files/paths.py`).

---

## 8. Run clean inference

Clean inference is performed with `eval.py`.

The main required argument is `--modality`.

### 8.1 Feature-fusion model

```bash
python -m scripts.eval --modality feature-fusion
```

The model receives both clean text and clean images.

Configuration name:

```text
feature-fusion|clean
```

### 8.2 Text model

```bash
python -m scripts.eval --modality text
```

The model receives only clean text.

Configuration name:

```text
text|clean
```

### 8.3 Image model

```bash
python -m scripts.eval --modality image
```

The model receives only clean images.

Configuration name:

```text
image|clean
```

### 8.4 Clean late fusion

`eval.py` accepts `late-fusion` and one aggregation mode:

```bash
python -m scripts.eval --modality late-fusion --late_fusion_mode mean
```

Available modes:

```text
mean
min
max
```

---

## 9. `eval.py` arguments

| Argument | Type | Description |
| --- | --- | --- |
| `--modality` | `str` | `feature-fusion`, `intermediate-fusion`, `late-fusion`, `text`, or `image` |
| `--late_fusion_mode` | `str` | Late-fusion aggregation: `mean`, `min`, or `max` |
| `--threshold` | `float` | Classification threshold |
| `--name_llm` | `str` | Text-encoder model name |
| `--name_img_embed` | `str` | Image-encoder model name |
| `--batch_size` | `int` | Evaluation batch size |
| `--model_path` | `str` | Path to model weights |
| `--n_tokens` | `int` | Maximum text length in tokens |
| `--merge_tokens` | `int` | Token-merging parameter |
| `--lora_alpha` | `int` | LoRA alpha |
| `--lora_r` | `int` | LoRA rank |
| `--lora_dropout` | `float` | LoRA dropout |
| `--use_lora` | `bool` | Whether LoRA is enabled |
| `--set_params` | `bool` | Whether model parameters are configured automatically |
| `--results_path` | `str` | Root directory for results |
| `--dataset` | `str` | Dataset discovered from `data_loading/my_datasets.py` |

Clean evaluation creates:

```text
results.csv
parameters.json
```

Late-fusion evaluation also creates a text-vs-image diagnostic plot.

---

## 10. Run adversarial attacks

The attack scripts load model settings from the `parameters.json` files generated by clean inference.

Clean evaluation for the corresponding model must therefore be completed first.

The attack direction is controlled by:

```text
--source_label
--target_label
```

Perturbations are generated for samples whose ground-truth label matches `source_label`. Samples from the other class remain clean (pass through unchanged).

---

### 10.1 Text-model attack

Prerequisite:

```text
clean/text/parameters.json
```

Run:

```bash
python -m attacks.unimodal.text_attack
```

The script:

1. loads the clean text-model parameters;
2. uses [TrePAT](https://github.com/piotrmp/trepat) to generate the text perturbations;
3. generates adversarial text;
4. evaluates the text model on the perturbed input;
5. saves predictions and attack parameters.

Expected output:

```text
perturbed/text/perturbed_results.csv
perturbed/text/parameters.json
```

Configuration name:

```text
text|perturbed
```

Relevant arguments:

```text
--k
--threshold_pred_score
--max_words_to_attack
--max_candidates_per_word
--max_words_for_importance
--source_label
--target_label
```

---

### 10.2 Image-model attack

Prerequisite:

```text
clean/image/parameters.json
```

Run:

```bash
python -m attacks.unimodal.image_attack
```

The script applies PGD to the image model and evaluates it on perturbed images.

Expected output:

```text
perturbed/image/perturbed_results.csv
perturbed/image/parameters.json
```

Configuration name:

```text
image|perturbed
```

Relevant arguments:

```text
--pgd_iters
--epsilon
--alpha_factor
--source_label
--target_label
```

---

### 10.3 Feature-fusion attack

Prerequisite:

```text
clean/feature-fusion/parameters.json
```

Run:

```bash
python -m attacks.multimodal.multimodal_attack
```

For each attacked sample, the script generates both a text perturbation and an image perturbation.

The same feature-fusion model is then evaluated in three input conditions:

1. perturbed text and perturbed image;
2. perturbed text and clean image;
3. clean text and perturbed image.

Generated files:

```text
feature-fusion/perturbed_results.csv
feature-fusion/parameters.json

feature-fusion/text-perturbed/perturbed_results.csv
feature-fusion/text-perturbed/parameters.json

feature-fusion/image-perturbed/perturbed_results.csv
feature-fusion/image-perturbed/parameters.json
```

Relevant arguments:

```text
--pgd_iters
--epsilon
--alpha_factor
--k
--threshold_pred_score
--max_words_to_attack
--max_candidates_per_word
--max_words_for_importance
--source_label
--target_label
```

---

## 11. Build perturbed late-fusion results

Late fusion is constructed from the perturbed scores produced independently by the text and image models.

Prerequisites:

```text
perturbed/text/perturbed_results.csv
perturbed/image/perturbed_results.csv
perturbed/text/parameters.json
perturbed/image/parameters.json
```

Run:

```bash
python -m scripts.late_fusion_perturbation
```

The script creates:

```text
perturbed/late-fusion/mean/
perturbed/late-fusion/min/
perturbed/late-fusion/max/
```

Configuration names:

```text
late-fusion-mean|biperturbed
late-fusion-min|biperturbed
late-fusion-max|biperturbed
```

`biperturbed` means that:

- the text score is produced from perturbed text;
- the image score is produced from a perturbed image;
- the two scores are subsequently aggregated.

Input and output paths are read from `configuration_files/paths.py` (`PERT_BASE`, `PER_TEXT_CSV`, etc.). To change the result root, update `RESULT_PATH` in `configuration_files/paths.py`.

---

## 12. Compute metrics

Use `metrics.py` to compute metrics, save `metrics.json`, create a confusion matrix, and optionally update a ROC comparison group.

### Feature fusion

Clean:

```bash
python -m scripts.metrics --type clean --modality feature-fusion
```

Both modalities perturbed:

```bash
python -m scripts.metrics --type perturbed --modality feature-fusion --perturbation_type biperturbed
```

Only text perturbed:

```bash
python -m scripts.metrics --type perturbed --modality feature-fusion --perturbation_type text-perturbed
```

Only image perturbed:

```bash
python -m scripts.metrics --type perturbed --modality feature-fusion --perturbation_type image-perturbed
```

### Text model

```bash
python -m scripts.metrics --type clean --modality text
python -m scripts.metrics --type perturbed --modality text
```

### Image model

```bash
python -m scripts.metrics --type clean --modality image
python -m scripts.metrics --type perturbed --modality image
```

### Late fusion

Clean:

```bash
python -m scripts.metrics --type clean --modality late-fusion --mode mean
python -m scripts.metrics --type clean --modality late-fusion --mode min
python -m scripts.metrics --type clean --modality late-fusion --mode max
```

Perturbed:

```bash
python -m scripts.metrics --type perturbed --modality late-fusion --mode mean --perturbation_type biperturbed
python -m scripts.metrics --type perturbed --modality late-fusion --mode min  --perturbation_type biperturbed
python -m scripts.metrics --type perturbed --modality late-fusion --mode max  --perturbation_type biperturbed
```

Each configuration produces:

```text
metrics.json
confusion_matrix.png
```

The implementation inverts labels, predictions, and scores before computing metrics:

```python
y_true = 1 - y_true
y_pred = 1 - y_pred
scores = 1 - scores
```

This makes the fake-news class (label 0) the positive class for precision, recall, F1, and ROC/AUC.

The result root is `RESULT_PATH` from `configuration_files/paths.py` (`results/Recovery/classification_results`).

---

## 13. Generate ROC plots

Run:

```bash
python -m scripts.create_rocs_plots
```

The script repeatedly invokes `metrics.py` and builds four comparison groups:

1. clean multimodal configurations;
2. perturbed multimodal configurations;
3. clean unimodal configurations;
4. perturbed unimodal configurations.

ROC labels use the `model|input` convention, for example:

```text
feature-fusion|clean
feature-fusion|text-perturbed
feature-fusion|image-perturbed
feature-fusion|biperturbed
text|clean
text|perturbed
image|clean
image|perturbed
late-fusion-mean|clean
late-fusion-mean|biperturbed
```

The ROC cache is updated one curve at a time, and the corresponding comparison plot is regenerated after every update.

---

## 14. Complete execution order

For a complete ReCOVery experiment, use the following order.

### Step 1: generate clean predictions

```bash
python -m scripts.eval --modality feature-fusion
python -m scripts.eval --modality text
python -m scripts.eval --modality image
```

Optionally create the clean late-fusion configurations:

```bash
python -m scripts.eval --modality late-fusion --late_fusion_mode mean
python -m scripts.eval --modality late-fusion --late_fusion_mode min
python -m scripts.eval --modality late-fusion --late_fusion_mode max
```

### Step 2: generate adversarial predictions

```bash
python -m attacks.multimodal.multimodal_attack
python -m attacks.unimodal.text_attack
python -m attacks.unimodal.image_attack
```

### Step 3: organize feature-fusion outputs

Place:

```text
txts_perturbed_results.csv
```

at:

```text
perturbed/feature-fusion/text-perturbed/perturbed_results.csv
```

Place:

```text
imgs_perturbed_results.csv
```

at:

```text
perturbed/feature-fusion/image-perturbed/perturbed_results.csv
```

### Step 4: construct perturbed late fusion

```bash
python -m scripts.late_fusion_perturbation
```

### Step 5: compute metrics and generate ROC curves

```bash
python -m scripts.create_rocs_plots
```

---

## 15. Minimal workflows

The complete pipeline does not need to be executed for every experiment.

### Clean feature fusion only

```bash
python -m scripts.eval --modality feature-fusion
python -m scripts.metrics --type clean --modality feature-fusion
```

### Text robustness only

```bash
python -m scripts.eval --modality text
python -m attacks.unimodal.text_attack
python -m scripts.metrics --type clean --modality text
python -m scripts.metrics --type perturbed --modality text
```

### Image robustness only

```bash
python -m scripts.eval --modality image
python -m attacks.unimodal.image_attack
python -m scripts.metrics --type clean --modality image
python -m scripts.metrics --type perturbed --modality image
```

### Compare feature-fusion input conditions

```bash
python -m scripts.eval --modality feature-fusion
python -m attacks.multimodal.multimodal_attack

python -m scripts.metrics --type perturbed --modality feature-fusion --perturbation_type biperturbed
python -m scripts.metrics --type perturbed --modality feature-fusion --perturbation_type text-perturbed
python -m scripts.metrics --type perturbed --modality feature-fusion --perturbation_type image-perturbed
```

### Perturbed late fusion only

```bash
python -m scripts.eval --modality text
python -m scripts.eval --modality image

python -m attacks.unimodal.text_attack
python -m attacks.unimodal.image_attack
python -m scripts.late_fusion_perturbation

python -m scripts.metrics --type perturbed --modality late-fusion --mode mean --perturbation_type biperturbed
python -m scripts.metrics --type perturbed --modality late-fusion --mode min  --perturbation_type biperturbed
python -m scripts.metrics --type perturbed --modality late-fusion --mode max  --perturbation_type biperturbed
```

---

## 16. Output files

Depending on the executed scripts, a result directory may contain:

| File | Description |
| --- | --- |
| `results.csv` | Clean labels, predictions, scores, logits, and sample indices |
| `perturbed_results.csv` | Predictions produced from perturbed inputs |
| `parameters.json` | Model and attack parameters |
| `metrics.json` | Classification and ROC/AUC metrics |
| `confusion_matrix.png` | Confusion matrix |
| `text_vs_image_clean.png` | Text-versus-image diagnostic plot for clean late fusion |
| ROC cache files | Stored FPR, TPR, and AUC values |
| ROC plots | Comparison curves for the selected ROC group |

---

## 17. Troubleshooting

### A clean `parameters.json` file cannot be found

The attack scripts depend on files generated by `eval.py`.

Run clean inference for the corresponding model first and verify that the path used by the attack script matches the actual output path.

### CUDA device error

Change the explicit `cuda:1` or `cuda:2` assignments to devices available on the current machine.

### A feature-fusion single-modality result cannot be found

Copy:

```text
txts_perturbed_results.csv
```

to:

```text
perturbed/feature-fusion/text-perturbed/perturbed_results.csv
```

Copy:

```text
imgs_perturbed_results.csv
```

to:

```text
perturbed/feature-fusion/image-perturbed/perturbed_results.csv
```

### Results are written to an unexpected location

Check:

- `RESULT_PATH` in `configuration_files/paths.py` (the canonical result root);
- `--results_path` CLI argument (defaults to `RESULT_PATH`).

### The test annotation file is not found

Ensure that this pattern matches an existing file:

```text
data_loading/<DatasetName>/test.*
```

### TrePAT setup fails

Verify that TrePAT and its dependencies are installed following the [official TrePAT repository](https://github.com/piotrmp/trepat). For offline execution, download and cache every required model beforehand.

---

## Experiments

Experiment outputs may be stored in the `experiments/` directory.

---

## Dataset and model

**Dataset:** ReCOVery, adapted following the `Is-It-Fake-Or-Not` project:

https://github.com/demon-prin/Is-It-Fake-Or-Not

**Models:** Themis, its text-only variant, and its image-only variant.

Pretrained weights are not distributed automatically by this repository and must be trained, requested, or provided separately.

---

## Repository

https://github.com/Davi2082/RobustnessMultimodal

---

## License

Distributed under the MIT License.
