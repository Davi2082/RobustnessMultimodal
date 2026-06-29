# RobustnessMultimodal

Official repository for evaluating the robustness of the Themis model and its unimodal components under adversarial perturbations applied to text, images, or both modalities.

The project is organized as a modular pipeline. Clean inference, adversarial attacks, late-fusion construction, metric computation, and ROC-curve generation are executed through separate scripts.

---

## Quick start

To run the **entire pipeline end-to-end with the default configuration** — no
need to edit `configuration.py` or `paths.py` — use the orchestrator script:

```bash
python run_pipeline.py
```

It runs every stage in order and stops at the first failure:

1. clean evaluation — text, image, feature-fusion (`eval.py`);
2. adversarial attacks — image (PGD), text (BERTAttack), multimodal (`attacks/`);
3. late-fusion aggregation (`late_fusion_perturbation.py`);
4. metrics + ROC plots (`create_rocs_plots.py`).

Each stage logs to its own file under `logs/` (e.g. `logs/eval_text.log`,
`logs/multimodal_attack.log`). Results land under `RESULT_PATH` and ROC plots
under the `figures/` ROC directory (both defined in `paths.py`).

This assumes the environment, datasets, and model weights are already in place
(steps 1–6 below). For a fast smoke test of the whole pipeline, set
`SUBSET_SIZE` in `configuration.py` to a small number (see
[Quick test runs](#quick-test-runs-subset_size)) before launching it.

The individual scripts documented in the rest of this README remain available
for running or customizing a single stage.

---

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

Model and attack hyperparameters are defined in `configuration.py`.
Filesystem paths (result directories, CSV paths, weight file paths) are defined in `paths.py`.

Before running the pipeline, check at least:

- model and encoder names (`NAME_LLM`, `NAME_IMG_EMBED`);
- model-weight paths (`TEXT_WEIGHTS_PATH`, `IMAGE_WEIGHTS_PATH`, `FF_WEIGHTS_PATH`);
- batch size and maximum token length;
- classification threshold;
- result root directory (`RESULT_PATH` in `paths.py`);
- PGD parameters (`EPSILON`, `PGD_ITERS`, `ALPHA_FACTOR`);
- BERTAttack parameters (`K_BERT_ATTACK`, etc.);
- source and target labels;
- subset size for quick test runs (`SUBSET_SIZE`).

Most values can also be overridden through command-line arguments.

### Quick test runs (`SUBSET_SIZE`)

`SUBSET_SIZE` in `configuration.py` restricts both the clean evaluation
(`eval.py`) and all three attacks to the **first `N` samples** of the test set,
so you can sanity-check the full pipeline in seconds instead of running on the
whole dataset:

```python
# configuration.py
SUBSET_SIZE = None  # full dataset (default)
SUBSET_SIZE = 8     # only the first 8 samples — quick smoke test
```

It is a single switch that applies everywhere (clean eval + image / text /
multimodal attacks), so the clean and perturbed result sets stay aligned on the
same samples. Set it back to `None` for a full run.

### CUDA devices

The current scripts contain explicit CUDA assignments:

- `eval.py` uses `cuda`;
- `image_attack.py` uses `cuda:1`;
- `text_attack.py` uses `cuda:1` for the classifier and `cuda:2` for BERT;
- `multimodal_attack.py` uses `cuda:1` for the classifier and `cuda:2` for BERT.

Adapt these assignments to the available hardware.

For a single-GPU machine, replace them with `cuda:0`, provided that enough memory is available.

---

## 5. Prepare the datasets

Each dataset must be stored inside the `data/` directory.

A dataset folder must contain:

- an `images/` directory;
- one test annotation file matching `test.*`;
- any additional train or validation files required by the dataset implementation.

Example:

```text
data/
├── Recovery/
│   ├── images/
│   ├── test.csv
│   ├── train.csv
│   └── val.csv
├── Fakeddit/
│   ├── images/
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   ├── test.tsv
│   ├── train.tsv
│   └── val.tsv
└── YourNewDataset/
    ├── images/
    └── test.csv
```

The scripts locate the test annotations with:

```python
glob.glob(f"data/{dataset}/test.*")[0]
```

Each dataset directory should therefore contain only one intended test annotation file matching that pattern.

### Adding a new dataset

For every new dataset, define the corresponding dataset class and annotation loader in `my_datasets.py`.

Expected naming convention:

```text
{DatasetName}_Dataset
{datasetname}_load_annotations_file
```

Example:

```text
YourNewDataset_Dataset
yournewdataset_load_annotations_file
```

Available datasets are discovered dynamically through `load_available_datasets()`.

---

## 6. Prepare the model weights

Place pretrained weights in the `model/` directory or provide their paths through `--model_path`.

The pipeline supports three base predictive models:

| Model identifier | Input |
| --- | --- |
| `feature-fusion` | Text and image |
| `text` | Text only |
| `image` | Image only |

The following late-fusion configurations are derived from the scores generated independently by the text and image models:

- `late-fusion-mean`;
- `late-fusion-min`;
- `late-fusion-max`.

The text and image models may use different weight files. Make sure that each clean evaluation uses the correct model path.

---

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

All path constants are centralised in `paths.py` (`RESULT_PATH`, `CLEAN_BASE`, `PERT_BASE`, and specific CSV / parameter paths). The `--results_path` CLI argument defaults to `RESULT_PATH`.

The commands below assume:

```text
results/Recovery/classification_results
```

as the result root (the value of `RESULT_PATH` in `paths.py`).

---

## 8. Run clean inference

Clean inference is performed with `eval.py`.

The main required argument is `--modality`.

### 8.1 Feature-fusion model

```bash
python eval.py --modality feature-fusion
```

The model receives both clean text and clean images.

Configuration name:

```text
feature-fusion|clean
```

### 8.2 Text model

```bash
python eval.py --modality text
```

The model receives only clean text.

Configuration name:

```text
text|clean
```

### 8.3 Image model

```bash
python eval.py --modality image
```

The model receives only clean images.

Configuration name:

```text
image|clean
```

### 8.4 Clean late fusion

`eval.py` accepts `late-fusion` and one aggregation mode:

```bash
python eval.py --modality late-fusion --late_fusion_mode mean
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
| `--dataset` | `str` | Dataset discovered from `my_datasets.py` |

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
python attacks/text_attack.py
```

The script:

1. loads the clean text-model parameters;
2. loads `bert-base-uncased` for BERTAttack;
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
python attacks/image_attack.py
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
python attacks/multimodal_attack.py
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
python late_fusion_perturbation.py
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

Input and output paths are read from `paths.py` (`PERT_BASE`, `PER_TEXT_CSV`, etc.). To change the result root, update `RESULT_PATH` in `paths.py`.

---

## 12. Compute metrics

Use `metrics.py` to compute metrics, save `metrics.json`, create a confusion matrix, and optionally update a ROC comparison group.

### Feature fusion

Clean:

```bash
python metrics.py --type clean --modality feature-fusion
```

Both modalities perturbed:

```bash
python metrics.py --type perturbed --modality feature-fusion --perturbation_type biperturbed
```

Only text perturbed:

```bash
python metrics.py --type perturbed --modality feature-fusion --perturbation_type text-perturbed
```

Only image perturbed:

```bash
python metrics.py --type perturbed --modality feature-fusion --perturbation_type image-perturbed
```

### Text model

```bash
python metrics.py --type clean --modality text
python metrics.py --type perturbed --modality text
```

### Image model

```bash
python metrics.py --type clean --modality image
python metrics.py --type perturbed --modality image
```

### Late fusion

Clean:

```bash
python metrics.py --type clean --modality late-fusion --mode mean
python metrics.py --type clean --modality late-fusion --mode min
python metrics.py --type clean --modality late-fusion --mode max
```

Perturbed:

```bash
python metrics.py --type perturbed --modality late-fusion --mode mean --perturbation_type biperturbed
python metrics.py --type perturbed --modality late-fusion --mode min  --perturbation_type biperturbed
python metrics.py --type perturbed --modality late-fusion --mode max  --perturbation_type biperturbed
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

The result root is `RESULT_PATH` from `paths.py` (`results/Recovery/classification_results`).

---

## 13. Generate ROC plots

Run:

```bash
python create_rocs_plots.py
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
python eval.py --modality feature-fusion
python eval.py --modality text
python eval.py --modality image
```

Optionally create the clean late-fusion configurations:

```bash
python eval.py --modality late-fusion --late_fusion_mode mean
python eval.py --modality late-fusion --late_fusion_mode min
python eval.py --modality late-fusion --late_fusion_mode max
```

### Step 2: generate adversarial predictions

```bash
python attacks/multimodal_attack.py
python attacks/text_attack.py
python attacks/image_attack.py
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
python late_fusion_perturbation.py
```

### Step 5: compute metrics and generate ROC curves

```bash
python create_rocs_plots.py
```

---

## 15. Minimal workflows

The complete pipeline does not need to be executed for every experiment.

### Clean feature fusion only

```bash
python eval.py --modality feature-fusion
python metrics.py --type clean --modality feature-fusion
```

### Text robustness only

```bash
python eval.py --modality text
python attacks/text_attack.py
python metrics.py --type clean --modality text
python metrics.py --type perturbed --modality text
```

### Image robustness only

```bash
python eval.py --modality image
python attacks/image_attack.py
python metrics.py --type clean --modality image
python metrics.py --type perturbed --modality image
```

### Compare feature-fusion input conditions

```bash
python eval.py --modality feature-fusion
python attacks/multimodal_attack.py

python metrics.py --type perturbed --modality feature-fusion --perturbation_type biperturbed
python metrics.py --type perturbed --modality feature-fusion --perturbation_type text-perturbed
python metrics.py --type perturbed --modality feature-fusion --perturbation_type image-perturbed
```

### Perturbed late fusion only

```bash
python eval.py --modality text
python eval.py --modality image

python attacks/text_attack.py
python attacks/image_attack.py
python late_fusion_perturbation.py

python metrics.py --type perturbed --modality late-fusion --mode mean --perturbation_type biperturbed
python metrics.py --type perturbed --modality late-fusion --mode min  --perturbation_type biperturbed
python metrics.py --type perturbed --modality late-fusion --mode max  --perturbation_type biperturbed
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

- `RESULT_PATH` in `paths.py` (the canonical result root);
- `--results_path` CLI argument (defaults to `RESULT_PATH`).

### The test annotation file is not found

Ensure that this pattern matches an existing file:

```text
data/<DatasetName>/test.*
```

### BERTAttack cannot load its masked-language model

The first execution downloads:

```text
bert-base-uncased
```

through Hugging Face Transformers.

For offline execution, download and cache the model beforehand.

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
