import os

# Results root
RESULT_PATH = "results/Recovery/classification_results"
CLEAN_BASE  = os.path.join(RESULT_PATH, "clean")
PERT_BASE   = os.path.join(RESULT_PATH, "perturbed")

# Clean - CSVs
CLEAN_TEXT_CSV  = os.path.join(CLEAN_BASE, "text",  "results.csv")
CLEAN_IMAGE_CSV = os.path.join(CLEAN_BASE, "image", "results.csv")

# Clean - parameters.json
CLEAN_TEXT_PARAMS  = os.path.join(CLEAN_BASE, "text",           "parameters.json")
CLEAN_IMAGE_PARAMS = os.path.join(CLEAN_BASE, "image",          "parameters.json")
CLEAN_FF_PARAMS    = os.path.join(CLEAN_BASE, "feature-fusion", "parameters.json")

# Perturbed - output directories
PERT_IMAGE_DIR = os.path.join(PERT_BASE, "image")
PERT_TEXT_DIR  = os.path.join(PERT_BASE, "text")
PERT_FF_DIR    = os.path.join(PERT_BASE, "feature-fusion")

# Perturbed - CSVs
PER_TEXT_CSV  = os.path.join(PERT_TEXT_DIR,  "perturbed_results.csv")
PER_IMAGE_CSV = os.path.join(PERT_IMAGE_DIR, "perturbed_results.csv")

# Perturbed - parameters.json
PER_TEXT_PARAMS  = os.path.join(PERT_TEXT_DIR,  "parameters.json")
PER_IMAGE_PARAMS = os.path.join(PERT_IMAGE_DIR, "parameters.json")

# Perturbed sample dumps - generated images + texts
DATA_PERTURBED_BASE  = "data_perturbed"
DATA_PERTURBED_IMAGE = os.path.join(DATA_PERTURBED_BASE, "image")
DATA_PERTURBED_TEXT  = os.path.join(DATA_PERTURBED_BASE, "text")
DATA_PERTURBED_FF    = os.path.join(DATA_PERTURBED_BASE, "feature-fusion")

# ROC directories
ROC_BASE     = "figures/classification_results/rocs"
ROC_SETS_DIR  = os.path.join(ROC_BASE, "roc_sets")
ROC_PLOTS_DIR = os.path.join(ROC_BASE, "roc_plots")
