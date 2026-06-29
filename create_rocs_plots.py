import subprocess
import sys

PYTHON = sys.executable

if __name__ == "__main__":
    # Rocs for feature-fusion analysis
    subprocess.run([PYTHON, "metrics.py", "--type", "clean", "--modality", "feature-fusion", "--roc-set", "feature-fusion"])
    subprocess.run([PYTHON, "metrics.py", "--type", "perturbed", "--modality", "feature-fusion", "--perturbation_type", "biperturbed", "--roc-set", "feature-fusion"])
    subprocess.run([PYTHON, "metrics.py", "--type", "perturbed", "--modality", "feature-fusion", "--perturbation_type", "image-perturbed", "--roc-set", "feature-fusion"])
    subprocess.run([PYTHON, "metrics.py", "--type", "perturbed", "--modality", "feature-fusion", "--perturbation_type", "text-perturbed", "--roc-set", "feature-fusion"])

    # Rocs for late-fusion-mean analysis
    subprocess.run([PYTHON, "metrics.py", "--type", "clean", "--modality", "late-fusion", "--mode", "mean", "--roc-set", "late-fusion-mean"])
    subprocess.run([PYTHON, "metrics.py", "--type", "perturbed", "--modality", "late-fusion", "--mode", "mean", "--perturbation_type", "biperturbed", "--roc-set", "late-fusion-mean"])
    subprocess.run([PYTHON, "metrics.py", "--type", "perturbed", "--modality", "late-fusion", "--mode", "mean", "--perturbation_type", "image-perturbed", "--roc-set", "late-fusion-mean"])
    subprocess.run([PYTHON, "metrics.py", "--type", "perturbed", "--modality", "late-fusion", "--mode", "mean", "--perturbation_type", "text-perturbed", "--roc-set", "late-fusion-mean"])

    # Rocs for late-fusion-min analysis
    subprocess.run([PYTHON, "metrics.py", "--type", "clean", "--modality", "late-fusion", "--mode", "min", "--roc-set", "late-fusion-min"])
    subprocess.run([PYTHON, "metrics.py", "--type", "perturbed", "--modality", "late-fusion", "--mode", "min", "--perturbation_type", "biperturbed", "--roc-set", "late-fusion-min"])
    subprocess.run([PYTHON, "metrics.py", "--type", "perturbed", "--modality", "late-fusion", "--mode", "min", "--perturbation_type", "image-perturbed", "--roc-set", "late-fusion-min"])
    subprocess.run([PYTHON, "metrics.py", "--type", "perturbed", "--modality", "late-fusion", "--mode", "min", "--perturbation_type", "text-perturbed", "--roc-set", "late-fusion-min"])
    
    # Rocs for late-fusion-max analysis
    subprocess.run([PYTHON, "metrics.py", "--type", "clean", "--modality", "late-fusion", "--mode", "max", "--roc-set", "late-fusion-max"])
    subprocess.run([PYTHON, "metrics.py", "--type", "perturbed", "--modality", "late-fusion", "--mode", "max", "--perturbation_type", "biperturbed", "--roc-set", "late-fusion-max"])
    subprocess.run([PYTHON, "metrics.py", "--type", "perturbed", "--modality", "late-fusion", "--mode", "max", "--perturbation_type", "image-perturbed", "--roc-set", "late-fusion-max"])
    subprocess.run([PYTHON, "metrics.py", "--type", "perturbed", "--modality", "late-fusion", "--mode", "max", "--perturbation_type", "text-perturbed", "--roc-set", "late-fusion-max"])