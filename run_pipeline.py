import os
import subprocess
import sys

PYTHON = sys.executable
LOG_DIR = "logs"


def run(cmd, log_file, label):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, log_file)
    print(f"[pipeline] {label} ...", flush=True)
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"[pipeline] FAILED: {label}  (exit {result.returncode})  →  {log_path}")
        sys.exit(result.returncode)
    print(f"[pipeline] done:   {label}")


if __name__ == "__main__":
    # 1. Clean evaluation
    run([PYTHON, "eval.py", "--modality", "text"],           "eval_text.log",    "Clean eval — text")
    run([PYTHON, "eval.py", "--modality", "image"],          "eval_image.log",   "Clean eval — image")
    run([PYTHON, "eval.py", "--modality", "feature-fusion"], "eval_ff.log",      "Clean eval — feature-fusion")

    # 2. Adversarial attacks
    run([PYTHON, "attacks/image_attack.py"],      "image_attack.log",     "Image attack     (PGD)")
    run([PYTHON, "attacks/text_attack.py"],       "text_attack.log",      "Text attack      (BERTAttack)")
    run([PYTHON, "attacks/multimodal_attack.py"], "multimodal_attack.log","Multimodal attack (PGD + BERTAttack)")

    # 3. Late-fusion aggregation (clean + perturbed)
    run([PYTHON, "late_fusion_perturbation.py"], "late_fusion.log", "Late-fusion aggregation")

    # 4. Metrics + ROC plots
    run([PYTHON, "create_rocs_plots.py"], "roc_plots.log", "ROC plots")

    print("\n[pipeline] All steps completed.")
