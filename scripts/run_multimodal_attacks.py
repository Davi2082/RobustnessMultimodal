"""Fusion-aware adversarial attacks on Recovery.

Runs five attack types against every fusion method:

  Attack type             Script                           Optimization
  ─────────────────────── ──────────────────────────────── ────────────
  PGD only                attacks.multimodal.sum.attack    sum, scope=image
  TREPAT only             attacks.multimodal.sum.attack    sum, scope=text
  PGD + TREPAT (sum)      attacks.multimodal.sum.attack    sum, scope=both
  PGD + TREPAT (alt.)     attacks.multimodal.sum.attack    interleaved, scope=both
  HotFlip + PGD (joint)   attacks.multimodal.joint.attack  scope=both

Fusion methods:  min, mean, max, svm-rbf, linear, feature-fusion

Output layout:
  results/.../perturbed/late-fusion/<fusion>/              ← sum attack (PGD/TREPAT/both)
  results/.../perturbed/late-fusion-interleaved/<fusion>/   ← interleaved
  results/.../perturbed/late-fusion-joint/<fusion>/         ← joint (default in script)

Usage:
    python3 run_multimodal_attacks.py                      # all attacks × all fusions
    python3 run_multimodal_attacks.py --attacks sum         # only sum attacks
    python3 run_multimodal_attacks.py --fusions mean max    # only mean and max
    python3 run_multimodal_attacks.py --device cuda:1
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configuration_files.paths import RESULT_PATH

FUSIONS = ["min", "mean", "max", "svm-rbf", "linear", "feature-fusion"]
ATTACKS = ["sum", "interleaved", "joint"]
LOG_DIR = "logs/multimodal_attacks"


def run(cmd, log_path=None):
    label = " ".join(cmd)
    print(f"\n{'='*70}")
    print(f">>> {label}")
    if log_path:
        print(f"    log: {log_path}")
    print(f"{'='*70}")
    sys.stdout.flush()

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        if log_path:
            print(f"  Check log: {log_path}")
    return result.returncode


def run_sum(fusion, device, log_to_file):
    """PGD-only, TREPAT-only, and PGD+TREPAT sum in one --attack-scope=both call."""
    log = os.path.join(LOG_DIR, f"sum_{fusion}.log") if log_to_file else None
    return run([
        sys.executable, "-m", "attacks.multimodal.sum.attack",
        "--fusion", fusion,
        "--attack-scope", "both",
        "--optimization", "sum",
        "--device", device,
    ], log)


def run_interleaved(fusion, device, log_to_file):
    """PGD+TREPAT alternating (interleaved optimization)."""
    output_dir = os.path.join(
        RESULT_PATH, "perturbed", "late-fusion-interleaved", fusion,
    )
    log = os.path.join(LOG_DIR, f"interleaved_{fusion}.log") if log_to_file else None
    return run([
        sys.executable, "-m", "attacks.multimodal.sum.attack",
        "--fusion", fusion,
        "--attack-scope", "both",
        "--optimization", "interleaved",
        "--output-dir", output_dir,
        "--device", device,
    ], log)


def run_joint(fusion, device, log_to_file):
    """HotFlip+PGD joint (shared backward pass)."""
    log = os.path.join(LOG_DIR, f"joint_{fusion}.log") if log_to_file else None
    return run([
        sys.executable, "-m", "attacks.multimodal.joint.attack",
        "--fusion", fusion,
        "--attack-scope", "both",
        "--device", device,
    ], log)


ATTACK_RUNNERS = {
    "sum": run_sum,
    "interleaved": run_interleaved,
    "joint": run_joint,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fusions", nargs="+", default=FUSIONS,
                        choices=FUSIONS, metavar="F",
                        help=f"Fusion methods to attack (default: all)")
    parser.add_argument("--attacks", nargs="+", default=ATTACKS,
                        choices=ATTACKS, metavar="A",
                        help=f"Attack types to run (default: all)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log", action="store_true",
                        help="Write subprocess output to log files instead of stdout")
    args = parser.parse_args()

    total = len(args.fusions) * len(args.attacks)
    done, failed = 0, 0

    print("=" * 70)
    print("MULTIMODAL ATTACKS — Recovery dataset")
    print(f"  Fusions: {', '.join(args.fusions)}")
    print(f"  Attacks: {', '.join(args.attacks)}")
    print(f"  Device:  {args.device}")
    print(f"  Total:   {total} runs")
    print("=" * 70)

    t0 = time.time()

    for attack in args.attacks:
        runner = ATTACK_RUNNERS[attack]
        for fusion in args.fusions:
            done += 1
            print(f"\n[{done}/{total}] {attack} × {fusion}")
            rc = runner(fusion, args.device, args.log)
            if rc != 0:
                failed += 1

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"MULTIMODAL ATTACKS COMPLETE — {done - failed}/{total} succeeded "
          f"({elapsed/60:.0f} min)")
    if failed:
        print(f"  {failed} runs FAILED — check logs in {LOG_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
