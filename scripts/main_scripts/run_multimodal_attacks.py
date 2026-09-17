"""Adversarial attacks on all fusion methods.

Runs attack types against every fusion method, sharding samples across GPUs
when multiple devices are configured.

  Attack type             Script                           Optimization
  ─────────────────────── ──────────────────────────────── ────────────
  PGD only                attacks.multimodal.sum.attack    sum, scope=image
  TREPAT only             attacks.multimodal.sum.attack    sum, scope=text
  PGD + TREPAT (sum)      attacks.multimodal.sum.attack    sum, scope=both
  PGD + TREPAT (alt.)     attacks.multimodal.sum.attack    interleaved, scope=both
  HotFlip + PGD (joint)   attacks.multimodal.joint.attack  scope=both

With multiple GPUs, each attack job splits its samples across all devices.
Each GPU loads both the victim model and any attack models (rewriter/MLM),
giving near-linear speedup for sequential attacks like TREPAT.

Usage:
    python3 -m scripts.main_scripts.run_multimodal_attacks
    python3 -m scripts.main_scripts.run_multimodal_attacks --devices cuda:0 cuda:1
    python3 -m scripts.main_scripts.run_multimodal_attacks --attacks sum joint --fusions mean max
"""

import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from configuration_files.configuration import (
    ATTACK_SCOPE, DATASET, PIPELINE_ATTACKS, PIPELINE_FUSIONS, dataset_devices,
)
from configuration_files.paths import clean_image_params, clean_text_params, model_perturbed_dir
from models.fusion import fusion_head_path
from scripts.utils.devices import resolve_devices

FUSIONS = PIPELINE_FUSIONS
ATTACKS = PIPELINE_ATTACKS
LOG_DIR = "logs/multimodal_attacks"


def build_base_cmd(attack, fusion, dataset):
    """Build the command list WITHOUT --device/--device-mlm/--output-dir/--shard args."""
    scope = ATTACK_SCOPE[attack]
    text_params = clean_text_params(dataset)
    image_params = clean_image_params(dataset)
    if attack == "joint":
        cmd = [sys.executable, "-m", "attacks.multimodal.joint.attack",
               "--fusion", fusion, "--attack-scope", scope,
               "--text-parameters", text_params, "--image-parameters", image_params,
               "--dataset", dataset]
    else:
        optimization = "interleaved" if attack == "interleaved" else "sum"
        cmd = [sys.executable, "-m", "attacks.multimodal.sum.attack",
               "--fusion", fusion, "--attack-scope", scope,
               "--optimization", optimization,
               "--text-parameters", text_params, "--image-parameters", image_params,
               "--dataset", dataset]
        if fusion == "svm-rbf":
            cmd += ["--svm-model", fusion_head_path("svm-rbf", dataset)]
    return cmd


def result_csv_path(attack, fusion, dataset):
    return os.path.join(model_perturbed_dir(fusion, attack, dataset), "perturbed_results.csv")


def run_single_gpu(base_cmd, device, output_dir):
    """Run a single job on one GPU (no sharding)."""
    cmd = base_cmd + ["--device", device, "--device-mlm", device, "--output-dir", output_dir]
    return subprocess.run(cmd).returncode


def run_sharded(base_cmd, devices, output_dir):
    """Run a single job sharded across multiple GPUs."""
    from scripts.utils.parallel import launch_sharded_attack
    _, failed = launch_sharded_attack(base_cmd, devices, output_dir)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--devices", nargs="+", default=None, metavar="DEV",
                        help="GPU device(s): cuda:0 cuda:1 ... or 'all'. "
                             "Multiple devices shard samples across GPUs.")
    parser.add_argument("--fusions", nargs="+", default=FUSIONS, choices=FUSIONS, metavar="F")
    parser.add_argument("--attacks", nargs="+", default=ATTACKS, choices=ATTACKS, metavar="A")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results already exist.")
    args = parser.parse_args()

    if args.devices is None:
        args.devices = dataset_devices(args.dataset)
    args.devices = resolve_devices(args.devices)
    multi_gpu = len(args.devices) > 1

    all_jobs, skipped = [], 0
    for attack in args.attacks:
        for fusion in args.fusions:
            csv = result_csv_path(attack, fusion, args.dataset)
            if not args.force and os.path.isfile(csv):
                skipped += 1
                continue
            all_jobs.append((attack, fusion))

    print("=" * 70)
    print(f"MULTIMODAL ATTACKS — {args.dataset}")
    print(f"  Fusions:  {', '.join(args.fusions)}")
    print(f"  Attacks:  {', '.join(args.attacks)}")
    if multi_gpu:
        print(f"  Devices:  {', '.join(args.devices)}  (sample sharding)")
    else:
        print(f"  Device:   {args.devices[0]}")
    print(f"  Jobs:     {len(all_jobs)} to run, {skipped} skipped")
    print("=" * 70)

    if not all_jobs:
        print("Nothing to run.")
        return

    t0 = time.time()
    done, failed = 0, 0

    for i, (attack, fusion) in enumerate(all_jobs):
        output_dir = model_perturbed_dir(fusion, attack, args.dataset)
        base_cmd = build_base_cmd(attack, fusion, args.dataset)

        print(f"\n[{i+1}/{len(all_jobs)}] {attack} × {fusion}")
        sys.stdout.flush()

        if multi_gpu:
            rc = run_sharded(base_cmd, args.devices, output_dir)
        else:
            rc = run_single_gpu(base_cmd, args.devices[0], output_dir)

        done += 1
        if rc != 0:
            failed += 1
            print(f"  FAILED: {attack} × {fusion}")
        else:
            params_path = os.path.join(output_dir, "parameters.json")
            if os.path.isfile(params_path):
                with open(params_path) as _f:
                    rt = json.load(_f).get("Runtime (s)", None)
                if rt is not None:
                    print(f"  {attack} × {fusion} — {rt:.1f}s")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"MULTIMODAL ATTACKS COMPLETE — {done - failed}/{done} succeeded ({elapsed/60:.0f} min)")
    if failed:
        print(f"  {failed} runs FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
