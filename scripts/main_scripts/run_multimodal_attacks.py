"""Adversarial attacks on all fusion methods.

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
  results/.../perturbed/late-fusion/<fusion>/              ← sum (both-perturbed)
  results/.../perturbed/late-fusion-pgd/<fusion>/          ← PGD only
  results/.../perturbed/late-fusion-trepat/<fusion>/       ← TREPAT only
  results/.../perturbed/late-fusion-interleaved/<fusion>/  ← interleaved
  results/.../perturbed/late-fusion-joint/<fusion>/        ← joint

Usage:
    python3 -m scripts.main_scripts.run_multimodal_attacks                          # single GPU, all 5×6
    python3 -m scripts.main_scripts.run_multimodal_attacks --devices cuda:0 cuda:1  # 2 GPUs parallel
    python3 -m scripts.main_scripts.run_multimodal_attacks --devices cuda:0 cuda:1 cuda:2  # 3 GPUs
    python3 -m scripts.main_scripts.run_multimodal_attacks --attacks sum joint --fusions mean max
"""

import argparse, os, subprocess, sys, threading, time
from queue import Queue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from configuration_files.configuration import DATASET, DEVICES
from scripts.utils.devices import resolve_devices

FUSIONS = ["min", "mean", "max", "svm-rbf", "linear", "feature-fusion"]
ATTACKS = ["pgd", "trepat", "sum", "interleaved"] #, "joint"
LOG_DIR = "logs/multimodal_attacks"

ATTACK_DIRS = {
    "pgd": "late-fusion-pgd", "trepat": "late-fusion-trepat",
    "sum": "late-fusion", "interleaved": "late-fusion-interleaved",
    "joint": "late-fusion-joint",
}
ATTACK_SCOPE = {
    "pgd": "image", "trepat": "text",
    "sum": "both", "interleaved": "both", "joint": "both",
}


def build_cmd(attack, fusion, device, result_path):
    scope = ATTACK_SCOPE[attack]
    output_dir = os.path.join(result_path, "perturbed", ATTACK_DIRS[attack], fusion)
    if attack == "joint":
        return [sys.executable, "-m", "attacks.multimodal.joint.attack",
                "--fusion", fusion, "--attack-scope", scope,
                "--device", device, "--results-path", result_path]
    optimization = "interleaved" if attack == "interleaved" else "sum"
    return [sys.executable, "-m", "attacks.multimodal.sum.attack",
            "--fusion", fusion, "--attack-scope", scope,
            "--optimization", optimization, "--device", device,
            "--output-dir", output_dir, "--results-path", result_path]


def result_csv_path(attack, fusion, result_path):
    return os.path.join(result_path, "perturbed", ATTACK_DIRS[attack],
                        fusion, "perturbed_results.csv")


def run_sequential(jobs, result_path, device, log_to_file):
    done, failed = 0, 0
    for attack, fusion in jobs:
        done += 1
        cmd = build_cmd(attack, fusion, device, result_path)
        log_path = os.path.join(LOG_DIR, f"{attack}_{fusion}.log") if log_to_file else None
        print(f"\n{'='*70}\n[{done}/{len(jobs)}] {attack} × {fusion}  ({device})")
        print(f">>> {' '.join(cmd)}")
        if log_path:
            print(f"    log: {log_path}")
        print("=" * 70)
        sys.stdout.flush()
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        else:
            rc = subprocess.run(cmd).returncode
        if rc != 0:
            failed += 1
            print(f"  FAILED (exit {rc})")
            if log_path:
                print(f"  Check log: {log_path}")
    return done, failed


def gpu_worker(device, job_queue, result_path, results, lock):
    while True:
        item = job_queue.get()
        if item is None:
            break
        idx, total, attack, fusion = item
        cmd = build_cmd(attack, fusion, device, result_path)
        log_path = os.path.join(LOG_DIR, f"{attack}_{fusion}.log")
        os.makedirs(LOG_DIR, exist_ok=True)
        with lock:
            print(f"  [{idx}/{total}] {attack} × {fusion}  →  {device}  (log: {log_path})")
            sys.stdout.flush()
        with open(log_path, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        with lock:
            print(f"  [{idx}/{total}] {attack} × {fusion}  →  {device}  {status}")
            sys.stdout.flush()
            results.append((attack, fusion, device, rc))


def run_parallel(jobs, result_path, devices):
    os.makedirs(LOG_DIR, exist_ok=True)
    total = len(jobs)
    print(f"\n  Parallel: {total} jobs across {len(devices)} GPU(s): {', '.join(devices)}")
    for i, (attack, fusion) in enumerate(jobs):
        print(f"    [{i+1}] {attack} × {fusion}")
    print()
    sys.stdout.flush()

    job_queue = Queue()
    for i, (attack, fusion) in enumerate(jobs):
        job_queue.put((i + 1, total, attack, fusion))
    for _ in devices:
        job_queue.put(None)

    results, lock, threads = [], threading.Lock(), []
    for device in devices:
        t = threading.Thread(target=gpu_worker,
                             args=(device, job_queue, result_path, results, lock))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    done = len(results)
    failed = sum(1 for _, _, _, rc in results if rc != 0)
    if failed:
        print(f"\n  Failed runs:")
        for attack, fusion, device, rc in results:
            if rc != 0:
                print(f"    {attack} × {fusion} on {device} — exit {rc}")
                print(f"      log: {LOG_DIR}/{attack}_{fusion}.log")
    return done, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--devices", nargs="+", default=DEVICES, metavar="DEV",
                        help="GPU device(s): cuda:0 cuda:1 ... or 'all'. Multiple = parallel.")
    parser.add_argument("--fusions", nargs="+", default=FUSIONS, choices=FUSIONS, metavar="F")
    parser.add_argument("--attacks", nargs="+", default=ATTACKS, choices=ATTACKS, metavar="A")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results already exist.")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    result_path = f"results/{args.dataset}/classification_results"

    all_jobs, skipped = [], 0
    for attack in args.attacks:
        for fusion in args.fusions:
            csv = result_csv_path(attack, fusion, result_path)
            if not args.force and os.path.isfile(csv):
                skipped += 1
                continue
            all_jobs.append((attack, fusion))

    args.devices = resolve_devices(args.devices)
    parallel = len(args.devices) > 1

    print("=" * 70)
    print(f"MULTIMODAL ATTACKS — {args.dataset}")
    print(f"  Fusions:  {', '.join(args.fusions)}")
    print(f"  Attacks:  {', '.join(args.attacks)}")
    if parallel:
        print(f"  Devices:  {', '.join(args.devices)}  (parallel)")
    else:
        print(f"  Device:   {args.devices[0]}  (sequential)")
    print(f"  Jobs:     {len(all_jobs)} to run, {skipped} skipped")
    print("=" * 70)

    if not all_jobs:
        print("Nothing to run.")
        return

    t0 = time.time()
    if parallel:
        done, failed = run_parallel(all_jobs, result_path, args.devices)
    else:
        done, failed = run_sequential(all_jobs, result_path, args.devices[0], args.log)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"MULTIMODAL ATTACKS COMPLETE — {done - failed}/{done} succeeded ({elapsed/60:.0f} min)")
    if failed:
        print(f"  {failed} runs FAILED — check logs in {LOG_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
