"""Multi-GPU sample sharding for attack scripts.

Each attack script accepts --shard-id and --num-shards.  When num_shards > 1
the script processes only its slice of the sample list and writes results to a
shard subdirectory.  The launcher spawns one process per GPU, then merges.
"""

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd


SHARD_PREFIX = ".shard_"


def shard_sampler(sampler, dataset_len, shard_id, num_shards):
    """Return the subset of sample indices belonging to this shard.

    *sampler* is a list of indices (from ``build_subset_sampler``) or ``None``
    (full dataset).  The indices are split round-robin so each shard gets an
    even share while preserving the original ordering within each shard.
    """
    if num_shards <= 1:
        return sampler
    if sampler is None:
        sampler = list(range(dataset_len))
    return [idx for i, idx in enumerate(sampler) if i % num_shards == shard_id]


def shard_output_dir(output_dir, shard_id):
    """Per-shard subdirectory: <output_dir>/.shard_<id>/"""
    return os.path.join(output_dir, f"{SHARD_PREFIX}{shard_id}")


def merge_attack_shards(output_dir, num_shards, csv_name="perturbed_results.csv"):
    """Merge per-shard CSVs into a single result file sorted by index.

    Searches each shard directory recursively for CSVs named *csv_name* so that
    any nested output structure is handled automatically.
    Also keeps the first shard's ``parameters.json`` (updating
    runtime to the max across shards) and removes the shard subdirectories.
    """
    import json

    output_dir = Path(output_dir)

    # Discover all relative CSV paths from shard 0
    shard0 = output_dir / f"{SHARD_PREFIX}0"
    if not shard0.is_dir():
        return
    csv_relpaths = [p.relative_to(shard0) for p in shard0.rglob(csv_name)]

    max_runtime = 0.0

    for relpath in csv_relpaths:
        frames = []
        for sid in range(num_shards):
            csv_path = output_dir / f"{SHARD_PREFIX}{sid}" / relpath
            if csv_path.is_file():
                frames.append(pd.read_csv(csv_path))
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        if "index" in merged.columns:
            merged = merged.sort_values("index").reset_index(drop=True)
        dest = output_dir / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(dest, index=False)

    # Merge parameters.json — collect from all shards, keep max runtime
    params_relpaths = [p.relative_to(shard0) for p in shard0.rglob("parameters.json")]
    for relpath in params_relpaths:
        for sid in range(num_shards):
            pp = output_dir / f"{SHARD_PREFIX}{sid}" / relpath
            if pp.is_file():
                with open(pp) as f:
                    params = json.load(f)
                rt = params.get("Runtime (s)", 0) or 0
                if rt > max_runtime:
                    max_runtime = rt

    # Write merged parameters.json from first shard with max runtime
    for relpath in params_relpaths:
        first = output_dir / f"{SHARD_PREFIX}0" / relpath
        if first.is_file():
            with open(first) as f:
                params = json.load(f)
            params["Runtime (s)"] = round(max_runtime, 1)
            params["Num Shards"] = num_shards
            dest = output_dir / relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w") as f:
                json.dump(params, f, indent=4)

    # Merge perturbed text dumps if present
    text_relpaths = [p.relative_to(shard0) for p in shard0.rglob("perturbed_texts*.csv")]
    for relpath in text_relpaths:
        frames = []
        for sid in range(num_shards):
            tp = output_dir / f"{SHARD_PREFIX}{sid}" / relpath
            if tp.is_file():
                frames.append(pd.read_csv(tp))
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            if "index" in merged.columns:
                merged = merged.sort_values("index").reset_index(drop=True)
            dest = output_dir / relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(dest, index=False)

    # Clean up shard directories
    for sid in range(num_shards):
        shard_dir = output_dir / f"{SHARD_PREFIX}{sid}"
        if shard_dir.is_dir():
            shutil.rmtree(shard_dir)


def launch_sharded_attack(base_cmd, devices, output_dir, csv_name="perturbed_results.csv"):
    """Launch one subprocess per GPU device, sharding samples across them.

    *base_cmd* is the command list WITHOUT --device, --device-mlm, --shard-id,
    --num-shards, or --output-dir — those are added per worker.

    Returns (num_completed, num_failed).
    """
    num_shards = len(devices)
    output_dir = str(output_dir)
    threads = []
    results = []
    lock = threading.Lock()

    log_dir = os.path.join(output_dir, ".shard_logs")
    os.makedirs(log_dir, exist_ok=True)

    def worker(shard_id, device):
        shard_dir = shard_output_dir(output_dir, shard_id)
        cmd = list(base_cmd) + [
            "--device", device,
            "--device-mlm", device,
            "--shard-id", str(shard_id),
            "--num-shards", str(num_shards),
            "--output-dir", shard_dir,
        ]
        log_path = os.path.join(log_dir, f"shard_{shard_id}.log")
        with lock:
            print(f"  [shard {shard_id}/{num_shards}] {device}  (log: {log_path})")
            sys.stdout.flush()
        with open(log_path, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        with lock:
            status = "OK" if rc == 0 else f"FAILED (exit {rc}) — see {log_path}"
            print(f"  [shard {shard_id}/{num_shards}] {device} — {status}")
            sys.stdout.flush()
            results.append((shard_id, device, rc))

    for sid, dev in enumerate(devices):
        t = threading.Thread(target=worker, args=(sid, dev))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    failed = sum(1 for _, _, rc in results if rc != 0)

    if failed == 0:
        merge_attack_shards(output_dir, num_shards, csv_name)
        # Clean up logs after successful merge
        log_dir_path = Path(log_dir)
        if log_dir_path.is_dir():
            shutil.rmtree(log_dir_path)

    return len(results), failed
