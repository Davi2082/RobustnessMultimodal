"""Shared device resolution for multi-GPU scripts."""

import torch


def resolve_devices(devices_arg):
    """Resolve a --devices argument into a list of device strings.

    Accepts:
      None or []       → [default from config]
      ["all"] / ["ALL"] → [cuda:0, cuda:1, ..., cuda:N-1]
      ["cuda:0", ...]  → passed through as-is
    """
    if not devices_arg:
        return None
    if len(devices_arg) == 1 and devices_arg[0].lower() == "all":
        n = torch.cuda.device_count()
        if n == 0:
            raise RuntimeError("--devices all requested but no CUDA devices found")
        return [f"cuda:{i}" for i in range(n)]
    return devices_arg
