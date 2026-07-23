"""Command-line entry point for fusion-aware late-fusion attacks.

The implementation lives in ``attacks.late_fusion_multimodal_attack`` so the
CLI, the batch runner, and tests all exercise one canonical code path.
"""

from attacks.late_fusion_multimodal_attack import (
    DifferentiableRBFSVMFusion,
    LateFusionClassifier,
    main,
)

__all__ = (
    "DifferentiableRBFSVMFusion",
    "LateFusionClassifier",
    "main",
)

if __name__ == "__main__":
    main()
