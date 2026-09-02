"""One classifier interface for every fusion strategy.

build_classifier() returns late fusion (two unimodal models plus a rule) or
feature fusion (one joint model); both answer forward(images, texts), so no
fusion type needs a driver of its own.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from configuration_files.configuration import (
    DATASET, NAME_IMG_EMBED, FF_WEIGHTS_PATH, LATE_FUSION_INPUT,
)
from configuration_files.paths import DATASET_WEIGHTS_DIR

HEAD_FILES = {"svm-rbf": "svm_rbf_head.pkl", "linear": "linear_head.pkl"}
HEAD_METADATA = os.path.join("results", DATASET, "classification_results", "fusion_heads.json")


def fusion_head_path(rule: str) -> str:
    """Where the fitted head for a learned rule is stored."""
    return os.path.join(DATASET_WEIGHTS_DIR, HEAD_FILES[rule])


def head_input_space() -> str:
    """Feature space the stored heads were fitted on."""
    if os.path.exists(HEAD_METADATA):
        with open(HEAD_METADATA, encoding="utf-8") as handle:
            return json.load(handle).get("input_space", LATE_FUSION_INPUT)
    return LATE_FUSION_INPUT


LATE_FUSION_RULES = ("mean", "min", "max", "svm-rbf", "linear")
LEARNED_RULES = ("svm-rbf", "linear")
FUSION_CHOICES = LATE_FUSION_RULES + ("feature-fusion",)

PARAMETER_KEYS = {
    "Name LLM", "Image Embedder Name", "Model Path", "Number of Tokens",
    "Merge Tokens", "LoRA Alpha", "LoRA R", "LoRA Dropout", "Use LoRA", "Dataset",
}

COMPONENT_OUTPUT_NAMES = ("text_scores", "text_logits", "image_scores", "image_logits")
COMPONENT_OUTPUT_COLUMNS = tuple(name[:-1] for name in COMPONENT_OUTPUT_NAMES)


def read_parameters(path: Path, description: str) -> dict[str, Any]:
    """Read and validate one clean-model parameter file."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            parameters = json.load(handle)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{description} parameters do not exist: {path}. "
            "Run the clean unimodal evaluation first."
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {description} parameters: {path}") from error

    missing = PARAMETER_KEYS.difference(parameters)
    if missing:
        raise ValueError(f"{description} parameters are missing: {', '.join(sorted(missing))}")
    return parameters


def model_args_from_parameters(
    parameters: dict[str, Any], modality: str, model_path: Path,
) -> argparse.Namespace:
    """Build exactly the namespace expected by ``utils.load_model``."""
    return argparse.Namespace(
        modality=modality,
        name_llm=parameters["Name LLM"],
        name_img_embed=parameters["Image Embedder Name"],
        model_path=str(model_path),
        n_tokens=int(parameters["Number of Tokens"]),
        merge_tokens=parameters["Merge Tokens"],
        lora_alpha=parameters["LoRA Alpha"],
        lora_r=parameters["LoRA R"],
        lora_dropout=parameters["LoRA Dropout"],
        use_lora=parameters["Use LoRA"],
        set_params=False,
    )


def load_pytorch_head(rule: str, device="cpu") -> torch.nn.Module:
    """Load a fitted sklearn head and return the equivalent PyTorch module."""
    path = fusion_head_path(rule)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No fitted {rule} head at {path}")
    fitted = joblib.load(path)
    if rule == "svm-rbf":
        return DifferentiableRBFSVMFusion(fitted).to(device).eval()
    return linear_fusion_from_sklearn(fitted).to(device).eval()


def pytorch_head_score(rule: str, features, device="cpu"):
    """Run a learned fusion head on numpy features, returning numpy scores."""
    head = load_pytorch_head(rule, device)
    with torch.no_grad():
        t = torch.as_tensor(features, dtype=torch.float32, device=device)
        return head(t).cpu().numpy()


class DifferentiableRBFSVMFusion(torch.nn.Module):
    """Differentiable binary StandardScaler + RBF-SVC probability model."""

    def __init__(self, fitted_model: Any):
        super().__init__()

        scaler: StandardScaler | None = None
        svc: SVC

        if isinstance(fitted_model, Pipeline):
            steps = list(fitted_model.steps)
            if not steps or not isinstance(steps[-1][1], SVC):
                raise TypeError("The final SVM pipeline step must be sklearn.svm.SVC")
            svc = steps[-1][1]
            preprocessing = [step for _, step in steps[:-1] if step != "passthrough"]
            if len(preprocessing) > 1 or (
                preprocessing and not isinstance(preprocessing[0], StandardScaler)
            ):
                raise TypeError(
                    "Differentiable SVM fusion supports only an optional StandardScaler before SVC"
                )
            if preprocessing:
                scaler = preprocessing[0]
        elif isinstance(fitted_model, SVC):
            svc = fitted_model
        else:
            raise TypeError("--svm-model must contain an SVC or a Pipeline ending in SVC")

        if svc.kernel != "rbf":
            raise ValueError(f"Expected an RBF SVC, got kernel={svc.kernel!r}")
        if not getattr(svc, "probability", False):
            raise ValueError("The SVC must have been trained with probability=True")
        if getattr(svc, "n_features_in_", None) != 2:
            raise ValueError("The SVC must use exactly [text, image] as its two features")
        if len(svc.classes_) != 2 or 1 not in svc.classes_:
            raise ValueError(f"The SVC must be binary and contain class 1; got {svc.classes_}")
        if len(svc.probA_) != 1 or len(svc.probB_) != 1:
            raise ValueError("The fitted SVC does not contain binary probability calibration")

        if scaler is None:
            mean = np.zeros(2, dtype=np.float32)
            scale = np.ones(2, dtype=np.float32)
        else:
            if getattr(scaler, "n_features_in_", None) != 2:
                raise ValueError("The StandardScaler must contain exactly two features")
            mean = (np.asarray(scaler.mean_, dtype=np.float32)
                    if scaler.with_mean else np.zeros(2, dtype=np.float32))
            scale = (np.asarray(scaler.scale_, dtype=np.float32)
                     if scaler.with_std else np.ones(2, dtype=np.float32))

        self.register_buffer("mean", torch.as_tensor(mean))
        self.register_buffer("scale", torch.as_tensor(scale))
        self.register_buffer("support_vectors",
                             torch.as_tensor(np.asarray(svc.support_vectors_), dtype=torch.float32))
        self.register_buffer("dual_coef",
                             torch.as_tensor(np.asarray(svc.dual_coef_[0]), dtype=torch.float32))
        self.register_buffer("intercept",
                             torch.tensor(float(svc.intercept_[0]), dtype=torch.float32))
        self.register_buffer("gamma",
                             torch.tensor(float(svc._gamma), dtype=torch.float32))
        self.register_buffer("prob_a",
                             torch.tensor(float(svc.probA_[0]), dtype=torch.float32))
        self.register_buffer("prob_b",
                             torch.tensor(float(svc.probB_[0]), dtype=torch.float32))

        # libSVM's sigmoid outputs P(class_0); invert when class 1 is second.
        self.invert_probability = svc.classes_[0] == 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the differentiable probability of class 1."""
        features = features.to(dtype=self.support_vectors.dtype)
        transformed = (features - self.mean) / self.scale
        squared_distance = (transformed.unsqueeze(1) - self.support_vectors.unsqueeze(0)).square().sum(dim=-1)
        kernel = torch.exp(-self.gamma * squared_distance)
        decision = kernel.matmul(self.dual_coef) + self.intercept
        probability = torch.sigmoid(-(self.prob_a * decision + self.prob_b))
        if self.invert_probability:
            probability = 1.0 - probability
        return probability


def linear_fusion_from_sklearn(fitted_model: Any) -> torch.nn.Module:
    """Build a torch.nn.Sequential (StandardScaler → Linear → Sigmoid) from a fitted sklearn pipeline."""
    estimator = getattr(fitted_model, "best_estimator_", fitted_model)
    steps = dict(getattr(estimator, "named_steps", {}))
    scaler = steps.get("standardscaler") or steps.get("scaler")
    lr = steps.get("lr") or steps.get("logisticregression")
    if lr is None:
        raise TypeError("The linear fusion head must be a Pipeline ending in LogisticRegression")

    classes = np.asarray(lr.classes_)
    mean = torch.zeros(2)
    scale = torch.ones(2)
    if scaler is not None:
        if getattr(scaler, "with_mean", True):
            mean = torch.as_tensor(scaler.mean_, dtype=torch.float32)
        if getattr(scaler, "with_std", True):
            scale = torch.as_tensor(scaler.scale_, dtype=torch.float32)

    weight = torch.as_tensor(lr.coef_[0], dtype=torch.float32) / scale
    bias = torch.tensor(float(lr.intercept_[0]) - (lr.coef_[0] @ (mean / scale).numpy()), dtype=torch.float32)
    if int(classes[0]) == 1:
        weight, bias = -weight, -bias

    linear = torch.nn.Linear(2, 1)
    linear.weight.data = weight.unsqueeze(0)
    linear.bias.data = bias.unsqueeze(0)

    return torch.nn.Sequential(linear, torch.nn.Sigmoid(), torch.nn.Flatten(0))


class LateFusionClassifier(torch.nn.Module):
    """Expose two frozen unimodal models and score fusion as one classifier."""

    def __init__(self, text_model, image_model, fusion, svm_fusion=None, svm_input="scores"):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.fusion = fusion
        self.svm_fusion = svm_fusion
        self.svm_input = svm_input

        if fusion not in set(LATE_FUSION_RULES):
            raise ValueError(f"Unknown fusion: {fusion}")
        if fusion in LEARNED_RULES and svm_fusion is None:
            raise ValueError(f"a fitted head is required for fusion={fusion!r}")
        if svm_input not in {"scores", "logits"}:
            raise ValueError(f"Unknown SVM input space: {svm_input}")

        for p in self.text_model.parameters():
            p.requires_grad_(False)
        for p in self.image_model.parameters():
            p.requires_grad_(False)
        self.text_model.eval()
        self.image_model.eval()

    @staticmethod
    def _vector(value: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        if value.ndim == 0:
            value = value.unsqueeze(0)
        if value.ndim > 2 or (value.ndim == 2 and value.shape[1] != 1):
            raise ValueError(f"{name} must contain one scalar per sample; got {value.shape}")
        return value.reshape(-1)

    def forward(self, images, texts, return_components=False):
        if images is None or texts is None:
            raise ValueError("Late fusion requires both image and text inputs")

        text_score, text_logit = self.text_model(None, texts)
        image_score, image_logit = self.image_model(images, None)

        text_score = self._vector(text_score, "text score")
        image_score = self._vector(image_score, "image score")
        text_logit = self._vector(text_logit, "text logit")
        image_logit = self._vector(image_logit, "image logit")

        batch_sizes = {text_score.numel(), image_score.numel(), text_logit.numel(), image_logit.numel()}
        if len(batch_sizes) != 1:
            raise ValueError(
                f"Text and image models returned different batch sizes: "
                f"text={text_score.numel()}, image={image_score.numel()}"
            )

        if self.fusion == "mean":
            fused_score = 0.5 * (text_score + image_score)
        elif self.fusion == "min":
            fused_score = torch.minimum(text_score, image_score)
        elif self.fusion == "max":
            fused_score = torch.maximum(text_score, image_score)
        else:
            if self.svm_input == "scores":
                features = torch.stack((text_score, image_score), dim=1)
            else:
                features = torch.stack((text_logit, image_logit), dim=1)
            fused_score = self.svm_fusion(features)

        eps = torch.finfo(fused_score.dtype).eps
        fused_logit = torch.logit(fused_score.clamp(eps, 1.0 - eps))
        fused_outputs = (fused_score.unsqueeze(1), fused_logit.unsqueeze(1))

        if not return_components:
            return fused_outputs
        return (
            *fused_outputs,
            text_score.unsqueeze(1), text_logit.unsqueeze(1),
            image_score.unsqueeze(1), image_logit.unsqueeze(1),
        )


class FeatureFusionClassifier(torch.nn.Module):
    """Wrap the jointly trained Themis in the same output contract."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.text_model = model
        self.image_model = model

    def forward(self, images, texts, return_components=False):
        score, logit = self.model(images, texts)
        score = score.reshape(-1, 1)
        logit = logit.reshape(-1, 1)
        if not return_components:
            return score, logit
        return score, logit, score, logit, score, logit


def build_classifier(args, device):
    """Return the classifier named by ``args.fusion``."""
    from utils import load_model

    if args.fusion == "feature-fusion":
        ff_args = model_args_from_parameters(
            args.text_parameters_data, "feature-fusion", Path(FF_WEIGHTS_PATH)
        )
        ff_args.name_img_embed = NAME_IMG_EMBED
        model, tokenizer, processor = load_model(device, ff_args, str(FF_WEIGHTS_PATH))
        return FeatureFusionClassifier(model).to(device).eval(), tokenizer, processor

    if args.fusion not in LATE_FUSION_RULES:
        raise ValueError(f"Unknown fusion {args.fusion!r}; expected one of {FUSION_CHOICES}")

    text_args = model_args_from_parameters(args.text_parameters_data, "text", args.text_model_path)
    image_args = model_args_from_parameters(args.image_parameters_data, "image", args.image_model_path)
    text_model, tokenizer, _ = load_model(device, text_args, str(args.text_model_path))
    image_model, _, processor = load_model(device, image_args, str(args.image_model_path))

    fusion_head = None
    if args.fusion in LEARNED_RULES:
        head_path = fusion_head_path(args.fusion)
        if not os.path.exists(head_path):
            raise FileNotFoundError(
                f"No fitted {args.fusion} head at {head_path}. "
                "Run scripts/fit_fusion_heads.py first."
            )
        fitted = joblib.load(head_path)
        fusion_head = (
            DifferentiableRBFSVMFusion(fitted) if args.fusion == "svm-rbf"
            else linear_fusion_from_sklearn(fitted)
        ).to(device)

    classifier = LateFusionClassifier(
        text_model=text_model, image_model=image_model,
        fusion=args.fusion, svm_fusion=fusion_head, svm_input=head_input_space(),
    ).to(device)
    classifier.eval()
    return classifier, tokenizer, processor
