"""PGD on the image channel.

Pure algorithm: it receives an already-built classifier and one sample, and
returns the perturbed sample. It knows nothing about fusion, datasets or
result files.

The loop reproduces torchattacks.PGD (untargeted, L-inf, sign steps,
projection on the eps-ball around the input, clipping to [0,1]) so that
can log the loss and the fused score at every iteration.
"""

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure


class WrappedModel(torch.nn.Module):
    def __init__(self, model, fixed_txt, processor, threshold=0.5):
        super().__init__()
        self.model = model
        self.fixed_txt = fixed_txt
        self.processor = processor
        self.threshold = threshold

    def score(self, x):
        """Fused P(Real) for a [0,1] image batch with the fixed text."""
        mean = torch.tensor(
            self.processor.image_mean, device=x.device, dtype=x.dtype
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            self.processor.image_std, device=x.device, dtype=x.dtype
        ).view(1, -1, 1, 1)
        x = (x - mean) / std
        if self.fixed_txt is not None:
            fixed_txt_repeated = {}
            for key, tensor in self.fixed_txt.items():
                tensor = tensor.to(x.device)

                if tensor.size(0) == 1 and x.size(0) > 1:
                    fixed_txt_repeated[key] = tensor.repeat(x.size(0), 1)
                elif tensor.size(0) == x.size(0):
                    fixed_txt_repeated[key] = tensor
                else:
                    raise ValueError(
                        f"Batch mismatch for {key}: text batch={tensor.size(0)}, image batch={x.size(0)}"
                    )
            out, _ = self.model({"pixel_values": x}, fixed_txt_repeated)
        else:
            out, _ = self.model({"pixel_values": x}, None)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        return out

    def forward(self, x):
        # Class 1 (Real) wins iff p > t, so the CE loss targets the method's own
        # decision boundary (for t = 0.5 this has the same sign gradient as [1-p, p]).
        out = self.score(x)
        return torch.cat((torch.full_like(out, float(self.threshold)), out), dim=1)


def _pgd(wrapped, images, labels, eps, alpha, steps, random_start, history=None):
    """torchattacks.PGD (untargeted) with optional per-step logging."""
    loss_fn = torch.nn.CrossEntropyLoss()
    images = images.clone().detach()
    adv = images.clone().detach()
    if random_start:
        adv = torch.clamp(adv + torch.empty_like(adv).uniform_(-eps, eps), 0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        with torch.enable_grad():
            logits = wrapped(adv)
            cost = loss_fn(logits, labels)
            grad = torch.autograd.grad(cost, adv)[0]
        if history is not None:
            history.append({"loss": float(cost.detach()), "score": float(logits[0, 1].detach())})
        adv = adv.detach() + alpha * grad.sign()
        delta = torch.clamp(adv - images, min=-eps, max=eps)
        adv = torch.clamp(images + delta, min=0, max=1).detach()
    return adv


def img_perturbation(model, tokenizer, processor, args, news, label,
                     steps=None, random_start=True, pixel_values=None, history=None):
    """PGD on the image channel.

    ``steps``/``random_start`` let the attack be advanced one iteration at a
    time from an already perturbed image. The step size always follows the
    configured budget, so stepping does not change how far an iteration moves.

    ``pixel_values`` is an optional [0,1] tensor (before CLIP normalization).
    When provided, the PIL image in ``news["img"]`` is not re-processed,
    avoiding a redundant Image.open + processor call.

    ``history``: optional list; one dict per PGD step is appended with the
    CE loss being maximised and the fused score P(Real) before that step.
    """
    device = label.device
    # Text tokenization for fixed text
    token_txt = tokenizer(
        news["txt"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_attention_mask=False,
        max_length=args.n_tokens,
    )
    # Image: use pre-computed [0,1] tensor if available, otherwise process PIL
    if pixel_values is not None:
        process_img = {"pixel_values": pixel_values.to(device)}
    else:
        process_img = processor(images=news["img"], return_tensors="pt", do_normalize=False)
        process_img = {k: v.to(device) for k, v in process_img.items()}

    wrapped_model = WrappedModel(model, token_txt, processor,
                                 threshold=getattr(args, "threshold", 0.5))
    # Step size is fixed by the full single-modality budget (pgd_step_iters),
    # so a halved joint budget takes fewer steps of the same size instead of
    # half as many steps of double size.
    alpha = args.epsilon / (getattr(args, "pgd_step_iters", args.pgd_iters) * args.alpha_factor)
    corr_img = _pgd(
        wrapped_model,
        process_img["pixel_values"],
        label,
        eps=args.epsilon,
        alpha=alpha,
        steps=args.pgd_iters if steps is None else steps,
        random_start=random_start,
        history=history,
    )

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_val = ssim(preds=corr_img.float(), target=process_img["pixel_values"].float())

    return corr_img.detach(), ssim_val, process_img["pixel_values"]


def project_to_epsilon_ball(perturbed, clean_pixels, epsilon):
    """Clip a perturbed [0,1] tensor into the L-inf ball of clean_pixels."""
    if perturbed.shape != clean_pixels.shape:
        raise ValueError(
            f"Cannot project {tuple(perturbed.shape)} onto "
            f"{tuple(clean_pixels.shape)}"
        )
    return torch.clamp(
        perturbed, clean_pixels - epsilon, clean_pixels + epsilon
    ).clamp(0, 1)
