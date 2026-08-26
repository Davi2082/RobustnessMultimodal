"""PGD on the image channel.

Pure algorithm: it receives an already-built classifier and one sample, and
returns the perturbed sample. It knows nothing about fusion, datasets or
result files.
"""

import numpy as np
import torch
import torchattacks
import torchvision.transforms as T
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure


class WrappedModel(torch.nn.Module):
    def __init__(self, model, fixed_txt, processor):
        super().__init__()
        self.model = model
        self.fixed_txt = fixed_txt
        self.processor = processor

    def forward(self, x):
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
        logits = torch.cat((1 - out, out), dim=1)
        return logits


def img_perturbation(model, tokenizer, processor, args, news, label,
                     steps=None, random_start=True):
    """PGD on the image channel.

    ``steps``/``random_start`` let the attack be advanced one iteration at a
    time from an already perturbed image. The step size always follows the
    configured budget, so stepping does not change how far an iteration moves.
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
    # Image processing for clean image
    process_img = processor(images=news["img"], return_tensors="pt", do_normalize=False)
    process_img = {k: v.to(device) for k, v in process_img.items()}

    # PGD Attack
    wrapped_model = WrappedModel(model, token_txt, processor)
    alpha = args.epsilon / (args.pgd_iters * args.alpha_factor)
    attack = torchattacks.PGD(
        wrapped_model,
        eps=args.epsilon,
        alpha=alpha,
        steps=args.pgd_iters if steps is None else steps,
        random_start=random_start,
    )
    corr_img = attack(process_img["pixel_values"], label)

    # Compute SSIM before converting back to PIL
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_val = ssim(preds=corr_img.float(), target=process_img["pixel_values"].float())

    # Convert perturbed tensor back to PIL Image so downstream processor can handle it uniformly
    arr = (corr_img.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    corr_news = {"txt": news["txt"], "img": Image.fromarray(arr)}

    return corr_news, ssim_val, process_img["pixel_values"]


def project_to_epsilon_ball(perturbed_img, clean_pixels, epsilon):
    """Clip a perturbed image back into the L-inf ball of the clean image.

    Stepping restarts PGD from the previous adversarial image, which would let
    the perturbation drift past epsilon. ``clean_pixels`` is in [0, 1].
    """
    to_tensor = T.ToTensor()
    perturbed = to_tensor(perturbed_img).unsqueeze(0).to(clean_pixels.device)

    if perturbed.shape != clean_pixels.shape:
        raise ValueError(
            f"Cannot project {tuple(perturbed.shape)} onto "
            f"{tuple(clean_pixels.shape)}"
        )

    projected = torch.clamp(
        perturbed, clean_pixels - epsilon, clean_pixels + epsilon
    ).clamp(0, 1)

    array = (
        projected.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
    ).astype(np.uint8)
    return Image.fromarray(array)
