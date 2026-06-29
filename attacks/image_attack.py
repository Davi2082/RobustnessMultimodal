import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torchattacks
import json
import glob
import argparse

from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# Custom imports
from utils import (
    load_model,
    use_model,
    img_perturbation,
    bertattack,
    load_available_datasets,
    save_predictions,
    save_perturbed_image,
)
from configuration import (
    SOURCE_LABEL,
    TARGET_LABEL,
    PGD_ITERS,
    EPSILON,
    ALPHA_FACTOR,
    DEVICE,
    SUBSET_SIZE,
)
from paths import RESULT_PATH, CLEAN_IMAGE_PARAMS, DATA_PERTURBED_IMAGE
import my_datasets

# Main evaluation function
def main():
    dataset_classes, load_functions = load_available_datasets()
    # "Parameters" contains information about the model that would be attacked
    parameters_path = CLEAN_IMAGE_PARAMS
    with open(parameters_path, 'r', encoding='utf-8') as f:
        parameters = json.load(f)
        
    # Here there are the attack parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default=parameters["Modality"], choices=["feature-fusion", "intermediate-fusion", "text", "image"])
    parser.add_argument("--name_llm", type=str, default=parameters["Name LLM"])
    parser.add_argument("--name_img_embed", type=str, default=parameters["Image Embedder Name"])
    parser.add_argument("--batch_size", type=int, default=parameters["Batch Size"])
    parser.add_argument("--model_path", type=str, default=parameters["Model Path"])
    parser.add_argument("--n_tokens", type=int, default=parameters["Number of Tokens"])
    parser.add_argument("--threshold", type=float, default=parameters["Threshold"])
    parser.add_argument("--merge_tokens", type=int, default=parameters["Merge Tokens"])
    parser.add_argument("--lora_alpha", type=int, default=parameters["LoRA Alpha"])
    parser.add_argument("--lora_r", type=int, default=parameters["LoRA R"])
    parser.add_argument("--lora_dropout", type=float, default=parameters["LoRA Dropout"])
    parser.add_argument("--use_lora", type=bool, default=parameters["Use LoRA"])
    parser.add_argument("--dataset", type=str, default=parameters["Dataset"])
    parser.add_argument("--set_params", type=bool, default=False)
    parser.add_argument("--source_label", type=int, default=SOURCE_LABEL, choices=(0,1))
    parser.add_argument("--target_label", type=int, default=TARGET_LABEL, choices=(0,1))
    parser.add_argument("--pgd_iters", type=int, default=PGD_ITERS)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--alpha_factor", type=float, default=ALPHA_FACTOR)
    parser.add_argument("--results_path", type=str, default=RESULT_PATH)
    args = parser.parse_args()

    # Device setting
    device = torch.device(DEVICE)

    # Model with relative tokenizer and processor loading
    model, tokenizer, processor = load_model(device, args, args.model_path)

    # Select dataset class and load function dynamically
    dataset_class = dataset_classes[args.dataset]
    load_func = load_functions[args.dataset]

    # Results dir setup
    output_dir = os.path.join(args.results_path, "perturbed", "image")
    os.makedirs(output_dir, exist_ok=True)

    # Dataset obtaination
    dataset_test = my_datasets.get_dataset(
        dataset_class,
        load_func,
        args.n_tokens,
        processor,
        tokenizer,
        glob.glob(f"data/{args.dataset}/test.*")[0],
        f"data/{args.dataset}/images",
    )
    
    # Dataloader creation (optionally restricted to the first N samples for quick tests)
    if SUBSET_SIZE is not None:
        sampler = list(range(min(SUBSET_SIZE, len(dataset_test))))
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, sampler=sampler)
    else:
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    y_true_list = [] # True labels 0 V 1
    indices_list = [] # Indices of the samples in the original dataset

    logits_list = [] # Logits [-inf, +inf]
    scores_list = [] # Scores [0, 1]

    for images, labels, texts, imgs_path, indices in tqdm(dataloader_test, desc="Evaluating", total=len(dataloader_test)):
        images = images.to(device)
        texts = texts.to(device)

        imgs_per_list = [] # Perturbed images
        
        # Challenging the model
        for i, label in tqdm(enumerate(labels.tolist()), desc="Challenging the model", total=len(labels), leave=False):
            # Clean news
            news = {
                "txt": dataset_test.texts[indices[i].item()],
                "img": Image.open(os.path.join(dataset_test.img_dir, dataset_test.imgs_path[indices[i].item()])).convert("RGB"),
            }
            # Only consider correctly classified samples
            if label == args.source_label:
                # Image perturbation
                news_img_per, ssim_pgd, proccess_img = img_perturbation(model, tokenizer, processor, args, news, torch.tensor([label], device=device))
                img_per = news_img_per["img"]
                # Dump the perturbed image for qualitative analysis
                save_perturbed_image(os.path.join(DATA_PERTURBED_IMAGE, "images"), indices[i].item(), img_per)
            else:
                img_per = news["img"]
                ssim_pgd = 1.0
            imgs_per_list.append(img_per)

        # Processing of multimodal corrupted images
        imgs_per_list = processor(images=imgs_per_list, return_tensors="pt", do_normalize=False).to(device)
        mean = torch.tensor(processor.image_mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(processor.image_std, device=device).view(1, -1, 1, 1)
        imgs_per_list = {"pixel_values": ((imgs_per_list["pixel_values"] - mean) / std)}
        imgs_per_list["pixel_values"] = imgs_per_list["pixel_values"].unsqueeze(1)
        imgs_per = {k: v.to(device) for k, v in imgs_per_list.items()}

        # Get predictions on corrupted samples in batch
        with torch.no_grad():
            # Evaluation with perturbed samples in every channel
            batch_scores, batch_logits = model(imgs_per, None)
            logits_list.append(batch_logits.detach().cpu())
            scores_list.append(batch_scores.detach().cpu())

        y_true_list.append(labels)
        indices_list.append(indices)
    
    logits = torch.cat(logits_list, dim=0).numpy().squeeze()
    scores = torch.cat(scores_list, dim=0).numpy().squeeze()

    y_true = torch.cat(y_true_list, dim=0).numpy()
    indices = torch.cat(indices_list, dim=0).numpy()

    # Model predictions 0 V 1
    y_preds = (scores > args.threshold).astype(int).squeeze()

    # ---------- RESULTS ----------
    # Save results
    save_predictions(y_true, y_preds, scores, logits, indices, os.path.join(output_dir, "perturbed_results.csv"))
    
    # Save "Parameters" in a file
    attack_parameters = {
        "Source Label": args.source_label,
        "Target Label": args.target_label,
        "PGD Iters": args.pgd_iters,
        "Epsilon": args.epsilon,
        "Alpha Factor": args.alpha_factor,
    }
    parameters = {
        "Model Parameters": parameters,
        "Attack Parameters": attack_parameters
    }
    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
        json.dump(parameters, f, indent=4)


if __name__ == "__main__":
    main()
