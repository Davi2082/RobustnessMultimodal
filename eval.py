import os
import torch
import json
import glob
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate

# Custom imports
import my_datasets
from utils import (
    load_available_datasets,
    load_model,
    plot_text_vs_image,
    compute_threshold,
    preds_fusion,
)
from configuration import (
    NAME_LLM,
    NAME_IMG_EMBED,
    TEXT_WEIGHTS_PATH,
    IMAGE_WEIGHTS_PATH,
    BATCH_SIZE,
    N_TOKENS,
    THRESHOLD,
    DEVICE_EVAL,
    SUBSET_SIZE,
)
from paths import RESULT_PATH

# Main evaluation function
def main():
    dataset_classes, load_functions = load_available_datasets()
    # Here there are the evaluation parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, choices=["feature-fusion", "intermediate-fusion", "late-fusion", "text", "image"])
    parser.add_argument("--late_fusion_mode", type=str, choices=["mean", "max", "min"])
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--name_llm", type=str, default=NAME_LLM)
    parser.add_argument("--name_img_embed", type=str, default=NAME_IMG_EMBED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--n_tokens", type=int, default=N_TOKENS)
    parser.add_argument("--merge_tokens", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--use_lora", type=bool)
    parser.add_argument("--set_params", type=bool, default=True)
    parser.add_argument("--results_path", type=str, default=RESULT_PATH)
    parser.add_argument("--dataset", type=str, default="Recovery", choices=list(dataset_classes.keys()))
    args = parser.parse_args()

    # "Parameters" will be the dictionary that will be saved in the json file with the evaluation parameters
    parameters = {
        "Modality": args.modality,
        "Name LLM": args.name_llm,
        "Image Embedder Name": args.name_img_embed,
        "Batch Size": args.batch_size,
        "Number of Tokens": args.n_tokens,
        "Dataset": args.dataset,
    }

    # Device setting
    device = torch.device(DEVICE_EVAL)

    # Model with relative tokenizer and processor loading
    if args.modality == "late-fusion":
        args.modality = "text"
        if args.model_path is None:
            args.model_path = TEXT_WEIGHTS_PATH
        txt_model, tokenizer, _ = load_model(device, args)
        parameters["Text Model Path"] = args.model_path
        parameters["Fusion Mode"] = args.late_fusion_mode
        args.model_path = IMAGE_WEIGHTS_PATH
        args.modality = "image"
        img_model, _, processor = load_model(device, args)
        parameters["Image Model Path"] = args.model_path
        args.modality = "late-fusion"
    else:
        if args.model_path is None:
            if args.modality == "text":
                args.model_path = TEXT_WEIGHTS_PATH
            elif args.modality == "image":
                args.model_path = IMAGE_WEIGHTS_PATH
        model, tokenizer, processor = load_model(device, args)
    
    # Other parameters saved in the parameters dictionary
    parameters["Model Path"] = args.model_path
    parameters["Merge Tokens"] = args.merge_tokens
    parameters["LoRA Alpha"] = args.lora_alpha
    parameters["LoRA R"] = args.lora_r
    parameters["LoRA Dropout"] = args.lora_dropout
    parameters["Use LoRA"] = args.use_lora
    parameters["Set Params"] = args.set_params

    # Threshold computation if a threshold is not provided
    if args.threshold is None:
        if args.modality == "late-fusion":
            thr = compute_threshold(txt_model, processor, tokenizer, device, args, img_model, args.late_fusion_mode)
        else:
            thr = compute_threshold(model, processor, tokenizer, device, args)
    else:
        thr = args.threshold
    
    parameters["Threshold"] = thr

    # Select dataset class and load function dynamically
    dataset_class = dataset_classes[args.dataset]
    load_func = load_functions[args.dataset]

    # Results dir setup
    output_dir = os.path.join(args.results_path, "clean", args.modality)
    if args.modality == "late-fusion":
        output_dir = os.path.join(output_dir, args.late_fusion_mode)
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

    # If the modality is late-fusion, we need to save a couple of data to evaluate
    if args.modality == "late-fusion":
        logits_txt_list = []
        scores_txt_list = []
        logits_img_list = []
        scores_img_list = []
    else:
        logits_list = [] # Logits [-inf, +inf]
        scores_list = [] # Scores [0, 1]
    
    # Batch evaluation
    for images, labels, texts, _, indices in tqdm(dataloader_test, desc="Evaluating", total=len(dataloader_test)):
        images = images.to(device)
        texts = texts.to(device)
        
        # Clean prediction
        with torch.no_grad():
            if args.modality == "late-fusion":
                txt_batch_scores, txt_batch_logits = txt_model(images=None,texts=texts)
                img_batch_scores, img_batch_logits = img_model(images=images,texts=None)

                logits_txt_list.append(txt_batch_logits.detach().cpu())
                scores_txt_list.append(txt_batch_scores.detach().cpu())
                logits_img_list.append(img_batch_logits.detach().cpu())
                scores_img_list.append(img_batch_scores.detach().cpu())
            elif args.modality == "text":
                batch_scores, batch_logits = model(None, texts)
            elif args.modality == "image":
                batch_scores, batch_logits = model(images, None)
            else:
                batch_scores, batch_logits = model(images, texts)
            
            logits_list.append(batch_logits.detach().cpu())
            scores_list.append(batch_scores.detach().cpu())

        y_true_list.append(labels)
        indices_list.append(indices)
    
    if args.modality == "late-fusion":
        logits_txt = torch.cat(logits_txt_list, dim=0).numpy().squeeze()
        scores_txt = torch.cat(scores_txt_list, dim=0).numpy().squeeze()
        logits_img = torch.cat(logits_img_list, dim=0).numpy().squeeze()
        scores_img = torch.cat(scores_img_list, dim=0).numpy().squeeze()

        scores = preds_fusion(scores_txt, scores_img, args.late_fusion_mode).squeeze()
    else:
        logits = torch.cat(logits_list, dim=0).numpy().squeeze()
        scores = torch.cat(scores_list, dim=0).numpy().squeeze()
    
    # Concatenation of true labels and indices of the samples in the original dataset
    y_true = torch.cat(y_true_list, dim=0).numpy()
    indices = torch.cat(indices_list, dim=0).numpy()

    # Model predictions 0 V 1
    y_preds = (scores > thr).astype(int).squeeze()

    # ---------- RESULTS ----------
    # Save results
    data = {
            'index': indices,
            'label': y_true,
            'pred': y_preds,
            'score': scores
        }

    # The late-fusion modality need two separate scores and logits for every sample
    if args.modality == "late-fusion":
        data.update({
            'score_text': scores_txt,
            'score_image': scores_img,
            'logit_text': logits_txt,
            'logit_image': logits_img
        })
    else:
        data.update({'logit': logits})
    
    # In case of late-fusion, plotting the logits of the text and image models to inspect their behavior
    if args.modality == "late-fusion":
        plot_text_vs_image(
            y_true,
            logits_txt,
            logits_img,
            os.path.join(output_dir, "text_vs_image_clean.png"),
            "CLEAN"
        )

    # Saving results in a csv file
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    
    # Save "Parameters" in a file
    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
        json.dump(parameters, f, indent=4)


if __name__ == "__main__":
    main()
