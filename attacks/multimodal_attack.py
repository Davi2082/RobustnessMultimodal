from pyexpat import model
import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torchattacks
import json
import glob
import argparse
from transformers import BertTokenizer, BertForMaskedLM

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
    save_perturbed_texts,
)
from configuration import (
    SOURCE_LABEL,
    TARGET_LABEL,
    ALTERNATION_ROUNDS,
    PGD_ITERS,
    EPSILON,
    ALPHA_FACTOR,
    K_BERT_ATTACK,
    THRESHOLD_PRED_SCORE,
    MAX_WORDS_TO_ATTACK,
    MAX_CANDIDATES_PER_WORD,
    MAX_WORDS_FOR_IMPORTANCE,
    MIN_TXT_SIMILARITY,
    DEVICE,
    DEVICE_MLM,
    SUBSET_SIZE,
)
from paths import RESULT_PATH, CLEAN_FF_PARAMS, DATA_PERTURBED_FF
import my_datasets

# Main evaluation function
def main():
    dataset_classes, load_functions = load_available_datasets()
    # "Parameters" contains information about the model that would be attacked
    parameters_path = CLEAN_FF_PARAMS
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
    parser.add_argument("--alternation_rounds", type=int, default=ALTERNATION_ROUNDS)
    parser.add_argument("--pgd_iters", type=int, default=PGD_ITERS)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--alpha_factor", type=float, default=ALPHA_FACTOR)
    parser.add_argument("--results_path", type=str, default=RESULT_PATH)
    parser.add_argument("--k", type=int, default=K_BERT_ATTACK)
    parser.add_argument("--threshold_pred_score", type=bool, default=THRESHOLD_PRED_SCORE)
    parser.add_argument("--max_words_to_attack", type=int, default=MAX_WORDS_TO_ATTACK)
    parser.add_argument("--max_candidates_per_word", type=int, default=MAX_CANDIDATES_PER_WORD)
    parser.add_argument("--max_words_for_importance", type=int, default=MAX_WORDS_FOR_IMPORTANCE)
    parser.add_argument("--min_txt_similarity", type=float, default=MIN_TXT_SIMILARITY)
    args = parser.parse_args()

    # Device setting
    device = torch.device(DEVICE)
    device_mlm = torch.device(DEVICE_MLM)

    # Model with relative tokenizer and processor loading
    model, tokenizer, processor = load_model(device, args, args.model_path)

    # Load BERT model and tokenizer for text corruption
    bertattack_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    bertattack_mlm = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device_mlm)
    bertattack_mlm.eval()

    # Select dataset class and load function dynamically
    dataset_class = dataset_classes[args.dataset]
    load_func = load_functions[args.dataset]

    # Results dir setup
    output_dir = os.path.join(args.results_path, "perturbed", "feature-fusion")
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

    # Lists that contain logits and scores with every modality perturbed
    logits_list = [] # Logits [-inf, +inf]
    scores_list = [] # Scores [0, 1]

    # Lists that contain logits and scores with only text perturbed
    logits_txts_per_list = []
    scores_txts_per_list = []

    # Lists that contain logits and scores with only image perturbed
    logits_imgs_per_list = []
    scores_imgs_per_list = []

    perturbed_text_rows = [] # (index, original, perturbed) for qualitative analysis

    for images, labels, texts, imgs_path, indices in tqdm(dataloader_test, desc="Evaluating batches", total=len(dataloader_test)):
        images = images.to(device)
        texts = texts.to(device)

        txts_per_list = [] # Perturbed texts
        imgs_per_list = [] # Perturbed images
        
        # Challenging the model
        for i, label in tqdm(enumerate(labels.tolist()), desc="Attacking samples of the batch", total=len(labels), leave=False):
            # Clean news
            news = {
                "txt": dataset_test.texts[indices[i].item()],
                "img": Image.open(os.path.join(dataset_test.img_dir, dataset_test.imgs_path[indices[i].item()])).convert("RGB"),
            }
            # Only consider correctly classified samples
            if label == args.source_label:
                # news_per is the single running perturbed sample: it starts clean and
                # accumulates the image and text perturbations across the alternation rounds
                news_per = news
                for _ in range(args.alternation_rounds):
                    # Image perturbation on the current version of the sample
                    news_img_per, _, _ = img_perturbation(model, tokenizer, processor, args, news_per, torch.tensor([label], device=device))
                    news_per = {"txt": news_per["txt"], "img": news_img_per["img"]}

                    # Text perturbation on the current version of the sample
                    with torch.no_grad():
                        news_txt_per, txt_similarity = bertattack(model, tokenizer, processor, args, news_per, label, device, bertattack_tokenizer, bertattack_mlm, device_mlm)
                    torch.cuda.empty_cache()

                    # Only keep the text perturbation if it stays semantically similar enough
                    if txt_similarity >= args.min_txt_similarity:
                        news_per = {"txt": news_txt_per["txt"], "img": news_per["img"]}
                # Dump the perturbed image + text for qualitative analysis
                save_perturbed_image(os.path.join(DATA_PERTURBED_FF, "images"), indices[i].item(), news_per["img"])
                perturbed_text_rows.append({
                    "index": indices[i].item(),
                    "original": news["txt"],
                    "perturbed": news_per["txt"],
                })
            else:
                # Correctly-classified Real sample: keep it unperturbed
                news_per = news
            # Adding the new (perturbed or clean) sample to the batch lists
            txts_per_list.append(news_per["txt"])
            imgs_per_list.append(news_per["img"])

        # Tokenization of multimodal corrupted texts
        txts_per_list = tokenizer(txts_per_list, return_tensors="pt", padding="max_length", truncation=True, return_attention_mask=False, max_length=args.n_tokens).to(device)
        txts_per = {"input_ids": txts_per_list.input_ids.unsqueeze(1)}

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
            batch_scores, batch_logits = model(imgs_per, txts_per)
            logits_list.append(batch_logits.detach().cpu())
            scores_list.append(batch_scores.detach().cpu())

            # Evaluation with perturbed samples only in the text channel
            batch_txts_per_scores, batch_txts_per_logits = model(images, txts_per)
            logits_txts_per_list.append(batch_txts_per_logits.detach().cpu())
            scores_txts_per_list.append(batch_txts_per_scores.detach().cpu())

            # Evaluation with perturbed samples only in the image channel
            batch_imgs_per_scores, batch_imgs_per_logits = model(imgs_per, texts)
            logits_imgs_per_list.append(batch_imgs_per_logits.detach().cpu())
            scores_imgs_per_list.append(batch_imgs_per_scores.detach().cpu())

        y_true_list.append(labels)
        indices_list.append(indices)
    
    logits = torch.cat(logits_list, dim=0).numpy().squeeze()
    scores = torch.cat(scores_list, dim=0).numpy().squeeze()

    txts_per_logits = torch.cat(logits_txts_per_list, dim=0).numpy().squeeze()
    txts_per_scores = torch.cat(scores_txts_per_list, dim=0).numpy().squeeze()

    imgs_per_logits = torch.cat(logits_imgs_per_list, dim=0).numpy().squeeze()
    imgs_per_scores = torch.cat(scores_imgs_per_list, dim=0).numpy().squeeze()

    y_true = torch.cat(y_true_list, dim=0).numpy()
    indices = torch.cat(indices_list, dim=0).numpy()

    # Model predictions 0 V 1
    y_preds = (scores > args.threshold).astype(int).squeeze() # Predictions with every channel perturbed
    y_txts_per_preds = (txts_per_scores > args.threshold).astype(int).squeeze() # Predictions with only text perturbed
    y_imgs_per_preds = (imgs_per_scores > args.threshold).astype(int).squeeze() # Predictions with only image perturbed

    # ---------- RESULTS ----------
    # Save results
    save_predictions(y_true, y_preds, scores, logits, indices, os.path.join(output_dir, "perturbed_results.csv"))
    save_predictions(y_true, y_txts_per_preds, txts_per_scores, txts_per_logits, indices, os.path.join(output_dir, "txts_perturbed_results.csv"))
    save_predictions(y_true, y_imgs_per_preds, imgs_per_scores, imgs_per_logits, indices, os.path.join(output_dir, "imgs_perturbed_results.csv"))

    # Dump perturbed texts for qualitative analysis (images already dumped during the loop)
    save_perturbed_texts(DATA_PERTURBED_FF, perturbed_text_rows)

    # Save "Parameters" in a file
    attack_parameters = {
        "Source Label": args.source_label,
        "Target Label": args.target_label,
        "Alternation Rounds for the MM Attack Optimization": args.alternation_rounds,
        "PGD Iters": args.pgd_iters,
        "Epsilon": args.epsilon,
        "Alpha Factor": args.alpha_factor,
        "K (BERT Attack)": args.k,
        "Threshold Pred Score": args.threshold_pred_score
    }
    parameters = {
        "Model Parameters": parameters,
        "Attack Parameters": attack_parameters
    }
    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
        json.dump(parameters, f, indent=4)


if __name__ == "__main__":
    main()
