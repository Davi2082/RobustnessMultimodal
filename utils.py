import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torchattacks
import torchvision.transforms as T
import ollama
import json
import glob
import gc

from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from torchmetrics.image import StructuralSimilarityIndexMeasure
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# Custom modules
import my_datasets

from themis_model import get_Themis
from bertattack import attack, Feature
from configuration import SOURCE_LABEL, TARGET_LABEL, FF_WEIGHTS_PATH, FF_NAME_IMG_EMBED
from paths import ROC_SETS_DIR, ROC_PLOTS_DIR

# Utilities for logging
def info(msg):
    print(f"\033[32m{msg}\033[0m")
def warning(msg):
    print(f"\033[33m{msg}\033[0m")
def error(msg):
    print(f"\033[31m{msg}\033[0m")

def cleanup_cuda(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_available_datasets():
    # Get available datasets from Data directory
    available_datasets = [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]
    # Create mappings dynamically
    dataset_classes = {}
    load_functions = {}
    for dataset in available_datasets:
        class_name = f"{dataset}_Dataset"
        load_name = f"{dataset.lower()}_load_annotations_file"
        try:
            dataset_classes[dataset] = getattr(my_datasets, class_name)
            load_functions[dataset] = getattr(my_datasets, load_name)
        except AttributeError:
            print(f"Warning: Class {class_name} or function {load_name} not found in my_datasets.py for dataset {dataset}")
    return dataset_classes, load_functions

# The wrapped model for PGD attack, to get only one modality perturbed
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

class BertAttackThemisWrapper(torch.nn.Module):
    def __init__(self, themis_model, themis_tokenizer, processor, fixed_image, args, device, bert_tokenizer):
        super().__init__()
        self.themis_model = themis_model
        self.themis_tokenizer = themis_tokenizer
        self.processor = processor
        self.args = args
        self.device = device
        self.bert_tokenizer = bert_tokenizer

        # immagine fissata
        if isinstance(fixed_image, Image.Image):
            processed = processor(images=fixed_image, return_tensors="pt")
            pixel_values = processed["pixel_values"].to(device)
            if pixel_values.dim() == 4:
                pixel_values = pixel_values.unsqueeze(1)
            self.fixed_images = {"pixel_values": pixel_values}
        else:
            pixel_values = fixed_image.to(device)
            if pixel_values.dim() == 4:
                pixel_values = pixel_values.unsqueeze(1)
            self.fixed_images = {"pixel_values": pixel_values}

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # batch di testi BERT -> stringhe
        texts = self.bert_tokenizer.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # ritokenizzazione con tokenizer di Themis
        themis_tokens = self.themis_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            max_length=self.args.n_tokens,
        )

        themis_tokens = {k: v.to(self.device) for k, v in themis_tokens.items()}

        if self.args.modality == "feature-fusion" or self.args.modality == "intermediate-fusion":
            # repeat immagine se batch > 1
            pixel_values = self.fixed_images["pixel_values"]
            if pixel_values.size(0) == 1 and len(texts) > 1:
                pixel_values = pixel_values.repeat(len(texts), 1, 1, 1, 1)
            elif pixel_values.size(0) != len(texts):
                raise ValueError(
                    f"Image batch mismatch: image batch={pixel_values.size(0)}, text batch={len(texts)}"
                )
            images = {"pixel_values": pixel_values}
            outputs, _ = self.themis_model(images, themis_tokens)  # [B,1] o [B]
            del images
        else:
            outputs, _ = self.themis_model(None, themis_tokens)  # [B,1] o [B]

        del themis_tokens
        
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(1)

        # output sigmoidato in [0,1] -> logits fake 2-class
        # classe 0 = 1 - p, classe 1 = p
        logits = torch.cat((1 - outputs, outputs), dim=1)

        # compatibilità con HuggingFace style: model(...)[0]
        return (logits,)

class BertAttackTextOnlyWrapper(torch.nn.Module):
    def __init__(self, text_model, themis_tokenizer, args, device, bert_tokenizer):
        super().__init__()
        self.text_model = text_model
        self.themis_tokenizer = themis_tokenizer
        self.args = args
        self.device = device
        self.bert_tokenizer = bert_tokenizer

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        texts = self.bert_tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        themis_tokens = self.themis_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            max_length=self.args.n_tokens,
        )
        themis_tokens = {k: v.to(self.device) for k, v in themis_tokens.items()}

        # with torch.inference_mode():
        outputs, _ = self.text_model(images=None, texts=themis_tokens)
        del themis_tokens
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(1)
        logits = torch.cat((1 - outputs, outputs), dim=1)
        return (logits,)

def use_model(model, tokenizer, processor, args, news, thr, modality=None):
    device = next(model.parameters()).device
    # Text tokenization
    token_txt = tokenizer(
        news["txt"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_attention_mask=False,
        max_length=args.n_tokens,
    ).to(device)
    # Image processing
    if isinstance(news["img"], Image.Image):
        process_img = processor(images=news["img"], return_tensors="pt").to(device)
        process_img["pixel_values"] = process_img["pixel_values"].unsqueeze(1)
    else:
        process_img = {"pixel_values": news["img"]}
        if process_img["pixel_values"].dim() == 4:
            process_img["pixel_values"] = process_img["pixel_values"].unsqueeze(1)

    # Using the model
    with torch.no_grad():
        if modality == "text":
            output, logits = model(images=None, texts=token_txt)
        elif modality == "image":
            output, logits = model(images=process_img, texts=None)
        else:
            output, logits = model(process_img, token_txt)
        preds = [1 if i > thr else 0 for i in output.cpu().detach().numpy()]

    return preds, output, logits

def save_img(img, save_path):
    to_pil = T.ToPILImage()
    to_pil(img.squeeze(0).cpu().float()).save(save_path)

def img_perturbation(model, tokenizer, processor, args, news, label):
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
    attack = torchattacks.PGD(wrapped_model, eps=args.epsilon, alpha=alpha, steps=args.pgd_iters, random_start=True)
    corr_img = attack(process_img["pixel_values"], label)

    # Compute SSIM before converting back to PIL
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_val = ssim(preds=corr_img.float(), target=process_img["pixel_values"].float())

    # Convert perturbed tensor back to PIL Image so downstream processor can handle it uniformly
    arr = (corr_img.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    corr_news = {"txt": news["txt"], "img": Image.fromarray(arr)}

    return corr_news, ssim_val, process_img["pixel_values"]


model_sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
def bertattack(
    model,
    themis_tokenizer,
    processor,
    args,
    news,
    label,
    device,
    bert_tokenizer,
    mlm_model,
    mlm_device,
    use_bpe=0,
):
    feat = Feature(news["txt"], int(label))

    tgt_model = BertAttackThemisWrapper(
        themis_model=model,
        themis_tokenizer=themis_tokenizer,
        processor=processor,
        fixed_image=news["img"],
        args=args,
        device=device,
        bert_tokenizer=bert_tokenizer,
    )

    attacked_feat = attack(
        feature=feat,
        tgt_model=tgt_model,
        mlm_model=mlm_model,
        tokenizer=bert_tokenizer,
        k=args.k,
        batch_size=args.batch_size,
        max_length=min(args.n_tokens, 512),
        cos_mat=None,
        w2i={},
        i2w={},
        use_bpe=use_bpe,
        threshold_pred_score=args.threshold_pred_score,
        target_device=device,
        mlm_device=mlm_device
    )

    corr_txt = attacked_feat.final_adverse
    corr_news = {"txt": corr_txt, "img": news["img"]}

    with torch.no_grad():
        emb_original = model_sbert.encode(news["txt"], convert_to_tensor=True, device="cpu")
        emb_corr = model_sbert.encode(corr_txt, convert_to_tensor=True, device="cpu")
        txt_similarity = util.cos_sim(emb_original, emb_corr).item()

    cleanup_cuda(tgt_model, attacked_feat, feat, emb_original, emb_corr)
    
    return corr_news, txt_similarity

def bertattack_text_only(
    model,
    themis_tokenizer,
    args,
    dataset,
    indices,
    labels,
    device,
    bert_tokenizer,
    mlm_model,
    mlm_device,
    min_similarity=0.5,
):
    corr_txts = []
    similarities = []

    indices = indices.tolist() if torch.is_tensor(indices) else list(indices)
    labels = labels.detach().cpu().tolist() if torch.is_tensor(labels) else list(labels)

    for idx, label in zip(indices, labels):
        original_txt = dataset.texts[idx]

        corr_txt, txt_similarity = bertattack_text_only_single(
            model=model,
            themis_tokenizer=themis_tokenizer,
            args=args,
            txt=original_txt,
            label=label,
            device=device,
            bert_tokenizer=bert_tokenizer,
            mlm_model=mlm_model,
            mlm_device=mlm_device,
        )

        if txt_similarity < min_similarity:
            corr_txt = original_txt
            txt_similarity = 1.0

        corr_txts.append(corr_txt)
        similarities.append(txt_similarity)

        cleanup_cuda()

    tokenized_corr_txts = themis_tokenizer(
        corr_txts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_attention_mask=False,
        max_length=args.n_tokens,
    )

    # IMPORTANTISSIMO: resta su CPU
    tokenized_corr_txts = {
        "input_ids": tokenized_corr_txts.input_ids.unsqueeze(1)
    }

    return tokenized_corr_txts, similarities

def bertattack_text_only_single(
    model,
    themis_tokenizer,
    args,
    txt,
    label,
    device,
    bert_tokenizer,
    mlm_model,
    mlm_device,
    use_bpe=0,
):
    feat = Feature(txt, int(label))

    tgt_model = BertAttackTextOnlyWrapper(
        text_model=model,
        themis_tokenizer=themis_tokenizer,
        args=args,
        device=device,
        bert_tokenizer=bert_tokenizer,
    )

    try:
        attacked_feat = attack(
            feature=feat,
            tgt_model=tgt_model,
            mlm_model=mlm_model,
            tokenizer=bert_tokenizer,
            k=args.k,
            batch_size=args.batch_size,
            max_length=min(args.n_tokens, 512),
            cos_mat=None,
            w2i={},
            i2w={},
            use_bpe=use_bpe,
            threshold_pred_score=args.threshold_pred_score,
            target_device=device,
            mlm_device=mlm_device,
            max_words_to_attack=args.max_words_to_attack,
            max_candidates_per_word=args.max_candidates_per_word,
            max_words_for_importance=args.max_words_for_importance
        )

        corr_txt = str(attacked_feat.final_adverse)

        with torch.no_grad():
            emb_original = model_sbert.encode(txt, convert_to_tensor=True, device="cpu")
            emb_corr = model_sbert.encode(corr_txt, convert_to_tensor=True, device="cpu")
            txt_similarity = util.cos_sim(emb_original, emb_corr).item()

        cleanup_cuda(emb_original, emb_corr)

        return corr_txt, txt_similarity

    finally:
        cleanup_cuda(tgt_model, feat)
        if "attacked_feat" in locals():
            cleanup_cuda(attacked_feat)

def txt_corruption(news):
    # LLM-based text corruption
    client = ollama.Client(host="http://127.0.0.1:11435")
    user_content = f"News article:\n{news['txt']}"
    response = client.chat(
        model="qwen2.5:14b-instruct",
        options={"temperature": 0.5, "max_tokens": 2048},
        messages=[
            {"role": "system", "content": LLM_CORRUPTER_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )
    corr_txt = response["message"]["content"].strip()
    corr_news = {"txt": corr_txt, "img": news["img"]}
    
    # Compute text similarity
    txt_similarity = util.cos_sim(model_sbert.encode(news["txt"], convert_to_tensor=True), model_sbert.encode(corr_txt, convert_to_tensor=True)).item()

    return corr_news, txt_similarity


def save_results(
    output_dir,
    idx,
    img_path,
    news,
    clean_img,
    img_corr_news_pgd,
    multimodal_corr_news,
    preds,
    ssim_pgd,
    txt_similarity,
):
    # Create result directory
    result_dir = os.path.join(output_dir, str(idx))
    os.makedirs(result_dir, exist_ok=True)
    # Save images, clean and corrupted
    save_img(clean_img, os.path.join(result_dir, "clean_img.png"))
    save_img(img_corr_news_pgd["img"], os.path.join(result_dir, "corr_img.png"))

    # Save result data
    result_data = {
        "news": int(idx),
        "img": img_path,
        "label": int(preds["label"]),
        "output_clean": float(preds["output_clean"]),
        "threshold": float(preds["thr_multimodal_cross"]),
        "pred_clean": int(preds["pred_clean"]),
        "output_txt_corr": float(preds["output_txt_corr"]),
        "pred_txt_corr": int(preds["pred_txt_corr"]),
        "output_img_corr": float(preds["output_img_corr"]),
        "pred_img_corr": int(preds["pred_img_corr"]),
        "output_multimodal_corr": float(preds["output_multimodal_corr"]),
        "pred_multimodal_corr": int(preds["pred_multimodal_corr"]),
        "SSIM": round(float(ssim_pgd.item() if hasattr(ssim_pgd, "item") else ssim_pgd), 3),
        "txt_similarity": round(float(txt_similarity.item() if hasattr(txt_similarity, "item") else txt_similarity), 3),
        "original_txt": news["txt"],
        "corr_txt": multimodal_corr_news["txt"],
    }
    with open(os.path.join(result_dir, "result.json"), "w") as f:
        json.dump(result_data, f, indent=4)

def save_predictions(y_true, y_preds, scores, logits, indices, output_dir, ssims=None, txt_similarities=None):
    data = {
            'index': indices,
            'label': y_true,
            'pred': y_preds,
            'score': scores,
            'logit': logits,
        }
    
    if ssims is not None:
        data.update({
            'ssim': ssims
        })
    if txt_similarities is not None:
        data.update({
            'text similarity': txt_similarities
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_dir, index=False)


def save_perturbed_image(img_dir, index, pil_img):
    """Dump a single perturbed PIL image as <index>.png for qualitative analysis."""
    os.makedirs(img_dir, exist_ok=True)
    pil_img.save(os.path.join(img_dir, f"{index}.png"))


def save_perturbed_texts(out_dir, rows):
    """Write collected perturbed texts (list of {index, original, perturbed}) to texts.csv."""
    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).sort_values("index").to_csv(os.path.join(out_dir, "texts.csv"), index=False)


# -----------------------
# Model loading
# -----------------------

def load_model(device, args, correct_model_path=None):
    if args.modality == "feature-fusion" or args.modality == "intermediate-fusion":
        args.name_img_embed = FF_NAME_IMG_EMBED
        args.model_path = FF_WEIGHTS_PATH

    if args.merge_tokens == 0:
        args.merge_tokens = None
    if args.set_params:
        p = args.model_path.split("\\")[-1].split("_")
        args.lora_alpha = int(p[2])
        args.lora_r = int(p[3])
        args.lora_dropout = float(p[4])
        args.use_lora = True if "True" in p[5] else False

    model, tokenizer, processor = get_Themis(
        name_llm=args.name_llm,
        name_img_embed=args.name_img_embed,
        use_lora=args.use_lora,
        is_pythia=True if "pythia" in args.name_llm else False,
        lora_alpha=args.lora_alpha,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        merge_tokens=args.merge_tokens,
        device=device
    )

    if os.path.exists(args.model_path):
        try:
            # when a serious GPU will be available change map_location to device
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        except Exception:
            error("Error loading weights, it will be used random weights. The results will be meaningless.")
    else:
        warning("Warning: weights file not found, using random weights. The results will be meaningless.")

    model.to(device).eval()

    return model, tokenizer, processor


# -----------------------
# Metrics & reporting
# -----------------------

def compute_metrics(y_true, y_pred, scores):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    conf_matr = confusion_matrix(y_true, y_pred)
    
    # AUC on on samples
    fpr, tpr, thr = roc_curve(y_true, scores)
    auc_score = auc(fpr, tpr)
    return {
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "macro_f1": round(macro_f1, 3),
        "auc": auc_score,
    }, fpr, tpr, conf_matr


def compute_robustness_metrics(
    y_true,
    y_clean,
    y_corr,
    ASYMMETRIC_ATTACK=True,
    source_label=SOURCE_LABEL,
    target_label=TARGET_LABEL,
):
    # y_true = 1-np.asarray(y_true)
    # y_clean = 1-np.asarray(y_clean)
    # y_corr = 1-np.asarray(y_corr)

    y_true = [abs(i-1) for i in y_true]

    # Accuracy on corrupted input (global)
    accuracy_on_corrupted = accuracy_score(y_true, y_corr)
    # Accuracy on clean
    accuracy_on_clean = accuracy_score(y_true, y_clean)
    # Delta Accuracy (global)
    delta_acc = accuracy_on_clean - accuracy_on_corrupted
    # Flip Rate (global)
    flip_rate = np.sum(y_clean != y_corr) / len(y_clean)
    # ASR
    # Asymmetry to be removed (We are considering attacking on l)
    is_attackable = (y_true == source_label) & (y_clean == source_label)
    total_attackable = np.sum(is_attackable)
    if total_attackable == 0:
        asr = 0.0
    else:
        is_successful_attack = is_attackable & (y_corr == target_label)
        successful_attacks = np.sum(is_successful_attack)
        asr = successful_attacks / total_attackable
    return {
        "accuracy_on_clean": round(accuracy_on_clean, 3),
        "accuracy_on_corrupted": round(accuracy_on_corrupted, 3),
        "delta_accuracy": round(delta_acc, 3),
        "flip_rate": round(flip_rate, 3),
        "attack_success_rate": round(asr, 3),
    }


def plot_confusion_matrix(cm, labels, out_file):
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues"
    )
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def plot_text_vs_image(y_true, logits_text, logits_image, out_file, title):
    logits_text = np.asarray(logits_text).reshape(-1)
    logits_image = np.asarray(logits_image).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    # mask_fake = 1-y_true == 1
    # mask_real = 1-y_true == 0
    
    mask_fake = y_true == 0
    mask_real = y_true == 1

    plt.figure(figsize=(10, 10))

    # Fake = cerchio vuoto rosso
    plt.scatter(
        # 1-logits_text[mask_fake],
        # 1-logits_image[mask_fake],
        logits_text[mask_fake],
        logits_image[mask_fake],
        c='red',
        label='Fake (label=0)',
        # label='Fake (label=1)',
        alpha=0.8
    )

    # Real = cerchio pieno blu
    plt.scatter(
        # 1-logits_text[mask_real],
        # 1-logits_image[mask_real],
        logits_text[mask_real],
        logits_image[mask_real],
        c='blue',
        label='Real (label=1)',
        # label='Real (label=0)',
        alpha=0.8
    )

    plt.axvline(0, linestyle='--')
    plt.axhline(0, linestyle='--')

    plt.xlabel("Logit testo")
    plt.ylabel("Logit immagine")
    plt.title(f"[{title.upper()}]Scatter logit: testo vs immagine")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def plot_shift_arrows(
    y_true,
    logit_text_clean,
    logit_image_clean,
    logit_text_corr,
    logit_image_corr,
    out_file
):
    # x0 = 1-np.asarray(logit_text_clean).reshape(-1)
    # y0 = 1-np.asarray(logit_image_clean).reshape(-1)
    # x1 = 1-np.asarray(logit_text_corr).reshape(-1)
    # y1 = 1-np.asarray(logit_image_corr).reshape(-1)
    x0 = np.asarray(logit_text_clean).reshape(-1)
    y0 = np.asarray(logit_image_clean).reshape(-1)
    x1 = np.asarray(logit_text_corr).reshape(-1)
    y1 = np.asarray(logit_image_corr).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    assert len(x0) == len(y0) == len(x1) == len(y1) == len(y_true), \
        f"Length mismatch: x0={len(x0)}, y0={len(y0)}, x1={len(x1)}, y1={len(y1)}, y_true={len(y_true)}"

    dx = x1 - x0
    dy = y1 - y0

    # print("DEBUG shifts:")
    # print("dx mean abs:", np.mean(np.abs(dx)))
    # print("dy mean abs:", np.mean(np.abs(dy)))
    # print("dx zero-ish:", np.mean(np.abs(dx) < 1e-6))
    # print("dy zero-ish:", np.mean(np.abs(dy) < 1e-6))

    plt.figure(figsize=(10, 10))

    # mask_fake = 1-y_true == 1
    # mask_real = 1-y_true == 0
    mask_fake = y_true == 0
    mask_real = y_true == 1

    plt.scatter(x0[mask_fake], y0[mask_fake], c="red", label="Fake clean", alpha=0.8)
    plt.scatter(x0[mask_real], y0[mask_real], c="blue", label="Real clean", alpha=0.8)

    plt.quiver(
        x0[mask_fake], y0[mask_fake],
        dx[mask_fake], dy[mask_fake],
        angles="xy", scale_units="xy", scale=1,
        color="red", alpha=0.5
    )

    plt.quiver(
        x0[mask_real], y0[mask_real],
        dx[mask_real], dy[mask_real],
        angles="xy", scale_units="xy", scale=1,
        color="blue", alpha=0.5
    )

    plt.axvline(0.5, linestyle="--")
    plt.axhline(0.5, linestyle="--")

    plt.xlabel("Score testo")
    plt.ylabel("Score immagine")
    plt.title("Spostamento dei sample dopo corruzione")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def plot_roc(rocs_info, out_file):
    plt.figure(figsize=(10, 10))

    for name, roc_data in rocs_info.items():
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        roc_auc = roc_data["roc_auc"]
        color = roc_data["color"]

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC = {roc_auc:.3f})",
            color=color
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")

    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_file)
    plt.close()

def compute_threshold(model, processor, tokenizer, device, args, model2=None, mode=None):
    # Select dataset class and load function dynamically
    dataset_class = getattr(my_dataset, f"{args.dataset}_Dataset")
    load_func = getattr(f"{args.dataset.lower()}_load_annotations_file")

    dataset_val = my_datasets.get_dataset(
        dataset_class,
        load_func,
        args.n_tokens,
        processor,
        tokenizer,
        glob.glob(f"data/{args.dataset}/val_augmented.*")[0],
        f"data/{args.dataset}/images",
    )
    
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
    )

    y, y_preds = [], []
    for images, labels, texts, _, _ in tqdm(dataloader_val, desc="Threshold computation", total=len(dataloader_val), leave=False):
        images = images.to(device)
        texts = texts.to(device)
        if modality == "multimodal":
            outputs = model(images, texts)
            y_preds.extend(outputs.to("cpu").detach().numpy())
        elif modality == "unimodal_txt":
            outputs = model(None, texts_for_txt)
            y_preds.extend(outputs.to("cpu").detach().numpy())
        elif modality == "unimodal_img":
            outputs = model(images_for_img, None)
            y_preds.extend(outputs.to("cpu").detach().numpy())
        else:
            outputs1 = model(None, texts_for_txt)
            outputs2 = model2(images_for_img, None)
            fused_outputs = preds_fusion(outputs1, outputs2, mode)
            y_preds.extend(fused_outputs)
        y.extend(labels.to("cpu").numpy())

    fpr, tpr, thr = roc_curve(y, y_preds)
    best_thr = thr[(tpr - fpr).argmax()]
    return best_thr

def preds_fusion(first_modality_outputs, second_modality_outputs, mode):
    first_modality_outputs = first_modality_outputs
    second_modality_outputs = second_modality_outputs
    if mode == "mean":
        fused_scores = (first_modality_outputs + second_modality_outputs) / 2
    elif mode == "max":
        fused_scores = np.maximum(first_modality_outputs, second_modality_outputs)
    elif mode == "min":
        fused_scores = np.minimum(first_modality_outputs, second_modality_outputs)
    else:
        raise ValueError(f"Invalid fusion mode: {mode}")
    
    return fused_scores

def create_late_fusion(text_csv, image_csv, output_dir, fusion_type, threshold=0.5, filename="perturbed_results.csv"):
    text_df = pd.read_csv(text_csv)
    image_df = pd.read_csv(image_csv)

    if len(text_df) != len(image_df):
        raise ValueError(
            f"Text and image CSV have different lengths: "
            f"{len(text_df)} != {len(image_df)}"
        )

    if not (text_df["index"].values == image_df["index"].values).all():
        raise ValueError("Text and image CSV have different sample indexes.")

    if not (text_df["label"].values == image_df["label"].values).all():
        raise ValueError("Text and image CSV have different labels.")

    if fusion_type == "mean":
        scores = (text_df["score"] + image_df["score"]) / 2

    elif fusion_type == "min":
        scores = pd.concat(
            [text_df["score"], image_df["score"]],
            axis=1,
        ).min(axis=1)

    elif fusion_type == "max":
        scores = pd.concat(
            [text_df["score"], image_df["score"]],
            axis=1,
        ).max(axis=1)

    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    preds = (scores >= threshold).astype(int)

    result_df = pd.DataFrame({
        "index": text_df["index"],
        "label": text_df["label"],

        "score": scores,
        "pred": preds,

        "text_score": text_df["score"],
        "image_score": image_df["score"],

        "text_logit": text_df["logit"],
        "image_logit": image_df["logit"],
    })

    os.makedirs(output_dir, exist_ok=True)
    result_df.to_csv(os.path.join(output_dir, filename), index=False)

    return result_df


def create_late_fusion_parameters(
    text_parameters,
    image_parameters,
    output_dir,
    fusion_type,
    scenario,
    text_state,
    image_state,
    text_csv=None,
    image_csv=None,
):
    with open(text_parameters) as f:
        txt_params = json.load(f)

    with open(image_parameters) as f:
        img_params = json.load(f)

    params = {
        "fusion_type": fusion_type,
        "scenario": scenario,

        "text_state": text_state,
        "image_state": image_state,

        "text_csv": text_csv,
        "image_csv": image_csv,

        "text_parameters_path": text_parameters,
        "image_parameters_path": image_parameters,

        "text_parameters": txt_params,
        "image_parameters": img_params,
    }

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
        json.dump(params, f, indent=2)

def build_curve_name(args):
    name = args.modality
    if args.modality == "late-fusion" or args.modality == "feature-fusion" or  args.modality == "intermediate-fusion":
        if args.mode is not None:
            name += f"-{args.mode}"
        if args.perturbation_type is not None:
            name += f"|{args.perturbation_type}"
        else:
            name += f"|{args.type}"
    return name

def update_roc_cache( roc_set, curve_name, auc, fpr, tpr):
    roc_dir = ROC_SETS_DIR
    os.makedirs(roc_dir, exist_ok=True)

    roc_file = os.path.join(roc_dir, f"{roc_set}.json")

    if os.path.exists(roc_file):
        with open(roc_file) as f:
            roc_cache = json.load(f)
    else:
        roc_cache = {}

    roc_cache[curve_name] = {
        "auc": float(auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }

    with open(roc_file, "w") as f:
        json.dump(roc_cache, f, indent=2)

    return roc_cache


def regenerate_plot(roc_cache, roc_set):
    os.makedirs(ROC_PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(10, 10))

    COLORS = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "cyan",
        "olive",
    ]

    for i, (name, data) in enumerate(roc_cache.items()):
        color = COLORS[i % len(COLORS)]

        plt.plot(data["fpr"], data["tpr"], color=color, linewidth=2, label=f"{name} (AUC={data['auc']:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random")

    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Comparison - {roc_set}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ROC_PLOTS_DIR, f"{roc_set}.png"))
    plt.close()