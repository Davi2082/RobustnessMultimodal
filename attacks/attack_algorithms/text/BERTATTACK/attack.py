"""BERT-Attack on the text channel.

The wrappers expose a Themis-style classifier through the HuggingFace-like
interface the BERT-Attack core expects; the entry points return the perturbed
sample and its semantic similarity to the original.
"""

import torch
from PIL import Image
from sentence_transformers import util

from attacks.attack_algorithms.text.BERTATTACK import bertattack as bert_attack
from attacks.attack_algorithms.text.common import model_sbert
from configuration_files.configuration import MAX_CHANGE_RATIO, USE_BPE
from utils import cleanup_cuda


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

        # Compatibility with Hugging Face style: model(...)[0]
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
    use_bpe=USE_BPE,
):
    feat = bert_attack.Feature(news["txt"], int(label))

    tgt_model = BertAttackThemisWrapper(
        themis_model=model,
        themis_tokenizer=themis_tokenizer,
        processor=processor,
        fixed_image=news["img"],
        args=args,
        device=device,
        bert_tokenizer=bert_tokenizer,
    )

    attacked_feat = bert_attack.attack(
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
        max_change_ratio=MAX_CHANGE_RATIO,
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
    feat = bert_attack.Feature(txt, int(label))

    tgt_model = BertAttackTextOnlyWrapper(
        text_model=model,
        themis_tokenizer=themis_tokenizer,
        args=args,
        device=device,
        bert_tokenizer=bert_tokenizer,
    )

    try:
        attacked_feat = bert_attack.attack(
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
