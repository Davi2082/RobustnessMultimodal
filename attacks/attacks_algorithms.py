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

def visible_text_window(text, tokenizer, n_tokens):
    """Prefix of `text` that survives the classifier's own truncation
    (max_length=n_tokens), reconstructed via decode() so it reflects exactly
    the token stream the model attends to -- tokenizer normalization (BPE
    merges, whitespace/Unicode handling) can make an offset-mapping character
    slice of the original string diverge from what decode() reconstructs.
    Edits placed after this point are silently discarded by every downstream
    tokenizer call, so the attacker should never spend budget rephrasing the
    invisible tail."""
    encoding = tokenizer(text, truncation=True, max_length=n_tokens, return_offsets_mapping=True)
    visible_txt = tokenizer.decode(encoding["input_ids"], skip_special_tokens=True)
    offsets = encoding["offset_mapping"]
    end_char = max((end for _, end in offsets), default=0)
    hidden_txt = text[end_char:]
    return visible_txt, hidden_txt

def trepat_attack(model, themis_tokenizer, processor, args, news, label, device, rephraser):
    visible_txt, hidden_txt = visible_text_window(news["txt"], themis_tokenizer, args.n_tokens)
    visible_news = {"txt": visible_txt, "img": news["img"]}

    victim = TrepatThemisVictim(model, themis_tokenizer, processor, args, device, image=news["img"])
    modifier = Modifier(rephraser, splitter="cascade", weak=False, max_variants=MAX_VARIANTS)
    attacker = TargetedTrepatAttacker(modifier, args.source_label, args.target_label)

    corr_visible = attacker.attack(victim, visible_news["txt"])

    if corr_visible is None:
        corr_visible = visible_news["txt"]

    corr_txt = corr_visible + hidden_txt

    corr_news = {
        "txt": corr_txt,
        "img": news["img"],
    }

    with torch.no_grad():
        emb_original = model_sbert.encode(visible_news["txt"], convert_to_tensor=True, device="cpu")
        emb_corr = model_sbert.encode(corr_txt, convert_to_tensor=True, device="cpu")
        txt_similarity = util.cos_sim(emb_original, emb_corr).item()

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