"""Helpers shared by every text attack."""

import ollama
import torch
from sentence_transformers import SentenceTransformer, util

# One shared encoder: every text attack reports semantic similarity to the
# original text with the same measure.
model_sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


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
