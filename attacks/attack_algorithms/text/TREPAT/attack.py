"""TREPAT on the text channel.

The victim wrapper turns any classifier into the probability oracle TREPAT
queries; the attacker searches LLM-generated rewritings under a query budget.
"""

import numpy as np
import torch
from PIL import Image
from sentence_transformers import util

from attacks.attack_algorithms.text.TREPAT.modifier import Modifier
from attacks.attack_algorithms.text.common import model_sbert, visible_text_window
from configuration_files.configuration import MAX_VARIANTS


class TargetedTrepatAttacker:
    def __init__(self, modifier, source_label, target_label):
        self.modifier = modifier
        self.source_label = source_label
        self.target_label = target_label

    def attack(self, victim, input_text):
        pred_old = victim.get_pred([input_text])[0]

        if pred_old != self.source_label:
            return None

        old_target_prob = victim.get_prob([input_text])[0, self.target_label]

        self.modifier.init(input_text)

        best_text = input_text
        best_target_prob = old_target_prob

        while True:
            x_new = self.modifier.get_next_variant()
            if x_new is None:
                break

            probs = victim.get_prob([x_new])[0]
            target_prob = probs[self.target_label]

            gain = target_prob - old_target_prob
            self.modifier.get_feedback(gain)

            if target_prob > best_target_prob:
                best_target_prob = target_prob
                best_text = x_new

            pred_new = victim.get_pred([x_new])[0]
            if pred_new == self.target_label:
                return x_new

        return best_text


class TrepatThemisVictim:
    def __init__(self, model, tokenizer, processor, args, device, image=None):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.args = args
        self.device = device
        self.image = image
        self._prob_cache = {}

    def _use_image(self):
            return self.args.modality in ["feature-fusion", "intermediate-fusion"]
    def _build_text_input(self, texts):
        tokenized = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, return_attention_mask=False, max_length=self.args.n_tokens,).to(self.device)
        return {"input_ids": tokenized.input_ids.unsqueeze(1)}

    def _build_image_input(self, batch_size):
        if not self._use_image():
            return None

        if self.image is None:
            return None

        # If the image is alread a tensor, we assume it's already processed and just move it to the correct device
        if torch.is_tensor(self.image):
            img = self.image.to(self.device)

            # [C, H, W] -> [1, C, H, W]
            if img.dim() == 3:
                img = img.unsqueeze(0)

            # [1, C, H, W] -> [batch_size, C, H, W]
            if img.shape[0] == 1 and batch_size > 1:
                img = img.repeat(batch_size, 1, 1, 1)

            return img

        # If the image is a PIL image, we process it using the processor
        processed = self.processor(images=[self.image] * batch_size, return_tensors="pt", do_normalize=False)

        return processed["pixel_values"].to(self.device)

    def _scores_from_texts(self, texts):
        txt_input = self._build_text_input(texts)
        img_input = self._build_image_input(batch_size=len(texts))

        with torch.no_grad():
            scores, logits = self.model(img_input, txt_input)

        return scores.detach().cpu().numpy().reshape(-1)

    def get_prob(self, input_):
        missing_texts = [
            text for text in input_
            if text not in self._prob_cache
        ]

        if missing_texts:
            scores = self._scores_from_texts(missing_texts)

            for text, score in zip(missing_texts, scores):
                self._prob_cache[text] = np.array(
                    [1.0 - score, score],
                    dtype=np.float32,
                )

        return np.stack(
            [self._prob_cache[text] for text in input_],
            axis=0,
        )

    def get_pred(self, input_):
        probs = self.get_prob(input_)
        return (probs[:, 1] > self.args.threshold).astype(int)


def trepat_attack(model, themis_tokenizer, processor, args, news, label, device, rephraser,
                  source_label=None, target_label=None):
    """TREPAT text attack.

    Pass ``source_label=label``, ``target_label=1 - label`` for an untargeted
    attack; otherwise the configured Fake->Real pair is used.
    """
    if source_label is None:
        source_label = args.source_label
    if target_label is None:
        target_label = args.target_label

    visible_txt, hidden_txt = visible_text_window(news["txt"], themis_tokenizer, args.n_tokens)
    visible_news = {"txt": visible_txt, "img": news["img"]}

    victim = TrepatThemisVictim(model, themis_tokenizer, processor, args, device, image=news["img"])
    modifier = Modifier(rephraser, splitter="cascade", weak=False, max_variants=MAX_VARIANTS)
    attacker = TargetedTrepatAttacker(modifier, source_label, target_label)

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
