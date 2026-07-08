"""Loads the unimodal *logit* CSVs and calls the plotting helper in utils.py.
One panel per fusion rule (min / max / mean), each with the fused-score
background, the decision boundary (mapped to the logit origin), and the
real / fake / text- / image- / both-attacked logits.
"""
import os
import pandas as pd

from utils import plot_score_space_fig7
from configuration import THRESHOLD, SOURCE_LABEL

RESULT = "results/Recovery/classification_results"
OUT_DIR = "figures/classification_results/scatter"


def _logits(path):
    return pd.read_csv(path)[["index", "label", "logit"]]


def main():
    txt_c = _logits(f"{RESULT}/clean/text/results.csv").rename(columns={"logit": "txt_clean"})
    img_c = _logits(f"{RESULT}/clean/image/results.csv").rename(columns={"logit": "img_clean"})
    txt_p = _logits(f"{RESULT}/perturbed/text/perturbed_results.csv").rename(columns={"logit": "txt_pert"})
    img_p = _logits(f"{RESULT}/perturbed/image/perturbed_results.csv").rename(columns={"logit": "img_pert"})

    df = (txt_c.merge(img_c[["index", "img_clean"]], on="index")
                .merge(txt_p[["index", "txt_pert"]], on="index")
                .merge(img_p[["index", "img_pert"]], on="index"))

    os.makedirs(OUT_DIR, exist_ok=True)
    for fusion in ("min", "max", "mean"):
        out = os.path.join(OUT_DIR, f"score_space_{fusion}.png")
        plot_score_space_fig7(
            df["label"].values,
            df["txt_clean"].values, df["img_clean"].values,
            df["txt_pert"].values, df["img_pert"].values,
            out, fusion=fusion, threshold=THRESHOLD, source_label=SOURCE_LABEL,
        )
        print(f"saved: {out}")



if __name__ == "__main__":
    main()
