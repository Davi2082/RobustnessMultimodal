"""Qualitative flip examples: image, amplified PGD perturbation, original vs TREPAT text.

    RM_RAND_SEED=50 python3 scripts/plot/plot_flip_examples.py --dataset Fakeddit
"""

import argparse
import difflib
import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker

from configuration_files.configuration import ATTACK_CLASSES, RAND_SEED, attacked_mask
from configuration_files.paths import dataset_result_root

# (method, sample index, attack shown, title); only the attack shown is applied
SAMPLES = (("min", 1665, "joint", "joint flips it, PGD and TREPAT do not"),
           ("mean", 1535, "joint", "joint flips it, PGD and TREPAT do not"),
           ("linear", 1720, "pgd", "PGD flips it, union and joint do not"),
           ("max", 1332, "trepat", "TREPAT flips it, union and joint do not"))
SHOWN = {"joint": "joint", "pgd": "PGD", "trepat": "TREPAT"}
ATTACKS = (("clean", None), ("PGD", "pgd"), ("TREPAT", "trepat"), ("union", "sum"), ("joint", "interleaved"))
LABELS = {0: "Fake", 1: "Real"}
NAMES = {"feature-fusion": "feature fusion"}
AMP = 127.5  # maps the eps = 1/255 ball onto [0, 1]
HIGHLIGHT = "#e8730c"  # TREPAT colour of the other figures
WRAP = 34


def words(text, other):
    """(word, changed) pairs of `text` against `other`."""
    a, b = other.split(), text.split()
    out = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        out += [(w, op != "equal") for w in b[j1:j2]]
    return out


def text_box(pairs, size):
    """Wrapped text with changed words highlighted."""
    lines, line, n = [], [], 0
    for w, changed in pairs:
        if line and n + len(w) > WRAP:
            lines.append(line)
            line, n = [], 0
        line.append(TextArea(w, textprops=dict(size=size, color=HIGHLIGHT if changed else "k",
                                               weight="bold" if changed else "normal")))
        n += len(w) + 1
    lines.append(line)
    return VPacker(children=[HPacker(children=l, sep=3.5, pad=0) for l in lines], sep=1, pad=0)


def load(root, attack, method, index):
    sub = "clean" if attack is None else os.path.join("perturbed", attack)
    name = "results.csv" if attack is None else "perturbed_results.csv"
    r = pd.read_csv(os.path.join(root, sub, method, name)).set_index("index").loc[index]
    if attack is None:
        return {"image_logit": r.get("logit_image"), "text_logit": r.get("logit_text"),
                "logit": r["logit"] if "logit" in r else None, "pred": int(r["pred"]), "label": int(r["label"])}
    return {"image_logit": r["image_logit"], "text_logit": r["text_logit"], "logit": r["logit"],
            "pred": int(r["pred"]), "label": int(r["label"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="Fakeddit")
    args = ap.parse_args()
    root = dataset_result_root(args.dataset)
    samples = [s for s in SAMPLES if attacked_mask(load(root, None, s[0], s[1])["label"])]
    ex_dir = os.path.join(root.replace("results", "data_perturbed", 1), "flip_examples")
    texts = {a: pd.read_csv(os.path.join(ex_dir, f"{args.dataset}_seed{RAND_SEED}_{a}_texts.csv"))
             .set_index(["fusion", "index"]) for a in ("joint", "trepat")}

    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(7.2, 6.3))
    cells = fig.subfigures(2, 2, wspace=0.04, hspace=0.04).ravel()
    md = [f"# {args.dataset} seed {RAND_SEED}: flip examples", "",
          f"Image perturbation shown as 0.5 + {AMP:g}·δ (ε = 1/255). Logits from "
          f"`{root}` (not shown in the figure); decision = prediction at the method's calibrated threshold.", ""]
    for r, (method, idx, shown, case) in enumerate(samples):
        gs = cells[r].add_gridspec(2, 2, height_ratios=[1, 0.62], wspace=0.05, hspace=0.12, top=0.92, bottom=0.0, left=0.02, right=0.98)
        ax_img, ax_d = cells[r].add_subplot(gs[0, 0]), cells[r].add_subplot(gs[0, 1])
        ax_t = cells[r].add_subplot(gs[1, :])
        img = torch.load(os.path.join(ex_dir, f"{args.dataset}_seed{RAND_SEED}_{'pgd' if shown == 'trepat' else shown}_{method}_{idx}.pt"))
        clean, adv = img["clean"][0], img["perturbed"][0]
        delta = adv - clean
        ax_img.imshow(clean.permute(1, 2, 0).numpy(), interpolation="none")
        for ax in (ax_img, ax_d):
            ax.set_xticks([]); ax.set_yticks([])
        ax_t.axis("off")
        ax_img.set_xlabel("image", fontsize=10, labelpad=2)
        if shown == "trepat":
            ax_d.axis("off")
            ax_d.text(0.5, 0.5, "image not attacked", transform=ax_d.transAxes, ha="center", va="center",
                      fontsize=10, color="0.4")
        else:
            ax_d.imshow((0.5 + AMP * delta).clamp(0, 1).permute(1, 2, 0).numpy(), interpolation="none")
            ax_d.set_xlabel(f"{SHOWN[shown]} image δ ×{AMP:g}", fontsize=10, labelpad=2)
        name = NAMES.get(method, method)
        cells[r].suptitle(f"({'abcd'[r]}) {name}, {SHOWN[shown]}", fontsize=12, y=0.995)

        orig = texts["trepat"].loc[(method, idx)]["original"] if shown == "trepat" else \
            pd.read_csv(os.path.join("data", args.dataset, "test.tsv"), sep="\t").loc[idx, "clean_title"]
        children = [TextArea("text:" if shown == "pgd" else "original text:", textprops=dict(size=9.5, color="0.4")),
                    text_box([(w, False) for w in orig.split()], 11)]
        if shown == "pgd":
            children.append(TextArea("(text not attacked)", textprops=dict(size=9.5, color="0.4")))
            pert = orig
        else:
            pert = " ".join(str(texts[shown].loc[(method, idx)]["perturbed"]).split())
            children += [TextArea(f"{SHOWN[shown]} text:", textprops=dict(size=9.5, color="0.4")),
                         text_box(words(pert, orig), 11)]
        box = VPacker(children=children, sep=2, pad=0)
        ax_t.add_artist(AnchoredOffsetbox(loc="upper left", child=box, frameon=False, pad=0.1,
                                          borderpad=0.2))

        rows = {a: load(root, d, method, idx) for a, d in ATTACKS}
        label = rows["clean"]["label"]
        late = method != "feature-fusion"
        cols = ["image", "text"] if late else ["fused"]
        md += [f"## {name}, sample {idx} ({case}), truth {LABELS[label]}; shown: {SHOWN[shown]} only", "",
               f"- original text: {orig}", f"- {SHOWN[shown]} text: {pert}",
               f"- {SHOWN[shown]} image: max |δ| = {0 if shown == 'trepat' else delta.abs().max().item() * 255:.3f}/255", "",
               "| attack | " + " | ".join(f"{c} logit" for c in cols) + " | decision | correct |",
               "|---|" + "---|" * (len(cols) + 2)]
        for a, _ in ATTACKS:
            v = rows[a]
            vals = [v["image_logit"], v["text_logit"]] if late else [v["logit"]]
            md.append(f"| {a} | " + " | ".join(f"{val:+.4f}" for val in vals)
                      + f" | {LABELS[v['pred']]} | {'yes' if v['pred'] == label else 'no'} |")
        md.append("")

    stem = os.path.join("figures", f"{args.dataset}_seed{RAND_SEED}_{ATTACK_CLASSES}_flip_examples")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    with open(f"{stem}.md", "w") as f:
        f.write("\n".join(md))
    print(f"wrote {stem}.{{pdf,md}}")


if __name__ == "__main__":
    main()
