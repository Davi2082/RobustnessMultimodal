"""Table 2: adversarial robustness of the six fusion methods, UNTARGETED.

Six fusion methods -- feature fusion (Themis) plus five late-fusion rules
(min, mean, max, RBF-SVM, linear) -- under three attacks:

    PGD          image channel only
    TREPAT       text channel only
    PGD+TREPAT   both channels, perturbed independently (disjoint sum)

Threat model is UNTARGETED, matching every attack script's default: each sample
the model classifies correctly on clean input is attacked and pushed toward the
other class, whatever that class is. So ASR is measured over all clean-correct
samples, not just correctly-detected fakes -- the targeted convention would use
the wrong denominator here and inflate nothing while shrinking |A| sevenfold.

Where the perturbations come from:

* Late fusion. All five rules are post-hoc functions of the two unimodal
  logits, and the two unimodal attacks are independent, so the disjoint sum
  needs no extra GPU run: take the image logit from the PGD run, the text logit
  from the TREPAT run, and the untouched channel's logit from the clean run.
* Feature fusion. Themis is a single joint network, so each scope is its own
  white-box run under perturbed/feature-fusion/.

Metrics put fake news (label 0) in the positive class, matching the other
tables in the paper.

Usage:
    python3 adv_robustness_table.py
"""

import os
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from configuration_files.configuration import RAND_SEED, THRESHOLD
from configuration_files.paths import RESULT_PATH, late_fusion_directory_name, late_fusion_scenario_path
from models.fusion import load_fitted_heads
from scripts.plot_clean_score_space import fit_heads, rule_scores

RULES = ("min", "mean", "max", "svm-rbf", "linear")
ATTACKS = ("PGD", "TREPAT", "PGD+TREPAT")
SCOPE = {"PGD": "image", "TREPAT": "text", "PGD+TREPAT": "both"}
ROW_ORDER = ("min", "mean", "max", "linear", "svm-rbf", "feature-fusion")
DISPLAY = {"min": "min", "mean": "mean", "max": "max", "linear": "linear",
           "svm-rbf": "SVM-RBF", "feature-fusion": "Feature fusion"}
FF_SUBDIR = {"PGD": "image-perturbed", "TREPAT": "text-perturbed", "PGD+TREPAT": ""}
OUT_CSV = os.path.join(RESULT_PATH, "fusion_analysis", "adv_robustness_table.csv")


def paper_table_path(filename="themis_adv_robustness.tex"):
    """Locate the Overleaf export, whose directory name changes on re-download."""
    projects = [p for p in sorted(glob.glob(os.path.join("paper", "overleaf", "*")))
                if os.path.isdir(p)]
    if not projects:
        return os.path.join("paper", "tables", filename)
    return os.path.join(max(projects, key=os.path.getmtime), "tables", filename)


def wilson(k, n, z=1.96):
    """Wilson score interval; the normal approximation is unusable at these n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def scored(clean_score, pert_score, label):
    """AUC / F1 / Acc under attack plus the untargeted ASR.

    ASR counts only samples the model got RIGHT on clean input -- one already
    misclassified needs no attack, and counting it would reward a weak model.
    """
    label = np.asarray(label)
    y = 1 - label                                   # fake is the positive class
    pert_pred = (np.asarray(pert_score) >= THRESHOLD).astype(int)
    correct_clean = (np.asarray(clean_score) >= THRESHOLD).astype(int) == label
    n_att = int(correct_clean.sum())
    flipped = int((pert_pred[correct_clean] != label[correct_clean]).sum())
    lo, hi = wilson(flipped, n_att)

    # Untargeted ASR pools two very different events on an 86%-real test set:
    # fake->real (evading the detector, the operationally meaningful direction)
    # and real->fake (a false alarm). Split them, or a rule that merely refuses
    # to move reals will read as "robust".
    out = {
        "AUC": roc_auc_score(y, 1 - np.asarray(pert_score)),
        "F1": f1_score(y, 1 - pert_pred),
        "Acc": accuracy_score(label, pert_pred),
        "ASR": flipped / n_att if n_att else float("nan"),
        "ASR_lo": lo, "ASR_hi": hi,
        "n_attackable": n_att,
    }
    for name, cls in (("fake", 0), ("real", 1)):
        sel = correct_clean & (label == cls)
        n = int(sel.sum())
        out[f"n_{name}"] = n
        out[f"ASR_{name}"] = float((pert_pred[sel] != cls).mean()) if n else float("nan")
    return out


def main():
    R = RESULT_PATH
    heads = load_fitted_heads() or fit_heads(
        pd.read_csv(os.path.join(R, "fusion_analysis",
                                 "train_unimodal_logits_clean.csv")),
        RAND_SEED)

    # ---- clean unimodal logits ----
    t = pd.read_csv(os.path.join(R, "clean", "text", "results.csv")).set_index("index")
    i = pd.read_csv(os.path.join(R, "clean", "image", "results.csv")).set_index("index")
    idx = t.index.intersection(i.index)
    label = t.loc[idx, "label"].values
    txt_clean, img_clean = t.loc[idx, "logit"].values, i.loc[idx, "logit"].values

    # ---- perturbed unimodal logits (each attack touches one channel) ----
    def perturbed(sub):
        p = os.path.join(R, "perturbed", sub, "perturbed_results.csv")
        return pd.read_csv(p).set_index("index").loc[idx, "logit"].values if os.path.exists(p) else None

    img_pgd, txt_trepat = perturbed("image"), perturbed("text")
    channels = {
        "PGD": (img_pgd, txt_clean),
        "TREPAT": (img_clean, txt_trepat),
        "PGD+TREPAT": (img_pgd, txt_trepat),
    }

    # clean baselines, same metric convention as the attacked rows
    def clean_row(score, label):
        pred = (np.asarray(score) >= THRESHOLD).astype(int)
        return {"AUC": roc_auc_score(1 - np.asarray(label), 1 - np.asarray(score)),
                "F1": f1_score(1 - np.asarray(label), 1 - pred),
                "Acc": accuracy_score(label, pred)}

    clean_metrics = {}
    # Late fusion: read the attacks crafted against the FUSED score. The
    # unimodal-attack CSVs are NOT valid here -- an attacker never sees an
    # internal branch of a fusion model. See the threat model in CLAUDE.md.
    rows = []
    for rule in RULES:
        clean_s = rule_scores(rule, img_clean, txt_clean, heads)
        clean_metrics[rule] = clean_row(clean_s, label)
        clean_ser = pd.Series(clean_s, index=idx)
        fusion_dir = os.path.join(R, "perturbed", "late-fusion",
                                  late_fusion_directory_name(rule))
        for attack in ATTACKS:
            p_csv = late_fusion_scenario_path(fusion_dir, SCOPE[attack])
            if not os.path.exists(p_csv):
                print(f"skip {rule}/{attack}: {p_csv} not found")
                continue
            d = pd.read_csv(p_csv).set_index("index")
            common = clean_ser.index.intersection(d.index)
            col = "fused_score" if "fused_score" in d.columns else "score"
            rows.append({"fusion": rule, "attack": attack,
                         **scored(clean_ser.loc[common].values,
                                  d.loc[common, col].values,
                                  t.loc[common, "label"].values)})

    # ---- feature fusion: one white-box run per scope ----
    ff_clean_path = os.path.join(R, "clean", "feature-fusion", "results.csv")
    ff_clean = pd.read_csv(ff_clean_path).set_index("index")
    clean_metrics["feature-fusion"] = clean_row(ff_clean["score"].values,
                                                ff_clean["label"].values)
    for attack in ATTACKS:
        sub = FF_SUBDIR[attack]
        p = os.path.join(R, "perturbed", "feature-fusion", sub, "perturbed_results.csv")
        if not os.path.exists(p):
            print(f"skip feature-fusion/{attack}: {p} not found")
            continue
        d = pd.read_csv(p).set_index("index")
        common = ff_clean.index.intersection(d.index)
        rows.append({"fusion": "feature-fusion", "attack": attack,
                     **scored(ff_clean.loc[common, "score"].values,
                              d.loc[common, "score"].values,
                              ff_clean.loc[common, "label"].values)})

    if not rows:
        raise SystemExit("no attack results found -- run the attacks first")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.round(4).to_csv(OUT_CSV, index=False)

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Fusion methods evaluated under different attacks. Each cell "
        r"reports the metric and, in parentheses, its difference with respect to "
        r"the same method on clean input. ASR is the untargeted attack success "
        r"rate over the samples each method classifies correctly when clean.}",
        r"\label{tab:adv-robustness}",
        r"\begin{tabular}{l l c c c c}", r"\hline",
        r"\textbf{Attack} & \textbf{Method} & \textbf{AUC ($\Delta$AUC)} & "
        r"\textbf{F1 ($\Delta$F1)} & \textbf{Acc ($\Delta$Acc)} & \textbf{ASR} \\",
        r"\hline",
    ]
    lookup = {(r["fusion"], r["attack"]): r for _, r in df.iterrows()}
    for attack in ATTACKS:
        present = [m for m in ROW_ORDER if (m, attack) in lookup]
        best = {c: max((lookup[(m, attack)][c] for m in present if m in RULES),
                       default=None) for c in ("AUC", "F1", "Acc")}
        lines.append(r"\multirow{7}{*}{\textit{%s}}" % attack.replace("+", "+"))
        lines.append(r"& \multicolumn{5}{l}{\textit{Late fusion}} \\")
        for m in ROW_ORDER:
            label = DISPLAY[m]
            if (m, attack) not in lookup:
                lines.append(f"& {label:<14} & --- & --- & --- & --- \\\\")
                continue
            r, cl = lookup[(m, attack)], clean_metrics[m]
            cells = []
            for c in ("AUC", "F1", "Acc"):
                d = r[c] - cl[c]
                cell = f"{r[c]:.3f} ($-${abs(d):.3f})" if d < 0 else f"{r[c]:.3f} ($+${d:.3f})"
                if m in RULES and best[c] is not None and r[c] == best[c]:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            lines.append(f"& {label:<14} & " + " & ".join(cells) + f" & {r['ASR']:.3f} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}", ""]

    out = paper_table_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines))

    print(df[["fusion", "attack", "n_attackable", "AUC", "F1", "Acc", "ASR",
              "n_fake", "ASR_fake", "n_real", "ASR_real"]]
          .round(3).to_string(index=False))
    print(f"\nWrote {out}\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
