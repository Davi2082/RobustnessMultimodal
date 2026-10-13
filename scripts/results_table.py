"""Clean, missing-modality ablation and adversarial metrics for one dataset,
kept separate. Writes only into the results folder of the run
(results/<dataset>/<subset>/<seed>/): CSV files with every metric and
Markdown tables with the reported ones. No LaTeX, nothing outside results/.

Real (label 1) is the positive class; predictions use each method's own
threshold (score > threshold). Metrics computed for every row: n, n_real,
n_fake, AUC, accuracy, balanced_accuracy, TPR, FPR, TNR, FNR, precision, F1.

* Clean (metrics_clean.csv, results_clean.md): AUC, accuracy, TPR, FPR.
* Ablation (metrics_ablation.csv, results_ablation.md): fusion methods with
  one modality missing. Late fusion: the missing unimodal output is set to its
  classifier's threshold (its logit for the heads on logits). Feature fusion:
  the available modality only (drop variant of scripts.utils.modality_ablation).
  Reported: AUC and accuracy with their change from both modalities.
* Adversarial (metrics_adv.csv, results_adv.md): PGD, TREPAT, Union
  (perturbed/sum, half budget per modality) and Joint (perturbed/interleaved,
  same budget). Also dTPR, dFPR (change from clean), n_clean_correct,
  n_flipped, n_fixed, ASR = n_flipped / n_clean_correct. Reported: TPR (dTPR),
  FPR (dFPR), ASR per attack. A run whose stored threshold differs from the
  clean one is stale and is skipped.
* Attack split (metrics_attack_split.csv, results_attack_split.md): ASR split by
  the attacks that succeed (total, unimodal only, union or joint only, union
  only, joint only).
* Thresholds (results_thresholds.md): per-method threshold, its logit and
  validation TPR/FPR.

Usage:
    python3 scripts/results_table.py --dataset Fakeddit
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configuration_files.paths import dataset_result_root  # noqa: E402
from configuration_files.configuration import RAND_SEED  # noqa: E402

ATTACKS = [("PGD", "pgd"), ("TREPAT", "trepat"), ("Union", "sum"), ("Joint", "interleaved")]
# Reporting order. Union = the "sum" attack (perturbed/sum, half budget per modality).
SCENARIOS = ["PGD", "TREPAT", "Union", "Joint"]
METHODS = [("Text-only", "text"), ("Image-only", "image"), ("Feature fusion", "feature-fusion"),
           ("min", "min"), ("mean", "mean"), ("max", "max"), ("SVM-RBF", "svm-rbf"), ("linear", "linear")]
CLEAN_COLS = [("AUC", "AUC"), ("Acc", "accuracy"), ("TPR", "TPR"), ("FPR", "FPR")]
ADV_COLS = [("TPR", "TPR"), ("FPR", "FPR"), ("ASR", "ASR")]


def threshold_of(run_dir):
    p = os.path.join(run_dir, "parameters.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        params = json.load(f)
    # Attack runs nest their settings; clean runs store them at the top level.
    for block in (params, *[v for v in params.values() if isinstance(v, dict)]):
        if "Threshold" in block:
            return block["Threshold"]
    return None


def classification_metrics(label, score, pred):
    real, fake = label == 1, label == 0
    tp, fp = int((pred[real] == 1).sum()), int((pred[fake] == 1).sum())
    fn, tn = int(real.sum()) - tp, int(fake.sum()) - fp
    tpr, fpr = tp / max(real.sum(), 1), fp / max(fake.sum(), 1)
    return {
        "n": len(label), "n_real": int(real.sum()), "n_fake": int(fake.sum()),
        "AUC": (roc_auc_score(label, score) if score is not None and real.any() and fake.any()
                else np.nan),
        "accuracy": (tp + tn) / len(label),
        "balanced_accuracy": (tpr + (1 - fpr)) / 2,
        "TPR": tpr, "FPR": fpr, "TNR": 1 - fpr, "FNR": 1 - tpr,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "F1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
    }


def load_clean(root, method):
    """(clean results frame, threshold, metrics) or None."""
    clean_dir = os.path.join(root, "clean", method)
    clean_csv = os.path.join(clean_dir, "results.csv")
    if not os.path.isfile(clean_csv):
        return None
    c = pd.read_csv(clean_csv)
    return c, threshold_of(clean_dir), classification_metrics(
        c["label"].values, c["score"].values, c["pred"].values)


def evaluate_clean(root, method):
    loaded = load_clean(root, method)
    if loaded is None:
        return []
    _, t_clean, clean = loaded
    return [{"method": method, "threshold": t_clean, **clean}]


def evaluate_adv(root, method):
    """One row per available attack for this method."""
    loaded = load_clean(root, method)
    if loaded is None:
        return []
    c, t_clean, clean = loaded
    rows = []

    def attacked_row(name, y, score, pred_clean, pred_adv, threshold):
        correct, wrong_adv = pred_clean == y, pred_adv != y
        n_ok = int(correct.sum())
        row = {"method": method, "scenario": name, "threshold": threshold,
               **classification_metrics(y, score, pred_adv)}
        row.update({"dTPR": row["TPR"] - clean["TPR"], "dFPR": row["FPR"] - clean["FPR"],
                    "n_clean_correct": n_ok, "n_flipped": int((correct & wrong_adv).sum()),
                    "n_fixed": int((~correct & ~wrong_adv).sum()),
                    "ASR": (correct & wrong_adv).sum() / n_ok if n_ok else np.nan})
        return row

    for name, attack in ATTACKS:
        run_dir = os.path.join(root, "perturbed", attack, method)
        pert_csv = os.path.join(run_dir, "perturbed_results.csv")
        if not os.path.isfile(pert_csv):
            continue
        t_adv = threshold_of(run_dir)
        if t_clean is None or t_adv is None or not np.isclose(t_adv, t_clean, rtol=0, atol=1e-12):
            print(f"WARNING: {attack}/{method} threshold {t_adv} != clean {t_clean}: stale run, skipped")
            continue
        p = pd.read_csv(pert_csv)
        m = c[["index", "label", "pred"]].merge(p[["index", "score", "pred"]], on="index",
                                                suffixes=("_clean", "_adv"))
        if len(m) != len(c):
            print(f"WARNING: {attack}/{method} has {len(m)} of {len(c)} samples")
        rows.append(attacked_row(name, m["label"].values, m["score"].values,
                                 m["pred_clean"].values, m["pred_adv"].values, t_adv))

    order = {sc: k for k, sc in enumerate(SCENARIOS)}
    return sorted(rows, key=lambda r: order[r["scenario"]])


ABLATION_CONDITIONS = [("Both", "both"), ("Image only", "image_only"), ("Text only", "text_only")]
ABLATION_METHODS = [m for m in METHODS if m[1] not in ("text", "image")]


def evaluate_ablation(root, dataset, method):
    """Rows (both, image only, text only) for one fusion method."""
    from configuration_files.configuration import method_threshold
    loaded = load_clean(root, method)
    if loaded is None:
        return []
    c, t, _ = loaded
    y = c["label"].values
    if method == "feature-fusion":
        path = os.path.join(root, "ablation", "feature-fusion", "modality_ablation.csv")
        if not os.path.isfile(path):
            return []
        a = c[["index"]].merge(pd.read_csv(path), on="index")
        scores = {"both": a["drop_score_both"].values, "image_only": a["drop_score_img_only"].values,
                  "text_only": a["drop_score_txt_only"].values}
    else:
        from models.fusion import head_input_space, pytorch_head_score
        t = method_threshold(method, dataset)
        st, si = c["score_text"].values, c["score_image"].values
        lt, li = c["logit_text"].values, c["logit_image"].values

        from scripts.main_scripts.run_ablation import missing_score
        # A missing modality gets its unimodal classifier's threshold (score, or
        # its logit for heads on logits): the classifier's decision point.
        t_txt, t_img = missing_score("text", dataset), missing_score("image", dataset)

        def fuse(text_present, image_present):
            if method in ("min", "mean", "max"):
                a_ = st if text_present else np.full_like(st, t_txt)
                b_ = si if image_present else np.full_like(si, t_img)
                return {"min": np.minimum, "max": np.maximum,
                        "mean": lambda u, v: (u + v) / 2}[method](a_, b_)
            logits = head_input_space(dataset) == "logits"
            fill = (lambda t: np.log(t / (1 - t))) if logits else (lambda t: t)
            tx, im = (lt, li) if logits else (st, si)
            X = np.column_stack([tx if text_present else np.full_like(tx, fill(t_txt)),
                                 im if image_present else np.full_like(im, fill(t_img))])
            return np.asarray(pytorch_head_score(method, X, dataset)).reshape(-1)

        scores = {"both": fuse(True, True), "image_only": fuse(False, True), "text_only": fuse(True, False)}
        assert ((scores["both"] > t).astype(int) == c["pred"].values).all(), f"{method}: both != clean"
    rows, base = [], None
    for _, cond in ABLATION_CONDITIONS:
        m = classification_metrics(y, scores[cond], (scores[cond] > t).astype(int))
        row = {"method": method, "condition": cond, "threshold": t, **m}
        if base is None:
            base = m
        else:
            row.update({f"d{k}": m[k] - base[k] for k in ("AUC", "accuracy", "TPR", "FPR")})
        rows.append(row)
    return rows


def fmt(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def with_delta(v, d):
    """'0.120 (+0.010)'."""
    if v is None or d is None:
        return "—"
    sign = "−" if d < -5e-4 else "+"
    return f"{v:.3f} ({sign}{abs(d):.3f})"


def write_table(path, title, header, body):
    """Markdown table (path + '.md')."""
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in body]
    with open(path + ".md", "w") as f:
        f.write(title + "\n\n" + "\n".join(md) + "\n")
    print(title + "\n" + "\n".join(md) + "\n")


def attack_overlap_asr(root, method):
    """ASR (%) over the clean-correct samples, split by which attacks succeed:
    total (any of PGD, TREPAT, union, joint); unimodal only (PGD or TREPAT, but
    neither union nor joint); union or joint only (but neither PGD nor TREPAT),
    and within it union only and joint only."""
    c = pd.read_csv(os.path.join(root, "clean", method, "results.csv")).set_index("index")
    ok = (c["pred"] == c["label"]).values

    def flipped(attack):
        p = pd.read_csv(os.path.join(root, "perturbed", attack, method, "perturbed_results.csv"))
        return ok & (p.set_index("index")["pred"].reindex(c.index).values != c["label"].values)

    uni = flipped("pgd") | flipped("trepat")
    u, j = flipped("sum"), flipped("interleaved")
    multi = u | j
    n = ok.sum()
    pct = lambda mask: 100 * mask.sum() / n if n else np.nan
    return {"method": method, "total": pct(uni | multi), "unimodal_only": pct(uni & ~multi),
            "multimodal_only": pct(multi & ~uni), "union_only": pct(u & ~j & ~uni),
            "joint_only": pct(j & ~u & ~uni)}


SPLIT_COLS = [("Total", "total"), ("Unimodal only", "unimodal_only"),
              ("Union or joint only", "multimodal_only"), ("Union only", "union_only"),
              ("Joint only", "joint_only")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    root = dataset_result_root(args.dataset)

    clean = pd.DataFrame([r for _, m in METHODS for r in evaluate_clean(root, m)])
    if clean.empty:
        sys.exit(f"no clean results under {root}")
    ablation = pd.DataFrame([r for _, m in ABLATION_METHODS for r in evaluate_ablation(root, args.dataset, m)])
    adv = pd.DataFrame([r for _, m in METHODS for r in evaluate_adv(root, m)])
    # Every saved result names the dataset and test-subset seed it was computed on.
    tag = f"{args.dataset} (seed {RAND_SEED})"
    for name, frame in (("clean", clean), ("ablation", ablation), ("adv", adv)):
        frame.insert(0, "seed", RAND_SEED)
        frame.insert(0, "dataset", args.dataset)
        frame.to_csv(os.path.join(root, f"metrics_{name}.csv"), index=False)

    def get(frame, m, k, **where):
        sel = frame["method"] == m if not frame.empty else None
        for col, val in where.items():
            sel &= frame[col] == val
        return frame.loc[sel, k].iloc[0] if sel is not None and sel.any() and k in frame else None

    # Thresholds
    from configuration_files.configuration import thresholds_path
    with open(thresholds_path(args.dataset)) as f:
        th = json.load(f)["thresholds"]
    write_table(os.path.join(root, "results_thresholds"), f"{tag}, calibrated thresholds",
                ["Method", "Threshold", "Logit", "Val TPR", "Val FPR"],
                [[name, f"{th[m]['threshold']:.5f}",
                  f"{np.log(th[m]['threshold'] / (1 - th[m]['threshold'])):.2f}",
                  f"{th[m]['val_tpr']:.3f}", f"{th[m]['val_fpr']:.3f}"]
                 for name, m in METHODS if m in th])

    # Clean
    write_table(os.path.join(root, "results_clean"),
                f"{tag}, clean test set (Real = positive, per-method threshold)",
                ["Method"] + [h for h, _ in CLEAN_COLS],
                [[name] + [fmt(get(clean, m, k)) for _, k in CLEAN_COLS] for name, m in METHODS])

    # Missing-modality ablation: AUC and accuracy, change from both modalities
    if not ablation.empty:
        cols = [("AUC", "AUC"), ("Acc", "accuracy")]
        conds = [c for c in ABLATION_CONDITIONS if c[1] != "both"]
        write_table(os.path.join(root, "results_ablation"),
                    f"{tag}, missing-modality ablation (change from both modalities in parentheses)",
                    ["Method"] + [f"{h} {c}" for h, _ in conds for c, _ in cols],
                    [[name] + [with_delta(get(ablation, m, k, condition=cond),
                                          get(ablation, m, "d" + k, condition=cond))
                               for _, cond in conds for _, k in cols]
                     for name, m in ABLATION_METHODS])

    # Adversarial
    if not adv.empty:
        write_table(os.path.join(root, "results_adv"),
                    f"{tag}, attacked test set (TPR/FPR on all samples, change from clean in "
                    f"parentheses; ASR on clean-correct; Union = sum attack, half budget each)",
                    ["Method"] + [f"{sc} {h}" for sc in SCENARIOS for h in ("TPR (Δ)", "FPR (Δ)", "ASR")],
                    [[name] + [c for sc in SCENARIOS for c in (
                        with_delta(get(adv, m, "TPR", scenario=sc), get(adv, m, "dTPR", scenario=sc)),
                        with_delta(get(adv, m, "FPR", scenario=sc), get(adv, m, "dFPR", scenario=sc)),
                        fmt(get(adv, m, "ASR", scenario=sc)))]
                     for name, m in METHODS])

        # ASR split by the attacks that succeed (fusion methods only)
        try:
            split = pd.DataFrame([attack_overlap_asr(root, m) for _, m in ABLATION_METHODS])
        except FileNotFoundError as e:
            print(f"attack split skipped: {e}")
        else:
            split.insert(0, "seed", RAND_SEED)
            split.insert(0, "dataset", args.dataset)
            split.to_csv(os.path.join(root, "metrics_attack_split.csv"), index=False)
            write_table(os.path.join(root, "results_attack_split"),
                        f"{tag}, ASR (%) split by the attacks that succeed",
                        ["Method"] + [h for h, _ in SPLIT_COLS],
                        [[name] + [f"{split.set_index('method').at[m, k]:.1f}" for _, k in SPLIT_COLS]
                         for name, m in ABLATION_METHODS])
    print(f"wrote metrics_*.csv and results_*.md in {root}")


if __name__ == "__main__":
    main()
