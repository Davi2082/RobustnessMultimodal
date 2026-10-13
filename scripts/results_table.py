"""Clean, missing-modality ablation and adversarial metrics for one dataset,
kept separate: each has its own CSV (all metrics), its own md/tex table
(reported metrics) and its own paper table.

Real (label 1) is the positive class; predictions use each method's own
threshold (score > threshold). Metrics computed for every row: n, n_real,
n_fake, AUC, accuracy, balanced_accuracy, TPR, FPR, TNR, FNR, precision, F1.

* Clean (metrics_clean.csv): every method on the clean test set.
  Reported: AUC, accuracy, TPR, FPR.
* Ablation (metrics_ablation.csv): fusion methods with one modality missing.
  Late fusion: the missing unimodal output is set to score 0.5 (logit 0 for
  the heads on logits), at the method's threshold. Feature fusion: the joint
  model with the missing modality's tokens dropped (drop variant of
  scripts.utils.modality_ablation). Conditions: both, image only, text only;
  deltas against both. Reported: AUC, accuracy, TPR, FPR per condition.
* Adversarial (metrics_adv.csv): PGD, TREPAT, Sum, Union (perturbed/sum),
  Joint (perturbed/interleaved). Sum is the additive baseline, not a run: a
  sample counts as misclassified if PGD or TREPAT (each with its full budget)
  misclassifies it. Also dTPR, dFPR (change from clean), n_clean_correct,
  n_flipped, n_fixed and ASR = n_flipped / n_clean_correct.
  Reported: TPR (dTPR), FPR (dFPR), ASR per attack. An attacked run whose
  stored threshold differs from the clean one is stale and is skipped.

Usage:
    python3 scripts/results_table.py --dataset Fakeddit
Writes <result root>/{metrics,results}_{clean,ablation,adv}.* and
<paper-dir>/{clean,ablation,adv}_results_<DS>.tex and thresholds_<DS>.tex.
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

ATTACKS = [("PGD", "pgd"), ("TREPAT", "trepat"), ("Union", "sum"), ("Joint", "interleaved")]
# Reporting order; Sum is derived from PGD and TREPAT (see docstring).
SCENARIOS = ["PGD", "TREPAT", "Sum", "Union", "Joint"]
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
    """One row per available attack (and Sum) for this method."""
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

    adv_preds = {}
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
        adv_preds[name] = m.set_index("index")[["label", "pred_clean", "pred_adv"]]

    if "PGD" in adv_preds and "TREPAT" in adv_preds:
        a, b = adv_preds["PGD"], adv_preds["TREPAT"].reindex(adv_preds["PGD"].index)
        y = a["label"].values
        either_wrong = (a["pred_adv"].values != y) | (b["pred_adv"].values != y)
        rows.append(attacked_row("Sum", y, None, a["pred_clean"].values,
                                 np.where(either_wrong, 1 - y, y), t_clean))
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

        def fuse(text_present, image_present):
            if method in ("min", "mean", "max"):
                a_, b_ = (st if text_present else np.full_like(st, 0.5)), (si if image_present else np.full_like(si, 0.5))
                return {"min": np.minimum, "max": np.maximum,
                        "mean": lambda u, v: (u + v) / 2}[method](a_, b_)
            logits = head_input_space(dataset) == "logits"
            missing = 0.0 if logits else 0.5
            tx, im = (lt, li) if logits else (st, si)
            X = np.column_stack([tx if text_present else np.full_like(tx, missing),
                                 im if image_present else np.full_like(im, missing)])
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


def with_delta(v, d, tex=False):
    """'0.120 (+0.010)'; in LaTeX the minus sign is $-$."""
    if v is None or d is None:
        return "—"
    sign = ("$-$" if tex else "−") if d < -5e-4 else "+"
    return f"{v:.3f} ({sign}{abs(d):.3f})"


def write_table(path, title, header, body):
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in body]
    with open(path + ".md", "w") as f:
        f.write(title + "\n\n" + "\n".join(md) + "\n")
    tex = [r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\hline",
           " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\", r"\hline"]
    tex += [" & ".join(r).replace("—", "---") + r" \\" for r in body]
    tex += [r"\hline", r"\end{tabular}"]
    with open(path + ".tex", "w") as f:
        f.write("\n".join(tex) + "\n")
    print(title + "\n" + "\n".join(md) + "\n")


PAPER_GROUPS = [("Unimodal", [("Text-only classifier", "text"), ("Image-only classifier", "image")]),
                ("Late fusion", [("min", "min"), ("mean", "mean"), ("max", "max"),
                                 ("linear", "linear"), ("SVM-RBF", "svm-rbf")]),
                ("Feature fusion", [("Feature fusion", "feature-fusion")])]


def paper_clean_table(df, dataset, path):
    """Clean table in the paper's layout; best value per column in bold
    (highest AUC/Acc/TPR, lowest FPR)."""
    cols = CLEAN_COLS
    clean = df.set_index("method")
    best = {k: (clean[k].min() if k == "FPR" else clean[k].max()) for _, k in cols}
    n = len(cols) + 1
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Clean performance on the {dataset} test set. Real is the positive class; "
             r"each method uses its own threshold, set to a $10\%$ false positive rate on the "
             r"validation split. Best value per metric in bold.}",
             r"\label{tab:clean-performance}", r"\begin{tabular}{l" + " c" * len(cols) + "}", r"\hline",
             rf"\multicolumn{{{n}}}{{c}}{{\textbf{{{dataset}}}}} \\", r"\hline",
             r"\textbf{Method} & " + " & ".join(rf"\textbf{{{h}}}" for h, _ in cols) + r" \\", r"\hline"]
    for group, methods in PAPER_GROUPS:
        lines.append(rf"\multicolumn{{{n}}}{{l}}{{\textit{{{group}}}}} \\")
        for name, m in methods:
            if m not in clean.index:
                continue
            cells = [(rf"\textbf{{{clean.at[m, k]:.3f}}}" if np.isclose(clean.at[m, k], best[k])
                      else f"{clean.at[m, k]:.3f}") for _, k in cols]
            lines.append(rf"\quad {name} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def paper_adv_table(df, dataset, path):
    """Adversarial table in the paper's layout: one row per method, one column
    group (TPR, FPR, ASR) per attack; lowest ASR per attack in bold."""
    adv = df
    scen = [sc for sc in SCENARIOS if (adv.scenario == sc).any()]
    best = {sc: adv.loc[adv.scenario == sc, "ASR"].min() for sc in scen}
    ncol = 1 + 3 * len(scen)
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Fusion methods under attack on the {dataset} test set. TPR and FPR are computed "
             r"on all test samples, with the change from clean inputs in parentheses; ASR is the fraction "
             r"of the samples classified correctly on clean input that are misclassified after the attack. "
             r"Sum counts a sample as misclassified if PGD or TREPAT, each with its full budget, "
             r"misclassifies it; the union and joint attacks use half of each budget. "
             r"Lowest ASR per attack in bold.}",
             r"\label{tab:adv-results}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + " ccc" * len(scen) + "}", r"\hline",
             r"\textbf{Method} & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{\textbf{{{sc}}}}}" for sc in scen)
             + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(scen))),
             " & " + " & ".join(r"TPR ($\Delta$) & FPR ($\Delta$) & ASR" for _ in scen) + r" \\",
             r"\hline"]
    for group, methods in PAPER_GROUPS:
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{group}}}}} \\")
        for name, m in methods:
            cells = []
            for sc in scen:
                r = adv[(adv.method == m) & (adv.scenario == sc)]
                if r.empty:
                    cells += ["---"] * 3
                    continue
                r = r.iloc[0]
                asr = f"{r['ASR']:.3f}"
                if np.isclose(r["ASR"], best[sc]):
                    asr = rf"\textbf{{{asr}}}"
                cells += [with_delta(r["TPR"], r["dTPR"], tex=True),
                          with_delta(r["FPR"], r["dFPR"], tex=True), asr]
            lines.append(rf"\quad {name} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\end{table*}"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def paper_ablation_table(df, dataset, path):
    """Ablation table in the paper's layout: one row per fusion method, one
    column group (AUC, Acc, TPR, FPR) per condition."""
    conds = [(h, c) for h, c in ABLATION_CONDITIONS if (df.condition == c).any()]
    cols = CLEAN_COLS
    ncol = 1 + len(cols) * len(conds)
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Missing-modality ablation on the {dataset} test set. For late fusion, the "
             r"output of the missing uni-modal classifier is set to a score of $0.5$; feature fusion is "
             r"fed the available modality only. Real is the positive class; each method keeps its "
             r"threshold.}",
             r"\label{tab:missing-modality-ablation}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + (" " + "c" * len(cols)) * len(conds) + "}", r"\hline",
             r"\textbf{Method} & " + " & ".join(rf"\multicolumn{{{len(cols)}}}{{c}}{{\textbf{{{h}}}}}"
                                                   for h, _ in conds) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + len(cols) * k}-{1 + len(cols) * (k + 1)}}}"
                     for k in range(len(conds))),
             " & " + " & ".join(" & ".join(h for h, _ in cols) for _ in conds) + r" \\", r"\hline"]
    for group, methods in PAPER_GROUPS:
        present = [(n, m) for n, m in methods if (df.method == m).any()]
        if not present:
            continue
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{group}}}}} \\")
        for name, m in present:
            cells = []
            for _, cond in conds:
                r = df[(df.method == m) & (df.condition == cond)]
                cells += ["---"] * len(cols) if r.empty else [f"{r.iloc[0][k]:.3f}" for _, k in cols]
            lines.append(rf"\quad {name} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\end{table*}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def paper_thresholds_table(dataset, path):
    """Calibrated per-method thresholds (checkpoints/<ds>/thresholds.json)."""
    from configuration_files.configuration import thresholds_path
    with open(thresholds_path(dataset)) as f:
        info = json.load(f)
    table = info["thresholds"]
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Decision thresholds on the score of the Real class, set per method on the "
             rf"{dataset} validation split ($n={info['n_real'] + info['n_fake']}$) to a false positive "
             rf"rate of ${100 * info['target_fpr']:g}\%$, with the resulting validation TPR and FPR. The "
             r"thresholds of the trained rules are set on 5-fold cross-fitted validation scores. Each "
             r"method uses its threshold in all experiments.}",
             r"\label{tab:thresholds}", r"\begin{tabular}{l c c c c}", r"\hline",
             r"\textbf{Method} & \textbf{Threshold} & \textbf{Logit} & \textbf{Val TPR} & "
             r"\textbf{Val FPR} \\", r"\hline"]
    for group, methods in PAPER_GROUPS:
        present = [(n, m) for n, m in methods if m in table]
        if not present:
            continue
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{group}}}}} \\")
        for name, m in present:
            t = table[m]["threshold"]
            lines.append(rf"\quad {name} & {t:.5f} & {np.log(t / (1 - t)):.2f} & "
                         rf"{table[m]['val_tpr']:.3f} & {table[m]['val_fpr']:.3f} \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--paper-dir", default="paper/generated/tables")
    args = ap.parse_args()
    root = dataset_result_root(args.dataset)

    os.makedirs(args.paper_dir, exist_ok=True)
    clean = pd.DataFrame([r for _, m in METHODS for r in evaluate_clean(root, m)])
    if clean.empty:
        sys.exit(f"no clean results under {root}")
    ablation = pd.DataFrame([r for _, m in ABLATION_METHODS for r in evaluate_ablation(root, args.dataset, m)])
    adv = pd.DataFrame([r for _, m in METHODS for r in evaluate_adv(root, m)])
    for name, frame in (("clean", clean), ("ablation", ablation), ("adv", adv)):
        frame.to_csv(os.path.join(root, f"metrics_{name}.csv"), index=False)
    stale = os.path.join(root, "metrics_all.csv")
    if os.path.isfile(stale):
        os.remove(stale)

    def get(frame, m, k, **where):
        sel = frame["method"] == m if not frame.empty else None
        for col, val in where.items():
            sel &= frame[col] == val
        return frame.loc[sel, k].iloc[0] if sel is not None and sel.any() and k in frame else None

    # 0. calibrated thresholds
    paper_thresholds_table(args.dataset, os.path.join(args.paper_dir, f"thresholds_{args.dataset}.tex"))

    # 1. clean
    write_table(os.path.join(root, "results_clean"),
                f"{args.dataset}, clean test set (Real = positive, per-method threshold)",
                ["Method"] + [h for h, _ in CLEAN_COLS],
                [[name] + [fmt(get(clean, m, k)) for _, k in CLEAN_COLS] for name, m in METHODS])
    paper_clean_table(clean, args.dataset, os.path.join(args.paper_dir, f"clean_results_{args.dataset}.tex"))

    # 2. missing-modality ablation
    if not ablation.empty:
        write_table(os.path.join(root, "results_ablation"),
                    f"{args.dataset}, missing-modality ablation (Real = positive, per-method threshold)",
                    ["Method"] + [f"{h} {c}" for h, _ in ABLATION_CONDITIONS for c, _ in CLEAN_COLS],
                    [[name] + [fmt(get(ablation, m, k, condition=cond))
                               for _, cond in ABLATION_CONDITIONS for _, k in CLEAN_COLS]
                     for name, m in ABLATION_METHODS])
        paper_ablation_table(ablation, args.dataset,
                             os.path.join(args.paper_dir, f"ablation_results_{args.dataset}.tex"))

    # 3. adversarial
    if not adv.empty:
        write_table(os.path.join(root, "results_adv"),
                    f"{args.dataset}, attacked test set (TPR/FPR on all samples, change from clean in "
                    f"parentheses; ASR on clean-correct; Sum = flipped by PGD or TREPAT)",
                    ["Method"] + [f"{sc} {h}" for sc in SCENARIOS for h in ("TPR (Δ)", "FPR (Δ)", "ASR")],
                    [[name] + [c for sc in SCENARIOS for c in (
                        with_delta(get(adv, m, "TPR", scenario=sc), get(adv, m, "dTPR", scenario=sc)),
                        with_delta(get(adv, m, "FPR", scenario=sc), get(adv, m, "dFPR", scenario=sc)),
                        fmt(get(adv, m, "ASR", scenario=sc)))]
                     for name, m in METHODS])
        paper_adv_table(adv, args.dataset, os.path.join(args.paper_dir, f"adv_results_{args.dataset}.tex"))


if __name__ == "__main__":
    main()
