"""Regenerate the paper tables (clean, missing-modality, adversarial, ASR summary)
from results/<ds>/200/42. Cells whose text differs from the table currently in
OUT_TABLES are wrapped in \\chg{}. A run whose epsilon is not the main one
(config.yaml) is shown as ---.

Usage:
    python3 scripts/paper/make_paper_tables.py [RESULTS_ROOT OLD_TABLES OUT_TABLES]
"""
import re, sys, os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configuration_files.configuration import EPSILON

if len(sys.argv) == 4:
    RES, OLD, OUT = sys.argv[1:4]
else:
    RES = "results"
    OUT = "paper/generated/tables"  # never write into the Overleaf folder
    OLD = "paper/tables_baseline_pre_2026-10-12"  # change marks are relative to this version
DS = [("Recovery", "ReCOVery"), ("Fakeddit", "Fakeddit")]
LATE = [("min", "min"), ("mean", "mean"), ("max", "max"), ("linear", "linear"), ("svm-rbf", "SVM-RBF")]
FF = ("feature-fusion", "Feature fusion")
ATT = [("pgd", r"\multirow{8}{*}{\textit{PGD}}", "PGD"),
       ("trepat", r"\multirow{8}{*}{\textit{TREPAT}}", "TREPAT"),
       ("sum", "\\multirow{8}{*}{\\textit{Union}}", "PGD+TREPAT (sum)"),
       ("interleaved", "\\multirow{8}{*}{\\textit{Joint}}", "PGD+TREPAT (alt.)")]


def base(ds): return f"{RES}/{ds}/200/42"


def metrics(df):
    y, p, s = df.label.values, df.pred.values, df.score.values
    tp = ((y == 1) & (p == 1)).sum(); fn = ((y == 1) & (p == 0)).sum(); fp = ((y == 0) & (p == 1)).sum()
    return dict(AUC=roc_auc_score(y, s), F1=2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                Acc=(y == p).mean(), TPR=tp / (y == 1).sum(), TNR=((y == 0) & (p == 0)).sum() / (y == 0).sum())


def f1_from_rates(tpr, tnr, n=100):  # balanced 100/100: Real-positive F1
    tp = tpr * n; fn = n - tp; fp = (1 - tnr) * n
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def stale(ds, a, m):
    """True when the run is absent or was made with an epsilon other than the main one."""
    p = f"{base(ds)}/perturbed/{a}/{m}/parameters.json"
    if not os.path.exists(p): return True
    eps = json.load(open(p))["Attack"].get("Epsilon")
    return eps is not None and abs(eps - EPSILON) > 1e-9


MISSING = {(ds, a, m) for ds, _ in DS for a in ("pgd", "trepat", "sum", "interleaved")
           for m, _ in LATE + [FF] if stale(ds, a, m)}


def clean(ds, m): return pd.read_csv(f"{base(ds)}/clean/{m}/results.csv")
def pert(ds, a, m): return pd.read_csv(f"{base(ds)}/perturbed/{a}/{m}/perturbed_results.csv")


def flips(ds, a, m):
    c = clean(ds, m)[["index", "label", "pred"]]
    p = pert(ds, a, m)[["index", "pred"]]
    x = c.merge(p, on="index", suffixes=("", "_p"))
    cc = x.label == x.pred
    return cc, cc & (x.pred_p != x.label)


def asr(ds, a, m):
    cc, fl = flips(ds, a, m); return fl.sum() / cc.sum()


f3 = lambda v: f"{v:.3f}"
def dlt(v): return f"(${'-' if v < -5e-4 else '+'}$" + f"{abs(v):.3f})"
def strip(s):
    s, prev = s.strip(), None
    while s != prev:
        prev, s = s, re.sub(r"^\\(textbf|chg)\{(.*)\}$", r"\2", s)
    return s


def bold(vals, higher=True):
    ok = [v for v in vals if v is not None]
    if len(ok) < 2: return [False] * len(vals)
    best = max(ok) if higher else min(ok)
    return [v is not None and round(v, 3) == round(best, 3) for v in vals]


# ---------- old-table parsing for change marks ----------
def old_rows(path):
    """{(dataset, block, method): [cell strings]} from an existing table."""
    out, ds, block = {}, None, None
    txt = open(path).read()
    for line in txt.splitlines():
        if "textbf{ReCOVery}" in line: ds = "ReCOVery"
        if "textbf{Fakeddit}" in line: ds = "Fakeddit"
        mb = re.search(r"\\textit\{(PGD|TREPAT|Image only|Text only|Unimodal|Late fusion|Feature fusion)\}", line)
        if mb and "multirow" in line: block = mb.group(1)
        if "(sum)" in line: block = "PGD+TREPAT (sum)"
        if "(alt.)" in line: block = "PGD+TREPAT (alt.)"
        if line.rstrip().endswith("\\\\") and "&" in line and "multicolumn" not in line:
            cells = [c.strip() for c in line.rstrip()[:-2].split("&")]
            if cells[0] == "": cells = cells[1:]
            name = cells[0].replace("\\quad", "").strip()
            out[(ds, block, name)] = [strip(c) for c in cells[1:]]
    return out


def mark(new_cells, old_cells):
    res = []
    for i, c in enumerate(new_cells):
        o = old_cells[i] if old_cells and i < len(old_cells) else None
        res.append(c if o is not None and strip(c) == o else r"\chg{" + c + "}")
    return res


def wrap_bold(c, b): return r"\textbf{" + c + "}" if b else c


# ---------- adversarial table ----------
def adv_table():
    old = old_rows(f"{OLD}/adv_results.tex")
    L = [r"""\begin{table*}[t]
\centering
\caption{Fusion methods evaluated under different adversarial attacks on the ReCOVery and Fakeddit test subsets (200 samples, 100 Fake / 100 Real), with $\epsilon=1/255$ for the image perturbation. Each classification metric reports the value and, in parentheses, its difference with respect to the same method on clean input. ASR is the untargeted attack success rate over the samples each method classifies correctly when clean. The union and joint attacks split the budget between the two modalities. Best result per column and attack is shown in bold; for ASR, lower is better. --- : run pending.}
\label{tab:adv-robustness}
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l l c c c c c c}
\hline
\textbf{Attack} & \textbf{Method} & \textbf{AUC ($\Delta$AUC)} & \textbf{F1 ($\Delta$F1)} & \textbf{Acc ($\Delta$Acc)} & \textbf{TPR ($\Delta$TPR)} & \textbf{TNR ($\Delta$TNR)} & \textbf{ASR} \\
\hline
"""]
    for ds, dname in DS:
        L.append(f"\n% {dname}\n\\multicolumn{{8}}{{c}}{{\\textbf{{{dname}}}}} \\\\\n\\hline\n")
        for a, head, blk in ATT:
            methods = LATE + [FF]
            data = {}
            for m, _ in methods:
                if (ds, a, m) in MISSING: data[m] = None; continue
                c, p = metrics(clean(ds, m)), metrics(pert(ds, a, m))
                data[m] = (p, {k: p[k] - c[k] for k in p}, asr(ds, a, m))
            cols = ["AUC", "F1", "Acc", "TPR", "TNR"]
            bmask = {k: bold([data[m][0][k] if data[m] else None for m, _ in methods]) for k in cols}
            bmask["ASR"] = bold([data[m][2] if data[m] else None for m, _ in methods], higher=False)
            L.append(head + "\n& \\multicolumn{7}{l}{\\textit{Late fusion}} \\\\\n")
            for i, (m, disp) in enumerate(methods):
                if m == FF[0]: L.append("& \\multicolumn{7}{l}{\\textit{Feature fusion}} \\\\\n")
                if data[m] is None:
                    L.append(f"& {disp:<14s} & " + " & ".join(["---"] * 6) + " \\\\ % TODO: run pending\n")
                    continue
                p, d, s = data[m]
                cells = [wrap_bold(f"{f3(p[k])} {dlt(d[k])}", bmask[k][i]) for k in cols]
                cells.append(wrap_bold(f3(s), bmask["ASR"][i]))
                cells = mark(cells, old.get((dname, blk, disp)))
                L.append(f"& {disp:<14s} & " + " & ".join(cells) + " \\\\\n")
            L.append("\\hline\n\n")
    L.append("\\end{tabular}\n\\end{table*}\n")
    return "".join(L)


# ---------- clean table ----------
def clean_table():
    old = old_rows(f"{OLD}/clean_results.tex")
    L = [r"""\begin{table}[t]
\centering
\caption{Clean performance of the unimodal classifiers and multimodal fusion methods on the ReCOVery and Fakeddit test subsets (200 samples, 100 Fake / 100 Real). Real is the positive class. AVG is the mean of the five metrics. Best result per metric and dataset in bold.}
\label{tab:clean-performance}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l c c c c c c}
"""]
    rows_def = [("Unimodal", [("text", "Text-only classifier"), ("image", "Image-only classifier")]),
                ("Late fusion", LATE), ("Feature fusion", [FF])]
    for ds, dname in DS:
        allm = [m for _, g in rows_def for m in g]
        met = {m: metrics(clean(ds, m)) for m, _ in allm}
        for m in met: met[m]["AVG"] = np.mean([met[m][k] for k in ["AUC", "F1", "Acc", "TPR", "TNR"]])
        cols = ["AUC", "F1", "Acc", "TPR", "TNR", "AVG"]
        bm = {k: bold([met[m][k] for m, _ in allm]) for k in cols}
        L.append(f"\n% {dname}\n\\hline\n\\multicolumn{{7}}{{c}}{{\\textbf{{{dname}}}}} \\\\\n\\hline\n"
                 "\\textbf{Method} & \\textbf{AUC} & \\textbf{F1} & \\textbf{Acc} & \\textbf{TPR} & \\textbf{TNR} & \\textbf{AVG} \\\\\n\\hline\n")
        i = 0
        for g, ms in rows_def:
            L.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{g}}}}} \\\\\n")
            for m, disp in ms:
                cells = [wrap_bold(f3(met[m][k]), bm[k][i]) for k in cols]
                cells = mark(cells, old.get((dname, None, disp)) or next(
                    (v for (d, _, n), v in old.items() if d == dname and n == disp), None))
                L.append(f"\\quad {disp} & " + " & ".join(cells) + " \\\\\n")
                i += 1
    L.append("\\hline\n\\end{tabular}\n\\end{table}\n")
    return "".join(L)


# ---------- missing-modality table ----------
def ablation_table():
    old = old_rows(f"{OLD}/ablation_results.tex")
    L = [r"""\begin{table*}[t]
\centering
\caption{Fusion methods evaluated under different input modalities on the ReCOVery and Fakeddit test subsets to probe each method's modality preference. For feature fusion the missing modality is replaced by a blank input. Each cell reports the metric and, in parentheses, its difference with respect to the same method using both modalities. F1 takes Real as the positive class. Best result per metric and modality is shown in bold.}
\label{tab:missing-modality-ablation}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l l c c c}
\hline
\textbf{Modalities} & \textbf{Method} & \textbf{AUC ($\Delta$AUC)} & \textbf{F1 ($\Delta$F1)} & \textbf{Acc ($\Delta$Acc)} \\
\hline
"""]
    for ds, dname in DS:
        lf = pd.read_csv(f"{base(ds)}/ablation/late-fusion/modality_ablation.csv")
        ff = pd.read_csv(f"{base(ds)}/ablation/feature-fusion/modality_ablation_metrics.csv")
        ff = ff[ff.method.str.contains("blank")].assign(method="feature-fusion")
        ab = pd.concat([lf, ff])
        ab["F1r"] = [f1_from_rates(t, n) for t, n in zip(ab.TPR, ab.TNR)]
        L.append(f"\n% {dname}\n\\multicolumn{{5}}{{c}}{{\\textbf{{{dname}}}}} \\\\\n\\hline\n")
        for cond, blk in [("image_only", "Image only"), ("text_only", "Text only")]:
            methods = LATE + [FF]
            vals = {}
            for m, _ in methods:
                r = ab[(ab.method == m) & (ab.condition == cond)].iloc[0]
                b = ab[(ab.method == m) & (ab.condition == "both")].iloc[0]
                vals[m] = {"AUC": (r.AUC, r.AUC - b.AUC), "F1": (r.F1r, r.F1r - b.F1r), "Acc": (r.Acc, r.Acc - b.Acc)}
            bm = {k: bold([vals[m][k][0] for m, _ in methods]) for k in ["AUC", "F1", "Acc"]}
            L.append(f"\\multirow{{8}}{{*}}{{\\textit{{{blk}}}}}\n& \\multicolumn{{4}}{{l}}{{\\textit{{Late fusion}}}} \\\\\n")
            for i, (m, disp) in enumerate(methods):
                if m == FF[0]: L.append("& \\multicolumn{4}{l}{\\textit{Feature fusion}} \\\\\n")
                cells = [wrap_bold(f"{f3(vals[m][k][0])} {dlt(vals[m][k][1])}", bm[k][i]) for k in ["AUC", "F1", "Acc"]]
                cells = mark(cells, old.get((dname, blk, disp)))
                L.append(f"& {disp:<14s} & " + " & ".join(cells) + " \\\\\n")
            L.append("\\hline\n\n")
    L.append("\\end{tabular}\n\\end{table*}\n")
    return "".join(L)


# ---------- ASR summary incl. independent union (new) ----------
def asr_table():
    L = [r"""\begin{table}[t]
\centering
\caption{\chg{Attack success rate (\%) per fusion method and attack ($\epsilon=1/255$). The union and joint attacks split the budget between the two modalities. $n$: clean-correct samples. Joint only: samples flipped by the joint attack but by none of PGD, TREPAT and the union attack.}}
\label{tab:asr-summary}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l c c c c c c}
\hline
\textbf{Method} & $n$ & \textbf{PGD} & \textbf{TREPAT} & \textbf{Union} & \textbf{Joint} & \textbf{Joint only} \\
"""]
    for ds, dname in DS:
        L.append(f"\\hline\n\\multicolumn{{7}}{{c}}{{\\textbf{{{dname}}}}} \\\\\n\\hline\n")
        for m, disp in LATE + [FF]:
            if any((ds, a, m) in MISSING for a in ("pgd", "trepat", "sum", "interleaved")):
                L.append(f"{disp} & " + " & ".join(["---"] * 6) + " \\\\ % TODO: run pending\n"); continue
            cc, fp = flips(ds, "pgd", m); _, ft = flips(ds, "trepat", m)
            _, fu = flips(ds, "sum", m); _, fj = flips(ds, "interleaved", m)
            n = cc.sum(); pc = lambda v: f"{100 * v.sum() / n:.1f}"
            only = int((fj & ~(fp | ft | fu)).sum())
            cells = [str(n), pc(fp), pc(ft), pc(fu), pc(fj), str(only)]
            L.append(f"{disp} & " + " & ".join(r"\chg{" + c + "}" for c in cells) + " \\\\\n")
    L.append("\\hline\n\\end{tabular}\n\\end{table}\n")
    return "".join(L)


print("pending (shown as ---):", sorted(MISSING) or "none")
os.makedirs(OUT, exist_ok=True)
for name, fn in [("adv_results", adv_table), ("clean_results", clean_table),
                 ("ablation_results", ablation_table), ("asr_summary", asr_table)]:
    s = fn()
    open(f"{OUT}/{name}.tex", "w").write(s)
    print(name, "cells marked:", s.count(r"\chg{"), "| unmarked numeric cells:",
          len(re.findall(r"& (?:\\textbf\{)?\d\.\d{3}", s)))
