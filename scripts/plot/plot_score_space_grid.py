"""Score-space scatters: one figure per scenario, one panel per fusion method.

Each panel places every test sample in the plane spanned by the two uni-modal
outputs (logit-scaled axes, labelled in logits) and shades the region the fusion rule accepts, so the
effect of an attack is visible as movement relative to a fixed boundary.
Feature fusion has no such plane -- its decision is not a function of the two
uni-modal scores -- so its panel plots the joint model's own scores when fed
the image only and the text only (drop ablation), with no decision region.

With ``--axes ablation`` every method is placed the same way, by its own
missing-modality ablation: x = fused score with the text score set to 0.5
(image only), y = fused score with the image score set to 0.5 (text only) --
the rule used by run_ablation.py. The background/boundary is the same rule
drawn as in the unimodal view.

Figures go to <out-dir>/{unimodal,ablation}_scores/<dataset>/.

Usage:
    python3 scripts/plot/plot_score_space_grid.py [--dataset Recovery] [--axes ablation]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.collections import LineCollection

from configuration_files.configuration import THRESHOLD, DATASET, method_threshold
from configuration_files.paths import dataset_result_root

LATE_RULES = ("min", "mean", "max", "svm-rbf", "linear")
METHODS = list(LATE_RULES) + ["feature-fusion"]

ATTACK_DIRS = {
    "PGD": "pgd",
    "TREPAT": "trepat",
    "Union": "sum",
    "Joint": "interleaved",
}

TITLES = {
    "clean": "No attack",
    "PGD": "PGD (image)",
    "TREPAT": "TREPAT (text)",
    "Union": "Union (PGD + TREPAT, independent)",
    "Joint": "Joint (PGD + TREPAT, alternating)",
}

LABELS = {"svm-rbf": "SVM-RBF", "feature-fusion": "feature fusion"}
FIG_DIR = "figures/classification_results/scatter"
GRID = 300
# Scores saturate near 0 and 1, so axes use a logit scale (ticks stay in score
# units). EPS clips exact 0/1 so every point lands on the axes.
EPS = 1e-6
LIMITS = (EPS / 2, 1 - EPS / 2)
TICKS = [1e-4, 1e-2, 0.5, 1 - 1e-2, 1 - 1e-4]
TICK_LABELS = ["1e-4", "0.01", "0.5", "0.99", "0.9999"]
MARKER_SCALE = 1.0  # --column draws smaller markers
LOGIT_TICKS = [-10, -5, 0, 5, 10]  # default tick labels (logit units); --score-ticks uses the above
# Candidate ticks for the ablation view; each panel keeps those in its range.
ABLATION_TICKS = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95,
                  0.99, 0.999, 0.9999]


# (image column, text column) per source.
COLUMNS = {
    "clean": ("score_image", "score_text"),
    "perturbed": ("image_score", "text_score"),
    "ablation": ("drop_score_img_only", "drop_score_txt_only"),
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def score_grid(lim):
    """GRID points evenly spaced on the logit scale between the limits."""
    lo, hi = (np.log(v / (1 - v)) for v in lim)
    return 1 / (1 + np.exp(-np.linspace(lo, hi, GRID)))

def head_input_space(dataset=None):
    """Feature space the learned heads were fitted on ("scores" or "logits")."""
    from models.fusion import head_input_space as _space
    return _space(dataset)


class TorchHead:
    """The pipeline's torch fusion head behind a sklearn-style predict_proba."""

    def __init__(self, rule, dataset):
        self.rule, self.dataset = rule, dataset

    def predict_proba(self, X):
        from models.fusion import pytorch_head_score
        p = np.asarray(pytorch_head_score(self.rule, X, self.dataset), dtype=float).reshape(-1)
        return np.column_stack([1 - p, p])


def rule_scores(name, pi, pt, heads=None, input_space="scores"):
    """Fused score for image score pi and text score pt."""
    if name == "min":
        return np.minimum(pi, pt)
    if name == "mean":
        return (pi + pt) / 2
    if name == "max":
        return np.maximum(pi, pt)
    if name in ("svm-rbf", "linear"):
        if heads is None or name not in heads:
            return None
        head = getattr(heads[name], "best_estimator_", heads[name])
        # Heads are fitted on (text, image) columns -- see run_clean.py.
        feats = np.column_stack([pt, pi])
        if input_space == "logits":
            feats = np.clip(feats, 1e-7, 1 - 1e-7)
            feats = np.log(feats / (1 - feats))
        return head.predict_proba(feats)[:, 1]
    raise ValueError(f"Unknown rule: {name}")


def load_clean(result_root, method):
    path = os.path.join(result_root, "clean", method, "results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_perturbed(result_root, attack_dir, method):
    path = os.path.join(result_root, "perturbed", attack_dir, method, "perturbed_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_ff_ablation(result_root):
    path = os.path.join(result_root, "ablation", "feature-fusion", "modality_ablation.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def get_axes(df, method, ff_ablation=None):
    """Extract (label, image score, text score) arrays.

    Clean CSVs name the columns score_image/score_text, perturbed CSVs
    image_score/text_score. Feature fusion uses the clean modality ablation
    when given, otherwise the per-modality components of an attacked run.
    """
    cols = COLUMNS
    if method == "feature-fusion":
        if ff_ablation is not None:
            img_col, txt_col = cols["ablation"]
            return ff_ablation["label"].values, ff_ablation[img_col].values, ff_ablation[txt_col].values
        # Runs before the FeatureFusionClassifier fix copied the joint score
        # into the component columns; treat those as missing.
        if df is None or "image_score" not in df.columns or (
            np.allclose(df["image_score"], df["score"])
            and np.allclose(df["text_score"], df["score"])
        ):
            return None
    for img_col, txt_col in (cols["clean"], cols["perturbed"]):
        if img_col in df.columns and txt_col in df.columns:
            return df["label"].values, df[img_col].values, df[txt_col].values
    return None


def get_ablation_axes(df, method, ff_ablation=None, dataset=None):
    """(label, image-only value, text-only value) from missing-modality ablation.

    Late fusion: the fused score with the absent modality's score set to
    DEFAULT_SCORE, exactly as run_ablation.py computes it. Feature fusion: its
    own image-only / text-only outputs (same as get_axes).
    """
    if method == "feature-fusion":
        return get_axes(df, method, ff_ablation)
    from scripts.main_scripts.run_ablation import DEFAULT_SCORE, fuse_scores

    unimodal = get_axes(df, method)
    if unimodal is None:
        return None
    label, s_img, s_txt = unimodal
    img_only = fuse_scores(np.full_like(s_txt, DEFAULT_SCORE), s_img, method, dataset)
    txt_only = fuse_scores(s_txt, np.full_like(s_img, DEFAULT_SCORE), method, dataset)
    if img_only is None or txt_only is None:
        return None
    return label, np.asarray(img_only).reshape(-1), np.asarray(txt_only).reshape(-1)


def _logit(v):
    return np.log(v / (1 - v))


def fit_logit_limits(values, bound, pad=0.4):
    """Data range padded in logit units, kept inside the rule's bound."""
    lo = max(bound[0], EPS / 2, float(sigmoid(_logit(values.min()) - pad)))
    hi = min(bound[1], 1 - EPS / 2, float(sigmoid(_logit(values.max()) + pad)))
    return lo, hi


def spaced_logit_ticks(lim, min_frac=0.15):
    """ABLATION_TICKS inside lim, at least min_frac of the axis apart in logit units."""
    lo, hi = (_logit(v) for v in lim)
    kept = []
    for t in ABLATION_TICKS:
        x = _logit(t)
        if lo <= x <= hi and (not kept or x - _logit(kept[-1]) >= min_frac * (hi - lo)):
            kept.append(t)
    return kept


def ablation_bounds(method, dataset, pad=0.02):
    """Axis limits = the range each ablation score can take.

    The image-only score is the rule applied to (DEFAULT_SCORE, s_img) for
    s_img in [0, 1], the text-only score to (s_txt, DEFAULT_SCORE); e.g. min is
    bounded to [0, 0.5], mean to [0.25, 0.75], max to [0.5, 1]. Learned heads
    are swept numerically. Feature fusion is unbounded: [0, 1].
    """
    if method not in LATE_RULES:
        return (-pad, 1 + pad), (-pad, 1 + pad)
    from scripts.main_scripts.run_ablation import DEFAULT_SCORE, fuse_scores

    sweep = np.linspace(0, 1, 1001)
    fixed = np.full_like(sweep, DEFAULT_SCORE)
    limits = []
    for s_txt, s_img in ((fixed, sweep), (sweep, fixed)):  # image-only, text-only
        fused = fuse_scores(s_txt, s_img, method, dataset)
        if fused is None:
            limits.append((-pad, 1 + pad))
            continue
        fused = np.asarray(fused).reshape(-1)
        lo, hi = float(fused.min()), float(fused.max())
        margin = pad * max(hi - lo, 1e-3)
        limits.append((lo - margin, hi + margin))
    return limits[0], limits[1]


def sample_index(df, method, ff_ablation=None):
    """Sample ids in the same order as get_axes / get_ablation_axes return them."""
    source = ff_ablation if method == "feature-fusion" and ff_ablation is not None else df
    return source["index"].values


def _clean_vs_attacked(result_root, method, attack_dir):
    clean = load_clean(result_root, method)
    attacked = load_perturbed(result_root, attack_dir, method)
    if clean is None or attacked is None:
        return None
    return clean[["index", "label", "pred"]].merge(
        attacked[["index", "pred"]], on="index", suffixes=("_clean", "_adv"))


def flipped_ids(result_root, method, attack_dir):
    """Ids the attack flipped: correct on clean inputs, wrong after the attack."""
    merged = _clean_vs_attacked(result_root, method, attack_dir)
    if merged is None:
        return None
    hit = (merged["pred_clean"] == merged["label"]) & (merged["pred_adv"] != merged["label"])
    return set(merged.loc[hit, "index"])


def attack_asr(result_root, method, attack_dir):
    """(flipped, clean-correct): ASR over every clean-correct sample (untargeted)."""
    merged = _clean_vs_attacked(result_root, method, attack_dir)
    if merged is None:
        return None
    correct = merged["pred_clean"] == merged["label"]
    flipped = correct & (merged["pred_adv"] != merged["label"])
    return int(flipped.sum()), int(correct.sum())


def attacked_title(name, asr, n_unique=None):
    """Panel title for an attacked scenario: ASR instead of a raw error count."""
    if asr is None:
        return name
    flipped, correct = asr
    rate = flipped / correct if correct else float("nan")
    if n_unique is None:
        return f"{name}\nASR {rate:.0%} ({flipped}/{correct} clean-correct)"
    return (f"{name}\n{n_unique} flipped only by this attack "
            f"({n_unique / correct:.0%} of {correct} clean-correct)")


def unique_flips(result_root, method, scenarios):
    """Per scenario: ids flipped by that attack and by none of the others."""
    flips = {sc: flipped_ids(result_root, method, ATTACK_DIRS[sc]) for sc in scenarios}
    flips = {sc: ids for sc, ids in flips.items() if ids is not None}
    return {sc: ids - set().union(*(o for other, o in flips.items() if other != sc))
            for sc, ids in flips.items()}


def decision_region(ax, rule, heads, xlim, ylim, input_space, threshold=THRESHOLD):
    """Shade where the rule predicts Real (score > threshold) and draw the boundary
    at the method's own decision threshold."""
    gx, gy = np.meshgrid(score_grid(xlim), score_grid(ylim))
    grid = rule_scores(rule, gx.ravel(), gy.ravel(), heads, input_space)
    if grid is None:
        return
    grid = grid.reshape(gx.shape)
    ax.contourf(gx, gy, grid, levels=[-1, threshold, 2],
                colors=["#f6d5d5", "#d9e8f5"], alpha=0.7)
    ax.contour(gx, gy, grid, levels=[threshold], colors="k", linewidths=1.3)


def panel(ax, label, img_val, txt_val, method, heads, pred, input_space,
          ff_linear=False, show_region=True, ablation=False, bounds=None, dataset=None,
          path_from=None):
    """``path_from``: (clean image, clean text) aligned with img_val/txt_val; a
    dashed segment is drawn from each clean position to the attacked one."""
    name = LABELS.get(method, method)
    if path_from is not None and not (ff_linear and method == "feature-fusion"):
        path_from = tuple(np.clip(v, EPS, 1 - EPS) for v in path_from)

    if ff_linear and method == "feature-fusion":
        # Linear axes zoomed to the data: the text-only scores sit in a narrow
        # band that 0-1 axes flatten into a line.
        def zoom(values, margin=0.05):
            lo, hi = float(np.min(values)), float(np.max(values))
            pad = margin * max(hi - lo, 1e-3)
            return max(lo - pad, 0.0), min(hi + pad, 1.0)
        xs = img_val if path_from is None else np.concatenate([img_val, path_from[0]])
        ys = txt_val if path_from is None else np.concatenate([txt_val, path_from[1]])
        xlim, ylim = zoom(xs), zoom(ys)
    elif ablation:
        # Ablation view: logit scale, zoomed to the range the rule can reach.
        img_val = np.clip(img_val, EPS, 1 - EPS)
        txt_val = np.clip(txt_val, EPS, 1 - EPS)
        ax.set_xscale("logit")
        ax.set_yscale("logit")
        ax.minorticks_off()
        bounds = bounds if bounds is not None else (LIMITS, LIMITS)
        # Include the clean end of each path so no segment is cut off.
        x_all = img_val if path_from is None else np.concatenate([img_val, path_from[0]])
        y_all = txt_val if path_from is None else np.concatenate([txt_val, path_from[1]])
        xlim, ylim = (fit_logit_limits(v, b) for v, b in ((x_all, bounds[0]), (y_all, bounds[1])))
        for set_ticks, lim in ((ax.set_xticks, xlim), (ax.set_yticks, ylim)):
            ticks = spaced_logit_ticks(lim)
            set_ticks(ticks, [f"{t:g}" for t in ticks])
    else:
        img_val = np.clip(img_val, EPS, 1 - EPS)
        txt_val = np.clip(txt_val, EPS, 1 - EPS)
        ax.set_xscale("logit")
        ax.set_yscale("logit")
        ax.minorticks_off()
        xlim = ylim = LIMITS
        ax.set_xticks(TICKS, TICK_LABELS)
        ax.set_yticks(TICKS, TICK_LABELS)

    if method in LATE_RULES and show_region:
        # Same background/boundary as the unimodal view: the rule applied to
        # the two axis values as (image score, text score).
        decision_region(ax, method, heads, xlim, ylim, input_space,
                        threshold=method_threshold(method, dataset) if dataset else THRESHOLD)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    real = label == 1
    fake = label == 0
    if path_from is not None:
        moved = (path_from[0] != img_val) | (path_from[1] != txt_val)
        segments = np.stack([np.column_stack([path_from[0], path_from[1]]),
                             np.column_stack([img_val, txt_val])], axis=1)[moved]
        colors = np.where(real[moved], "#1f77b4", "#d62728")
        ax.add_collection(LineCollection(segments, colors=colors, linestyles="--",
                                         linewidths=0.6, alpha=0.45, zorder=1))
    ax.scatter(img_val[real], txt_val[real], s=15 * MARKER_SCALE, c="#1f77b4",
               label="Real", alpha=0.75, edgecolors="none")
    ax.scatter(img_val[fake], txt_val[fake], s=20 * MARKER_SCALE, c="#d62728",
               label="Fake", alpha=0.9, marker="^", edgecolors="none")

    # Count errors from the pipeline's own predictions, not a re-derivation.
    wrong = int((pred != label).sum())
    ax.set_title(f"{name}\n{wrong} of {len(label)} misclassified", fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--result-root", default=None,
                        help="Override result root (default: derived from config)")
    parser.add_argument("--out-dir", default=FIG_DIR)
    parser.add_argument("--paper-dir", default=None)
    parser.add_argument("--axes", choices=("unimodal", "ablation"), default="unimodal",
                        help="unimodal: late fusion on its two unimodal scores (with "
                             "decision regions); ablation: every method on its "
                             "image-only / text-only ablation scores")
    parser.add_argument("--path", action="store_true",
                        help="In attacked figures, draw a dashed line from each "
                             "sample's clean position to its attacked position")
    parser.add_argument("--unique", action="store_true",
                        help="In attacked figures, keep only samples flipped by that "
                             "attack and by none of the other attacks")
    parser.add_argument("--column", action="store_true",
                        help="Single-column paper layout: smaller canvas, larger fonts, "
                             "fewer ticks, no figure title (the caption names it)")
    parser.add_argument("--score-ticks", action="store_true",
                        help="Label the logit-scaled axes in score units instead of "
                             "logit units log(p/(1-p))")
    parser.add_argument("--ff-linear", action="store_true",
                        help="Draw the feature-fusion panel on "
                             "linear 0-1 axes instead of the logit scale")
    args = parser.parse_args()
    if not args.score_ticks:
        global TICKS, TICK_LABELS
        if args.column:
            LOGIT_TICKS[:] = [-10, 0, 10]
        TICKS = list(sigmoid(np.array(LOGIT_TICKS, dtype=float)))
        TICK_LABELS = [f"{k:g}" for k in LOGIT_TICKS]
    if args.column:
        global MARKER_SCALE
        MARKER_SCALE = 0.45

    result_root = args.result_root or dataset_result_root(args.dataset)
    out_dir = os.path.join(args.out_dir, f"{args.axes}_scores", args.dataset)

    def axes_for(df, method, ablation):
        if args.axes == "ablation":
            return get_ablation_axes(df, method, ablation, args.dataset)
        return get_axes(df, method, ablation)
    if not os.path.isdir(result_root):
        print(f"No results at {result_root}")
        return

    # Learned heads are shaded with the same differentiable modules the
    # pipeline scores with (SVC.predict_proba deviates by up to ~0.005).
    from models.fusion import fusion_head_path
    heads = {rule: TorchHead(rule, args.dataset) for rule in ("svm-rbf", "linear")
             if os.path.isfile(fusion_head_path(rule, args.dataset))}
    input_space = head_input_space(args.dataset)
    ff_ablation = load_ff_ablation(result_root)

    # Discover which scenarios have results
    scenarios = ["clean"]
    perturbed_root = os.path.join(result_root, "perturbed")
    for scenario_name, attack_dir in ATTACK_DIRS.items():
        attack_path = os.path.join(perturbed_root, attack_dir)
        if os.path.isdir(attack_path):
            scenarios.append(scenario_name)

    if not scenarios:
        print("No scenarios found")
        return

    # Fonts in points at the saved size; --column is drawn at about twice the
    # printed column width, so 13 pt prints at about 7 pt.
    fs = ({"tick": 13, "title": 13, "label": 15, "legend": 12} if args.column
          else {"tick": 12, "title": 11, "label": 12, "legend": 10})
    plt.rcParams.update({"font.size": fs["tick"], "axes.titlesize": fs["title"]})
    os.makedirs(out_dir, exist_ok=True)

    attacked_scenarios = [sc for sc in scenarios if sc != "clean"]
    unique = ({m: unique_flips(result_root, m, attacked_scenarios) for m in METHODS}
              if args.unique else None)

    for scenario in scenarios:
        if args.unique and scenario == "clean":
            continue
        # --column: shared axes, panels labelled (a)-(f); methods and thresholds
        # go in the caption, so the panels can sit next to each other.
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9) if args.column else (13.5, 8.6),
                                 sharex=args.column, sharey=args.column)

        for ax, method in zip(axes.ravel(), METHODS):
            if scenario == "clean":
                df = load_clean(result_root, method)
            else:
                df = load_perturbed(result_root, ATTACK_DIRS[scenario], method)

            # The modality ablation was run on clean inputs only.
            ablation = ff_ablation if scenario == "clean" else None
            data = axes_for(df, method, ablation) if df is not None else None

            if data is None:
                msg = "not available"
                title = LABELS.get(method, method)
                if df is not None and "pred" in df.columns:
                    if scenario != "clean":
                        title = attacked_title(
                            title, attack_asr(result_root, method, ATTACK_DIRS[scenario]))
                    msg = "no per-modality outputs\n(re-run attack)"
                ax.text(0.5, 0.5, msg, ha="center", va="center",
                        transform=ax.transAxes, color="0.5")
                ax.set_title(title, fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            label, img_val, txt_val = data
            path_from = None
            if args.path and scenario != "clean":
                clean_df = load_clean(result_root, method)
                clean = axes_for(clean_df, method, ff_ablation) if clean_df is not None else None
                if clean is not None:
                    # Align clean positions to the attacked samples by sample id.
                    pos = pd.Series(np.arange(len(clean[0])),
                                    index=sample_index(clean_df, method, ff_ablation))
                    rows = pos.reindex(sample_index(df, method)).values
                    if not np.isnan(rows).any():
                        rows = rows.astype(int)
                        path_from = (clean[1][rows], clean[2][rows])
            pred = df["pred"].values
            if unique is not None:
                keep = np.isin(sample_index(df, method), list(unique[method].get(scenario, ())))
                label, img_val, txt_val, pred = label[keep], img_val[keep], txt_val[keep], pred[keep]
                if path_from is not None:
                    path_from = (path_from[0][keep], path_from[1][keep])
                if not keep.any():
                    ax.text(0.5, 0.5, "no sample flipped\nonly by this attack", ha="center",
                            va="center", transform=ax.transAxes, color="0.5")
                    ax.set_title(attacked_title(
                        LABELS.get(method, method),
                        attack_asr(result_root, method, ATTACK_DIRS[scenario]), 0), fontsize=11)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
            if method == "feature-fusion" and ablation is not None:
                pred = ablation.set_index("index").index.map(
                    df.set_index("index")["pred"]).values
            panel(ax, label, img_val, txt_val, method, heads, pred, input_space,
                  args.ff_linear,
                  show_region=True,
                  ablation=(args.axes == "ablation"),
                  bounds=(ablation_bounds(method, args.dataset)
                          if args.axes == "ablation" else None),
                  dataset=args.dataset, path_from=path_from)
            if scenario == "clean":
                # Clean panels: method and its decision threshold, no metrics.
                t = method_threshold(method, args.dataset)
                if args.column:
                    ax.set_title("")
                    k = list(axes.ravel()).index(ax)
                    ax.text(0.04, 0.96, f"({'abcdef'[k]})", transform=ax.transAxes, ha="left",
                            va="top", fontsize=fs["title"],
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
                else:
                    ax.set_title(f"{LABELS.get(method, method)} ($t$ = {t:.5g})", fontsize=fs["title"])
            else:
                ax.set_title(attacked_title(
                    LABELS.get(method, method),
                    attack_asr(result_root, method, ATTACK_DIRS[scenario]),
                    len(label) if unique is not None else None), fontsize=11)

        for ax in axes.ravel():
            if ax.get_legend_handles_labels()[0]:
                if args.column:
                    # Above the panels, so it hides no sample.
                    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
                               bbox_to_anchor=(0.5, 0.97), frameon=False, fontsize=fs["legend"],
                               markerscale=2.0)
                else:
                    ax.legend(loc="lower left", frameon=True, fontsize=fs["legend"])
                break

        if args.axes == "ablation":
            fig.supxlabel("image-only score (text missing)")
            fig.supylabel("text-only score (image missing)")
        else:
            unit = "score" if args.score_ticks else "logit"
            fig.supxlabel(f"image {unit}", fontsize=fs["label"])
            fig.supylabel(f"text {unit}", fontsize=fs["label"])
        title = TITLES.get(scenario, scenario)
        space = "score space" if args.score_ticks else "logit space"
        kind = f"Ablation {space}" if args.axes == "ablation" else space.capitalize()
        # Clean figure: no scenario suffix; attacked figures name the attack.
        suffix = "" if scenario == "clean" else f" -- {title}"
        if not args.column:
            fig.suptitle(f"{args.dataset}: {kind} by fusion method{suffix}", fontsize=15)
        if args.column:
            # Axes only on the outer panels: no ticks between neighbouring panels.
            for ax in axes.ravel():
                ax.tick_params(bottom=ax.get_subplotspec().is_last_row(),
                               left=ax.get_subplotspec().is_first_col())
        fig.tight_layout()
        if args.column:
            fig.subplots_adjust(wspace=0.05, hspace=0.05)

        stem = f"{args.dataset}_score_space_{scenario.replace('+', '_').lower()}"
        if args.unique:
            stem += "_unique"
        if args.path and scenario != "clean":
            stem += "_paths"
        if args.score_ticks:
            stem += "_scoreticks"
        if args.column:
            stem += "_column"
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"),
                        dpi=200, bbox_inches="tight")
        if args.paper_dir:
            os.makedirs(args.paper_dir, exist_ok=True)
            fig.savefig(os.path.join(args.paper_dir, f"{stem}.pdf"),
                        bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {os.path.join(out_dir, stem)}.{{png,pdf}}")


if __name__ == "__main__":
    main()
