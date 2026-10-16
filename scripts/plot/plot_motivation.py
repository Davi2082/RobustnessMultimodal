"""Motivating schematic for the Introduction (same example as the text).

One model, one noisy decision boundary between "real" and "fake", in the plane
spanned by an image and a text direction. Six correctly classified (real)
samples a-f are attacked: text moves up, image moves right, the union
attack applies the two independently found moves together, and the joint
attack searches both modalities jointly and may pick any direction.

* text alone flips a and b; image alone flips b, c, d;
* e, f resist both unimodal attacks (and so the union of their successes),
  but the union and the joint attack both flip them: each unimodal move stops
  short of the boundary, their combination crosses it;
* a few samples on both sides are not attacked.

The boundary is a quarter ellipse from the text axis to the image axis with
smooth multi-scale noise along its whole length. Samples are placed against it
by _build(); check() asserts the example's sets.

Usage:
    python3 scripts/plot/plot_motivation.py [--out-dir figures/motivation]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.path import Path as MplPath
import numpy as np

TEXT_C, IMAGE_C, JOINT_C, FAKE_C, INK, GREY = "#e8730c", "#2a78d6", "#6b3fa0", "#d62728", "#222222", "#9a9a9a"
ADV_C = "#c00000"                        # adversarial sample misclassified as real

A, B = 10.0, 6.25                        # quarter ellipse from (0, B) to (A, 0)
XMAX, YMAX = 10.8, 7.0
SEED = 1
NOISE = [(3.8, 0.35), (2.1, 0.28), (1.25, 0.2), (0.75, 0.08)]   # (wavelength, amplitude)
GAP = 0.35                               # how far an arrow ends from the boundary
HIT = 0.75                               # how far a successful attack lands inside the real region
# attacked (fake) samples: (position along the arc [deg, 90 = text axis], depth inside)
PLACEMENT = {"a": (80, 0.7), "e": (62, 2.0), "b": (47, 0.55), "f": (33, 2.0), "c": (22, 0.75), "d": (7, 0.9)}
FLIPS = {"a": {"text"}, "b": {"text", "image"}, "c": {"image"}, "d": {"image"}, "e": set(), "f": set()}
ROBUST = ("e", "f")
# unattacked samples: on the fake side (inside the arc) and on the real side
BACKGROUND = {"fake": [(1.6, 1.4), (4.5, 2.4), (3.0, 4.1), (7.0, 0.9)],
              "real": [(9.9, 5.75), (5.75, 6.5), (10.4, 2.9), (2.0, 6.7)]}


def noisy_arc(seed=SEED, n=1500):
    rng = np.random.default_rng(seed)
    t = np.linspace(np.pi / 2, 0, n)
    base = np.column_stack([A * np.cos(t), B * np.sin(t)])
    d = np.gradient(base, axis=0)
    nrm = np.column_stack([-d[:, 1], d[:, 0]]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(base, axis=0), axis=1))])
    noise = sum(a * np.sin(2 * np.pi * s / lam + rng.uniform(0, 2 * np.pi)) for lam, a in NOISE)
    return base + (noise * np.sin(np.pi * s / s[-1]) ** 0.6)[:, None] * nrm, base, nrm


_PTS, _BASE, _BNRM = noisy_arc()
_PATH = MplPath(np.vstack([_PTS, [(XMAX + 1, 0), (XMAX + 1, YMAX + 1), (0, YMAX + 1)]]))


def boundary():
    return [tuple(q) for q in _PTS]


def flipped(x, y):
    """True on the real side (outside the arc)."""
    return _PATH.contains_point((x, y))


def reach(p, u):
    """Distance from p along direction u to the boundary."""
    for r in np.arange(0.02, 6, 0.02):
        if flipped(*(p + r * u)):
            return r
    return 6.0


def _build():
    samples, attacks = {}, {}
    for name, (deg, depth) in PLACEMENT.items():
        i = int(round((90 - deg) / 90 * (len(_BASE) - 1)))
        p = _BASE[i] - depth * _BNRM[i]
        ends = {}
        for mod, u in (("image", np.array([1.0, 0.0])), ("text", np.array([0.0, 1.0]))):
            r = reach(p, u)
            length = r + HIT if mod in FLIPS[name] else min(r - GAP, 1.6)
            ends[mod] = p + length * u
        if name in ROBUST:
            # union: both partial unimodal moves applied together; it must cross
            # the boundary (with some margin) although neither move does alone
            corner = ends["image"] + ends["text"] - p
            ends["union"] = corner
            # joint: the shortest way out, among directions clearly different
            # from the text, image and union directions
            tu = np.degrees(np.arctan2(*(corner - p)[::-1]))
            angles = [a for a in np.arange(12, 79, 1.0) if abs(a - tu) > 14]
            best = min(angles, key=lambda a: reach(p, np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])))
            u = np.array([np.cos(np.radians(best)), np.sin(np.radians(best))])
            ends["joint"] = p + (reach(p, u) + HIT) * u
        samples[name] = tuple(p)
        attacks[name] = {k: tuple(v) for k, v in ends.items()}
    return samples, attacks


SAMPLES, ATTACKS = _build()


def check():
    for n, ends in ATTACKS.items():
        assert {m for m in ("text", "image") if flipped(*ends[m])} == FLIPS[n], n
        if n in ROBUST:
            assert flipped(*ends["union"]) and flipped(*ends["joint"]), n
    for side, pts in BACKGROUND.items():
        for q in pts:
            assert flipped(*q) == (side == "real"), (side, q)


def arrow(ax, a, b, color, lw=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=5, color=color,
                                 linewidth=lw, linestyle=ls, shrinkA=2.5, shrinkB=2.5))


def draw(ax, panel):
    """panel: "unimodal" (text + image attacks) or "joint" (union + joint attacks on e, f)."""
    ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX); ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("image", fontsize=7, labelpad=4)
    ax.set_ylabel("text", fontsize=7, labelpad=0)

    b = boundary()
    ax.add_patch(Polygon(b + [(XMAX, 0), (XMAX, YMAX), (0, YMAX)], closed=True,
                         facecolor=FAKE_C, alpha=0.10, edgecolor="none"))
    ax.plot(*zip(*b), color=INK, linewidth=0.9)
    ax.text(XMAX - 0.12, YMAX - 0.12, "real", color=FAKE_C, fontsize=7, ha="right", va="top")
    ax.text(0.18, 0.18, "fake", color="#555555", fontsize=7, ha="left", va="bottom")
    for q in BACKGROUND["fake"]:
        ax.plot(*q, "^", color=GREY, markersize=2.6)
    for q in BACKGROUND["real"]:
        ax.plot(*q, "o", color=GREY, markersize=2.3)

    for name, p in SAMPLES.items():
        ends = ATTACKS[name]
        if panel == "unimodal":
            arrow(ax, p, ends["image"], IMAGE_C)
            arrow(ax, p, ends["text"], TEXT_C)
            for m in FLIPS[name]:                 # successful attack: adversarial sample, now classified real
                ax.plot(*ends[m], "^", color=ADV_C, markersize=3.4, zorder=6)
        elif name in ROBUST:
            c = ends["union"]
            ax.plot([p[0], ends["image"][0], c[0], ends["text"][0], p[0]],
                    [p[1], ends["image"][1], c[1], ends["text"][1], p[1]],
                    color="0.6", lw=0.6, ls=(0, (1.5, 1.5)))
            arrow(ax, p, c, JOINT_C, lw=0.8, ls=(0, (2, 1.2)))
            arrow(ax, p, ends["joint"], JOINT_C, lw=1.1)
            ax.plot(*c, "^", color=ADV_C, markersize=3.4, zorder=6)
            ax.plot(*ends["joint"], "^", color=ADV_C, markersize=3.4, zorder=6)
        ax.plot(*p, "^", color=INK, markersize=3.2, zorder=5)
        low = p[1] < 1.0                     # keep labels near the x axis above the point
        ax.text(p[0] - (0.36 if low else 0.24), p[1] + (0.0 if low else -0.18), name, fontsize=8, style="italic", ha="right",
                va="center" if low else "top",
                color=INK)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="figures/motivation")
    args = p.parse_args()

    check()
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "font.serif": ["STIXGeneral", "DejaVu Serif"]})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.72))
    draw(axes[0], "unimodal")
    draw(axes[1], "joint")
    axes[1].set_ylabel("")
    axes[0].set_title("(a) Unimodal attacks", fontsize=7.5, pad=2)
    axes[1].set_title("(b) Union and joint attacks", fontsize=7.5, pad=2)
    att = [Line2D([], [], color=TEXT_C, lw=1.0, label="text"),
           Line2D([], [], color=IMAGE_C, lw=1.0, label="image"),
           Line2D([], [], color=JOINT_C, lw=0.8, ls=(0, (2, 1.2)), label="union"),
           Line2D([], [], color=JOINT_C, lw=1.1, label="joint")]
    cls = [Line2D([], [], color=GREY, marker="o", ls="none", markersize=3.2, label="real"),
           Line2D([], [], color=INK, marker="^", ls="none", markersize=3.4, label="fake"),
           Line2D([], [], color=ADV_C, marker="^", ls="none", markersize=3.4, label="adversarial fake"),
           Line2D([], [], ls="none", label=" ")]
    handles = att + cls[:3]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=7, fontsize=6,
               frameon=False, handlelength=1.0, columnspacing=0.6, handletextpad=0.25, borderaxespad=0.1)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.91, bottom=0.16, wspace=0.08)
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"motivation.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {args.out_dir}/motivation.{{pdf,png}}")


if __name__ == "__main__":
    main()
