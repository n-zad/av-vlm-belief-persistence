"""Report figures from results files. One panel per model, colour = mode in every chart."""
import os
from statistics import mean

from . import metrics

MODES = ("stateless", "stateful", "reconsider")
COLOR = {"stateless": "#2a78d6", "stateful": "#eb6834", "reconsider": "#1baf7a"}  # categorical slots 1-3
LABEL = {"stateless": "No history", "stateful": "With history", "reconsider": "With history + reconsider"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SIZE_BINS = ((0, 60, "< 60"), (60, 90, "60–89"), (90, 150, "90–149"), (150, float("inf"), "150+"))


def short(model):
    return model.split("/")[-1]


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK})
    return plt


def panels(plt, n, title, height=3.6):
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, height), sharey=True, squeeze=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    return fig, axes[0]


def bar_labels(ax, bars, texts, tops=None):
    for b, t, y in zip(bars, texts, tops or [b.get_height() for b in bars]):
        ax.annotate(t, (b.get_x() + b.get_width() / 2, y), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", color=INK2, fontsize=9)


def yes_by_frame(plt, by_model, out):
    fig, axes = panels(plt, len(by_model), "Share of answers saying a pedestrian is present")
    for ax, (model, recs) in zip(axes, by_model.items()):
        shown = set()
        for m in MODES:
            by = {}
            for r in recs:
                if r["mode"] == m:
                    for f in r["frames"]:
                        by.setdefault(f["t"] - r["evidence_index"], []).append(f["present"] is True)
            n_seq = sum(r["mode"] == m for r in recs)
            # offsets that only a few sequences reach are noise
            xs = [d for d in sorted(by) if len(by[d]) >= min(20, n_seq)]
            if not xs:
                continue
            shown.update(xs)
            ys = [100 * mean(by[d]) for d in xs]
            ax.plot(xs, ys, color=COLOR[m], lw=2, marker="o", ms=6, label=LABEL[m])
            ax.annotate(f"{ys[-1]:.0f}%", (xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points",
                        va="center", color=INK2, fontsize=9)
        ax.axvline(0, color=INK2, lw=1, ls="--")
        ax.annotate("first clear frame", (0, 97), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="top")
        ax.set(title=short(model), xlabel="Frames relative to the first clear frame", ylim=(0, 100))
        ticks = range(min(shown), max(shown) + 1) if shown else [0]
        ax.set_xticks(ticks, [f"{d:+d}" if d else "0" for d in ticks])
    axes[0].set_ylabel("Says “present” (%)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "yes_by_frame.png"), dpi=200)


def visible_without_history(plt, summary, out):
    fig, axes = panels(plt, len(summary), "Seen without history: still seen with history?")
    for ax, (model, s) in zip(axes, summary.items()):
        v = s.get("visible_without_history", {})
        modes = [m for m in MODES if m == "stateless" or m in v]
        vals = [100.0 if m == "stateless" else 100 * v[m]["evidence_frame_accuracy"] for m in modes]
        bars = ax.bar(range(len(modes)), vals, width=0.6, color=[COLOR[m] for m in modes])
        tops = list(vals)
        for i, m in enumerate(modes):
            if m != "stateless" and v[m]["ci"]:
                lo, hi = (100 * c for c in v[m]["ci"])
                ax.errorbar(i, vals[i], yerr=[[vals[i] - lo], [hi - vals[i]]], color=INK, lw=1.2, capsize=4)
                tops[i] = hi  # label above the whisker, not on it
        bar_labels(ax, bars, [f"{x:.0f}%" for x in vals], tops)
        ax.set_xticks(range(len(modes)), [LABEL[m] for m in modes], fontsize=9)
        ax.set(title=f"{short(model)} (n = {v.get('n', 0)} sequences)", ylim=(0, 110))
    axes[0].set_ylabel("Says “present” on the clear frame (%)")
    fig.text(0.01, 0.01, "Error bars: 95% bootstrap CI. “No history” is 100% by construction (it defines the subset).",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(os.path.join(out, "visible_without_history.png"), dpi=200)


def by_size(plt, by_model, seqs, out):
    fig, axes = panels(plt, len(by_model), "Seen without history, by pedestrian height")
    for ax, (model, recs) in zip(axes, by_model.items()):
        hits = {label: [] for _, _, label in SIZE_BINS}
        for r in recs:
            if r["mode"] == "stateless" and r["id"] in seqs:
                k = r["evidence_index"]
                h = seqs[r["id"]]["frames"][k]["px_height"]
                label = next(lab for lo, hi, lab in SIZE_BINS if lo <= h < hi)
                hits[label].append(r["frames"][k]["present"] is True)
        labels = [lab for lab in hits if hits[lab]]
        vals = [100 * mean(hits[lab]) for lab in labels]
        bars = ax.bar(range(len(labels)), vals, width=0.6, color=COLOR["stateless"])
        bar_labels(ax, bars, [f"{v:.0f}%\n(n={len(hits[lab])})" for v, lab in zip(vals, labels)])
        ax.set_xticks(range(len(labels)), labels)
        ax.set(title=short(model), xlabel="Pedestrian height on the clear frame (px, 1600×900 image)", ylim=(0, 120))
    axes[0].set_ylabel("Says “present” (%)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "seen_by_size.png"), dpi=200)


def copying(plt, by_model, out):
    fig, axes = panels(plt, len(by_model), "Answers that repeat the previous answer word for word")
    for ax, (model, recs) in zip(axes, by_model.items()):
        modes = [m for m in MODES if any(r["mode"] == m for r in recs)]
        vals = [100 * mean(r["frames"][t]["raw"] == r["frames"][t - 1]["raw"]
                           for r in recs if r["mode"] == m for t in range(1, len(r["frames"]))) for m in modes]
        bars = ax.bar(range(len(modes)), vals, width=0.6, color=[COLOR[m] for m in modes])
        bar_labels(ax, bars, [f"{v:.0f}%" for v in vals])
        ax.set_xticks(range(len(modes)), [LABEL[m] for m in modes], fontsize=9)
        ax.set(title=short(model), ylim=(0, 110))
    axes[0].set_ylabel("Identical to previous answer (%)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "copying.png"), dpi=200)


def examples(plt, model, recs, seqs, dataroot, out, n=2, crop=(560, 315)):
    """Sequences where the model sees the clear frame without history but not with it."""
    from PIL import Image
    r_by = {(r["id"], r["mode"]): r for r in recs}
    picks = [i for i in seqs if (i, "stateless") in r_by and (i, "stateful") in r_by
             # still seen one frame later too, so the example isn't re-occluded right after
             and all(r_by[(i, "stateless")]["frames"][seqs[i]["evidence_index"] + d]["present"] is True for d in (0, 1))
             and r_by[(i, "stateful")]["frames"][seqs[i]["evidence_index"]]["present"] is not True]
    picks = sorted(picks, key=lambda i: -seqs[i]["frames"][seqs[i]["evidence_index"]]["px_height"])[:n]
    if not picks:
        return None
    cols = len(seqs[picks[0]]["frames"])
    fig, axes = plt.subplots(len(picks), cols, figsize=(2.3 * cols, 2.6 * len(picks)), squeeze=False)
    fig.suptitle(f"{short(model)}: pedestrian (circled) seen without history, missed with it — crops around the pedestrian",
                 x=0.01, ha="left", fontsize=11)
    say = lambda f: {True: "yes", False: "no", None: "?"}[f["present"]]
    for row, i in zip(axes, picks):
        s, k = seqs[i], seqs[i]["evidence_index"]
        for t, (ax, f) in enumerate(zip(row, s["frames"])):
            img = Image.open(os.path.join(dataroot, f["image"]))
            x0 = min(max(f["u"] - crop[0] // 2, 0), img.width - crop[0])
            y0 = min(max(f["v"] - crop[1] // 2, 0), img.height - crop[1])
            ax.imshow(img.crop((x0, y0, x0 + crop[0], y0 + crop[1])))
            rad = max(0.6 * f["px_height"], 18)
            ax.add_patch(plt.Circle((f["u"] - x0, f["v"] - y0), rad, fill=False, color="#e34948", lw=1.5))
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(t == k)
                side.set_color(INK), side.set_linewidth(2)
            ax.set_title(f"{'first clear frame' if t == k else f'frame {t - k:+d}'}", fontsize=8,
                         color=INK if t == k else INK2)
            ax.set_xlabel(f"No history: {say(r_by[(i, 'stateless')]['frames'][t])}\n"
                          f"With history: {say(r_by[(i, 'stateful')]['frames'][t])}", fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out, f"examples_{short(model)}.png")
    fig.savefig(path, dpi=200)
    return path


def make_all(results, seqs, out, dataroot=None):
    plt = setup()
    os.makedirs(out, exist_ok=True)
    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)
    yes_by_frame(plt, by_model, out)
    visible_without_history(plt, metrics.summarize(results), out)
    by_size(plt, by_model, seqs, out)
    copying(plt, by_model, out)
    if dataroot:
        for model, recs in by_model.items():
            examples(plt, model, recs, seqs, dataroot, out)
    plt.close("all")
    return sorted(os.listdir(out))
