"""Metrics with 95% bootstrap confidence intervals, resampled by scene.

Correctness is `present == gt_present`, so the same metrics apply to positive sequences
(pedestrian present on every frame) and to control sequences (no pedestrian on any frame).
For a sequence with evidence frame k (positives: the first clearly visible frame; controls and
planted modes: the first frame the model answers after the planted turns):
- wrong_before: the answer on frame k-1, the belief held when evidence arrives, is wrong.
                On positives this is an initial miss, often a fair call on an occluded frame;
                in planted modes it is wrong by construction.
- persisted:    wrong_before and still wrong on frame k+1, after two contradictory frames.
- latency:      frames from k to the first correct answer (0 = fixed on the evidence frame);
                None if never corrected within the sequence.
Planted frames (fixed answers written into the history, not model output) are left out of
every per-frame metric.
"""
import math
import random
from statistics import mean, median

CAUTION = {"proceed": 0, "slow": 1, "stop": 2}
PAIRED_MODES = ("stateful", "reconsider", "planted_no", "planted_yes")


def scene_of(seq_id):
    return seq_id.split("_")[0]


def answered(frames):
    return [f for f in frames if not f.get("planted")]


def is_correct(f):
    return f["present"] == f["gt_present"]


def sequence_stats(rec):
    k, frames = rec["evidence_index"], rec["frames"]
    correct = [is_correct(f) for f in frames]
    wrong = not correct[k - 1]
    latency = next((t - k for t in range(k, len(frames)) if correct[t]), None) if wrong else None
    return {"id": rec["id"], "wrong_before": wrong, "persisted": wrong and not correct[k + 1],
            "latency": latency, "post_acc": mean(correct[k:]),
            "frame_acc": mean(is_correct(f) for f in answered(frames))}


def bootstrap_ci(values, n_boot, rng, groups=None):
    """95% percentile bootstrap CI of the mean of values.

    With groups (one label per value), whole groups are resampled: sequences from the same scene
    share frames and conditions, so they are not independent.
    """
    if not values:
        return None
    clusters = {}
    for i, v in enumerate(values):
        clusters.setdefault(groups[i] if groups else i, []).append(v)
    clusters = list(clusters.values())
    means = []
    for _ in range(n_boot):
        pick = rng.choices(clusters, k=len(clusters))
        means.append(sum(map(sum, pick)) / sum(map(len, pick)))
    means.sort()
    return [means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def safety(frames):
    gt_classes = {f["gt_action"] for f in frames}
    recall = [mean(f["action"] == c for f in frames if f["gt_action"] == c) for c in gt_classes]
    unsafe = [f["action"] is None or CAUTION[f["action"]] < CAUTION[f["gt_action"]] for f in frames]
    return mean(recall), mean(unsafe)


def switches(recs):
    """Answer changes between consecutive frames. A planted answer counts as the previous answer
    for the first frame the model answers, but changes between two planted frames do not count."""
    out = {"yes_to_no": 0, "no_to_yes": 0}
    for r in recs:
        f = r["frames"]
        for t in range(1, len(f)):
            if f[t].get("planted"):
                continue
            if f[t - 1]["present"] is True and f[t]["present"] is False:
                out["yes_to_no"] += 1
            elif f[t - 1]["present"] is False and f[t]["present"] is True:
                out["no_to_yes"] += 1
    return out


def by_offset(recs):
    """Accuracy at each frame offset from k (0 = evidence frame), with the number of sequences."""
    by = {}
    for r in recs:
        for t in range(r["evidence_index"], len(r["frames"])):
            by.setdefault(t - r["evidence_index"], []).append(is_correct(r["frames"][t]))
    return {d: {"accuracy": mean(v), "n": len(v)} for d, v in sorted(by.items())}


def by_k(recs):
    """Evidence-frame accuracy by k (= the number of planted turns in planted modes)."""
    by = {}
    for r in recs:
        by.setdefault(r["evidence_index"], []).append(is_correct(r["frames"][r["evidence_index"]]))
    return {k: {"accuracy": mean(v), "n": len(v)} for k, v in sorted(by.items())}


def mode_summary(recs, n_boot, rng):
    stats = [sequence_stats(r) for r in recs]
    scenes = [scene_of(s["id"]) for s in stats]
    wrong = [(s, g) for s, g in zip(stats, scenes) if s["wrong_before"]]
    lat = sorted(s["latency"] for s, _ in wrong if s["latency"] is not None)
    persisted = [float(s["persisted"]) for s, _ in wrong]
    frames = [f for r in recs for f in answered(r["frames"])]
    bal_acc, unsafe = safety(frames)
    out = {
        "n_sequences": len(stats),
        "n_wrong_before": len(wrong),
        "persistence_rate": mean(persisted) if wrong else None,
        "persistence_rate_ci": bootstrap_ci(persisted, n_boot, rng, [g for _, g in wrong]),
        "latency_median": median(lat) if lat else None,
        "latency_p90": lat[math.ceil(0.9 * len(lat)) - 1] if lat else None,
        "n_never_corrected": len(wrong) - len(lat),
        "post_evidence_accuracy": mean(s["post_acc"] for s in stats),
        "post_evidence_accuracy_ci": bootstrap_ci([s["post_acc"] for s in stats], n_boot, rng, scenes),
        "frame_accuracy": mean(s["frame_acc"] for s in stats),
        "says_present_rate": mean(f["present"] is True for f in frames),
        "safety_balanced_accuracy": bal_acc,
        "unsafe_action_rate": unsafe,
        "switches": switches(recs),
        "accuracy_by_offset": by_offset(recs),
        "evidence_accuracy_by_k": by_k(recs),
        "n_parsed_from_text": sum(f.get("parsed_from") == "text" for f in frames),
        "n_unparsed": sum(f["present"] is None for f in frames),
    }
    # profiling fields exist only for runs made after they were added
    for key, stat in (("latency_s", median), ("prompt_tokens", max), ("peak_gb", max)):
        vals = [f[key] for f in frames if f.get(key) is not None]
        if vals:
            out[f"{key}_{stat.__name__}"] = stat(vals)
    return out


def paired_summary(recs, per_mode, n_boot, rng):
    """Comparisons with stateless on the same sequences: the same frames, with vs without history."""
    out = {}
    by = {(r["id"], r["mode"]): r for r in recs}
    ids = sorted({r["id"] for r in recs if r["mode"] == "stateless"})
    others = [m for m in PAIRED_MODES if m in per_mode]

    if "stateful" in per_mode:
        pairs = [i for i in ids if (i, "stateful") in by]
        diffs = [sequence_stats(by[(i, "stateful")])["post_acc"] - sequence_stats(by[(i, "stateless")])["post_acc"]
                 for i in pairs]
        out["stateful_minus_stateless_post_acc"] = {
            "mean": mean(diffs) if diffs else None, "n_pairs": len(diffs),
            "ci": bootstrap_ci(diffs, n_boot, rng, [scene_of(i) for i in pairs])}

    # evidence frames the model answers correctly without history: any error in another mode on
    # that same frame is caused by the history, not by perception
    at_k = lambda i, m: is_correct(by[(i, m)]["frames"][by[(i, m)]["evidence_index"]])
    ok = [i for i in ids if at_k(i, "stateless")]
    cwh = {"n": len(ok)}
    for m in others:
        sub = [i for i in ok if (i, m) in by]
        hits = [float(at_k(i, m)) for i in sub]
        cwh[m] = {"evidence_frame_accuracy": mean(hits) if hits else None,
                  "ci": bootstrap_ci(hits, n_boot, rng, [scene_of(i) for i in sub])}
    out["correct_without_history"] = cwh

    # both directions on the evidence frame, not only the one the subset above conditions on
    out["mcnemar_evidence_frame"] = {}
    for m in others:
        pairs = [(at_k(i, "stateless"), at_k(i, m)) for i in ids if (i, m) in by]
        b = sum(s and not o for s, o in pairs)
        c = sum(o and not s for s, o in pairs)
        out["mcnemar_evidence_frame"][m] = {"n_pairs": len(pairs), "stateless_only": b, "mode_only": c,
                                            "p_exact": mcnemar_exact(b, c)}

    # persistence on the sequences both modes get wrong at k-1, so the rates share a denominator
    out["persistence_matched"] = {}
    for m in ("stateful", "reconsider"):
        if m not in per_mode:
            continue
        st = {mm: {i: sequence_stats(by[(i, mm)]) for i in ids if (i, mm) in by} for mm in ("stateless", m)}
        both = [i for i in st[m] if st["stateless"][i]["wrong_before"] and st[m][i]["wrong_before"]]
        out["persistence_matched"][m] = {
            "n": len(both), **{mm: mean(st[mm][i]["persisted"] for i in both) if both else None
                               for mm in ("stateless", m)}}
    return out


def summarize_kind(recs, n_boot, rng):
    per_mode = {m: mode_summary([r for r in recs if r["mode"] == m], n_boot, rng)
                for m in sorted({r["mode"] for r in recs})}
    if "stateless" in per_mode:
        per_mode.update(paired_summary(recs, per_mode, n_boot, rng))
    return per_mode


def summarize(results, n_boot=1000, seed=0):
    """Per model: positive-sequence metrics at the top level (visible_without_history is the
    headline metric), control-sequence metrics under "controls" (empty_without_history: empty
    frames the model gets right without history, and how many it still gets right with a
    planted "yes" in the history)."""
    rng = random.Random(seed)
    out = {}
    for model in sorted({r["model"] for r in results}):
        recs = [r for r in results if r["model"] == model]
        pos = [r for r in recs if r.get("kind", "positive") == "positive"]
        ctrl = [r for r in recs if r.get("kind") == "control"]
        s = summarize_kind(pos, n_boot, rng) if pos else {}
        if "correct_without_history" in s:
            s["visible_without_history"] = s.pop("correct_without_history")
        if ctrl:
            c = summarize_kind(ctrl, n_boot, rng)
            if "correct_without_history" in c:
                c["empty_without_history"] = c.pop("correct_without_history")
            s["controls"] = c
        out[model] = s
    return out
