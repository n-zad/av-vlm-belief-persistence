"""Run a VLM over mined sequences in stateless, stateful or reconsider mode.

stateless:  each frame is asked on its own.
stateful:   one conversation; the model sees its own earlier answers.
reconsider: stateful, plus a nudge to judge each new frame on its own (the mitigation).
"""
import json
import os
import re

MODES = ("stateless", "stateful", "reconsider")

PROMPT = (
    "You are the perception module of an autonomous vehicle. The image is from the front camera.\n"
    "Is there a {object} in the image?\n"
    'Answer with JSON only: {{"present": "yes" or "no", "confidence": 0.0 to 1.0, '
    '"evidence": "<short phrase>", "action": "proceed" or "slow" or "stop"}}'
)
RECONSIDER = "This is a new frame. Your earlier answers may have been wrong; judge only from this image.\n"


def parse(raw):
    """Pull present/confidence/action out of a reply; None for anything missing."""
    d = {}
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group())
        except json.JSONDecodeError:
            pass
    present = str(d.get("present", "")).strip().lower()
    if present not in ("yes", "no", "true", "false"):
        hit = re.search(r"\b(yes|no)\b", raw, re.I)
        present = hit.group(1).lower() if hit else ""
    action = str(d.get("action", "")).strip().lower()
    if action not in ("proceed", "slow", "stop"):
        hit = re.search(r"\b(proceed|slow|stop)\b", raw, re.I)
        action = hit.group(1).lower() if hit else None
    try:
        confidence = float(d["confidence"])
    except (KeyError, TypeError, ValueError):
        confidence = None
    return {"present": {"yes": True, "true": True, "no": False, "false": False}.get(present),
            "confidence": confidence, "action": action}


def run_sequence(model, seq, mode, dataroot):
    history, frames = [], []
    for t, f in enumerate(seq["frames"]):
        text = PROMPT.format(object=seq["object"])
        if mode == "reconsider" and t > 0:
            text = RECONSIDER + text
        msg = {"role": "user", "content": [
            {"type": "image", "path": os.path.join(dataroot, f["image"])},
            {"type": "text", "text": text}]}
        raw = model.generate([msg] if mode == "stateless" else history + [msg])
        if mode != "stateless":
            history += [msg, {"role": "assistant", "content": [{"type": "text", "text": raw}]}]
        frames.append({"t": t, "raw": raw, **parse(raw),
                       "gt_present": f["gt_present"], "gt_action": f["gt_action"]})
    return {"id": seq["id"], "mode": mode, "evidence_index": seq["evidence_index"], "frames": frames}


def run(model, model_name, sequences, modes, dataroot, out_path):
    """Append one JSON line per (sequence, mode); pairs already in out_path are skipped,
    so an interrupted GPU run can be resumed."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {(r["id"], r["mode"]) for r in map(json.loads, f) if r["model"] == model_name}
    with open(out_path, "a", encoding="utf-8") as out:
        for seq in sequences:
            for mode in modes:
                if (seq["id"], mode) in done:
                    continue
                rec = {"model": model_name, **run_sequence(model, seq, mode, dataroot)}
                out.write(json.dumps(rec) + "\n")
                out.flush()
