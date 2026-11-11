"""Mine nuScenes CAM_FRONT sequences where an object goes from barely visible to clearly visible.

Reads the raw nuScenes JSON tables directly, so the nuscenes-devkit is not required.

A sequence is `length` consecutive keyframes in which one target instance stays in the
front camera's view. The first frame has visibility level 1 (0-40% visible); the
"evidence" frame is the first one with level 4 (80-100%). Every frame before the evidence
frame is low visibility (level <= 2), and no other object of the same category is clearly
visible (level >= 3) before it, so the evidence frame is the first unambiguous view.
"""
import json
import os

import numpy as np

CAMERA = "CAM_FRONT"


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def to_camera(point, ego_pose, calib):
    """Global-frame point -> camera frame (x right, y down, z forward)."""
    p = quat_to_rot(ego_pose["rotation"]).T @ (np.asarray(point) - ego_pose["translation"])
    return quat_to_rot(calib["rotation"]).T @ (p - calib["translation"])


def in_image(p_cam, intrinsic, width, height):
    # shortcut: tests only the box centre, so objects cut off at the image edge may be
    # misjudged; project the 8 box corners if edge cases matter.
    if p_cam[2] <= 0.5:
        return False
    u, v, w = np.asarray(intrinsic) @ p_cam
    return 0 <= u / w < width and 0 <= v / w < height


def safe_action(objects):
    """Heuristic ground-truth action from in-view objects' camera-frame positions.

    shortcut: rule-based, not human-labelled; replace with manual labels for the final eval.
    """
    action = "proceed"
    for o in objects:
        dist = (o["x"] ** 2 + o["z"] ** 2) ** 0.5
        if dist < 15 and abs(o["x"]) < 2.5:
            return "stop"
        if dist < 15:
            action = "slow"
    return action


def load_tables(dataroot, version, category_prefix):
    """Load the tables, keeping only CAM_FRONT keyframes, their poses and target-category
    annotations; trainval's full sample_data/ego_pose/sample_annotation would need >10 GB RAM."""
    def table(name, keep=lambda r: True):
        with open(os.path.join(dataroot, version, name + ".json"), encoding="utf-8") as f:
            return {r["token"]: r for r in json.load(f) if keep(r)}

    t = {n: table(n) for n in ["scene", "sample", "instance", "category", "calibrated_sensor", "sensor"]}
    cam = {c for c, r in t["calibrated_sensor"].items() if t["sensor"][r["sensor_token"]]["channel"] == CAMERA}
    t["sample_data"] = table("sample_data", lambda r: r["is_key_frame"] and r["calibrated_sensor_token"] in cam)
    poses = {r["ego_pose_token"] for r in t["sample_data"].values()}
    t["ego_pose"] = table("ego_pose", lambda r: r["token"] in poses)
    targets = {i for i, r in t["instance"].items()
               if t["category"][r["category_token"]]["name"].startswith(category_prefix)}
    t["sample_annotation"] = table("sample_annotation", lambda r: r["instance_token"] in targets)
    return t


def frame_records(t, category_prefix):
    """Per sample: camera image info plus every in-view annotation of the target category."""
    cams = {}
    for sd in t["sample_data"].values():
        calib = t["calibrated_sensor"][sd["calibrated_sensor_token"]]
        if sd["is_key_frame"] and t["sensor"][calib["sensor_token"]]["channel"] == CAMERA:
            cams[sd["sample_token"]] = (sd, calib)

    anns = {}
    for a in t["sample_annotation"].values():
        cat = t["category"][t["instance"][a["instance_token"]]["category_token"]]["name"]
        if cat.startswith(category_prefix):
            anns.setdefault(a["sample_token"], []).append(a)

    frames = {}
    for sample_token, (sd, calib) in cams.items():
        pose = t["ego_pose"][sd["ego_pose_token"]]
        objects = {}
        for a in anns.get(sample_token, []):
            p = to_camera(a["translation"], pose, calib)
            if in_image(p, calib["camera_intrinsic"], sd["width"], sd["height"]):
                u, v, w = np.asarray(calib["camera_intrinsic"]) @ p
                objects[a["instance_token"]] = {
                    "u": float(u / w), "v": float(v / w),  # box centre in pixels, for figures
                    "visibility": int(a["visibility_token"]), "x": float(p[0]), "z": float(p[2]),
                    # pinhole approx: on-screen height of the box (nuScenes size = w, l, h)
                    "px_height": calib["camera_intrinsic"][0][0] * a["size"][2] / float(p[2])}
        frames[sample_token] = {"image": sd["filename"], "objects": objects}
    return frames


def scene_samples(t, scene):
    tokens, tok = [], scene["first_sample_token"]
    while tok:
        tokens.append(tok)
        tok = t["sample"][tok]["next"]
    return tokens


def find_window(seq, inst, length, min_px=40):
    """Index of the evidence frame within seq if it is a valid window for inst, else None.

    min_px: the target must be at least this many pixels tall (full 1600x900 frame) on the
    evidence frame, because nuScenes visibility measures occlusion, not apparent size.
    """
    if len(seq) < length or any(inst not in f["objects"] for f in seq):
        return None
    vis = [f["objects"][inst]["visibility"] for f in seq]
    if vis[0] != 1 or 4 not in vis:
        return None
    k = vis.index(4)
    # need the evidence frame plus one more frame after it to measure persistence
    if k > length - 2 or any(v > 2 for v in vis[:k]) or seq[k]["objects"][inst]["px_height"] < min_px:
        return None
    for f in seq[:k]:
        if any(o["visibility"] >= 3 for i, o in f["objects"].items() if i != inst):
            return None
    return k


def mine(dataroot, version="v1.0-mini", category_prefix="human.pedestrian",
         object_name="pedestrian", length=6, min_px=40, only_downloaded=False):
    """only_downloaded: skip windows with a frame whose image is not on disk (e.g. trainval
    metadata plus only some of the 10 blob parts), so no sequence points at a missing file."""
    t = load_tables(dataroot, version, category_prefix)
    frames = frame_records(t, category_prefix)
    usable = {s for s, f in frames.items()
              if not only_downloaded or os.path.exists(os.path.join(dataroot, f["image"]))}
    sequences = []
    for scene in t["scene"].values():
        samples = scene_samples(t, scene)
        seen = set()
        for start in range(len(samples) - length + 1):
            window = samples[start:start + length]
            # a missing frame breaks the window; skipping it would join non-consecutive frames
            if not usable.issuperset(window):
                continue
            seq = [frames[s] for s in window]
            for inst in seq[0]["objects"]:
                if inst in seen:
                    continue
                k = find_window(seq, inst, length, min_px)
                if k is None:
                    continue
                seen.add(inst)  # one sequence per instance, so windows don't overlap
                sequences.append({
                    "id": f"{scene['name']}_{inst[:8]}_{start}",
                    "scene": scene["name"],
                    "instance": inst,
                    "object": object_name,
                    "evidence_index": k,
                    "frames": [{
                        "sample_token": s,
                        "image": f["image"],
                        "visibility": f["objects"][inst]["visibility"],
                        "px_height": round(f["objects"][inst]["px_height"]),
                        "u": round(f["objects"][inst]["u"]), "v": round(f["objects"][inst]["v"]),
                        "gt_present": True,
                        "gt_action": safe_action(f["objects"].values()),
                    } for s, f in zip(window, seq)],
                })
    return sequences
