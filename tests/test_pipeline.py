"""End-to-end check on a tiny synthetic nuScenes: python -m unittest discover tests"""
import json
import os
import tempfile
import unittest

from belief_eval import metrics, mine, run
from belief_eval.models import ScriptedModel

CAM_ROT = [0.5, -0.5, 0.5, -0.5]  # nuScenes CAM_FRONT: camera z (forward) = ego x


def write_tables(root, objects, n=6):
    """objects: (instance, category, global_xyz, [visibility per frame])."""
    t = {k: [] for k in ["scene", "sample", "sample_data", "sample_annotation", "instance",
                         "category", "calibrated_sensor", "ego_pose", "sensor"]}
    t["sensor"].append({"token": "sen", "channel": "CAM_FRONT", "modality": "camera"})
    t["calibrated_sensor"].append({"token": "cs", "sensor_token": "sen", "translation": [1.7, 0, 1.5],
                                   "rotation": CAM_ROT,
                                   "camera_intrinsic": [[1266, 0, 816], [0, 1266, 491], [0, 0, 1]]})
    t["scene"].append({"token": "sc", "name": "scene-0001", "first_sample_token": "s0"})
    for i in range(n):
        t["sample"].append({"token": f"s{i}", "scene_token": "sc", "next": f"s{i+1}" if i < n - 1 else ""})
        t["ego_pose"].append({"token": f"e{i}", "rotation": [1, 0, 0, 0], "translation": [0, 0, 0]})
        t["sample_data"].append({"token": f"sd{i}", "sample_token": f"s{i}", "ego_pose_token": f"e{i}",
                                 "calibrated_sensor_token": "cs", "is_key_frame": True,
                                 "filename": f"samples/CAM_FRONT/{i}.jpg", "width": 1600, "height": 900})
    for inst, cat, xyz, vis in objects:
        t["category"].append({"token": "c_" + inst, "name": cat})
        t["instance"].append({"token": inst, "category_token": "c_" + inst})
        for i, v in enumerate(vis):
            t["sample_annotation"].append({"token": f"a_{inst}{i}", "sample_token": f"s{i}",
                                           "instance_token": inst, "visibility_token": str(v),
                                           "translation": xyz, "size": [0.6, 0.7, 1.7]})
    os.makedirs(os.path.join(root, "v1.0-mini"))
    for name, rows in t.items():
        with open(os.path.join(root, "v1.0-mini", name + ".json"), "w") as f:
            json.dump(rows, f)


PED_AHEAD = ("pedA", "human.pedestrian.adult", [12, 1, 1], [1, 1, 4, 4, 4, 4])
PED_BEHIND = ("pedB", "human.pedestrian.adult", [-10, 0, 1], [4] * 6)
CAR_AHEAD = ("carC", "vehicle.car", [20, -3, 1], [1, 1, 4, 4, 4, 4])


def answer(present, action):
    return json.dumps({"present": present, "confidence": 0.9, "evidence": "x", "action": action})


class TestPipeline(unittest.TestCase):
    def mine_with(self, objects, **kw):
        with tempfile.TemporaryDirectory() as root:
            write_tables(root, objects)
            return mine.mine(root, **kw)

    def test_mine_skips_tiny_far_pedestrian(self):
        far = ("pedF", "human.pedestrian.adult", [60, 1, 1], [1, 1, 4, 4, 4, 4])  # ~37 px tall
        self.assertEqual(self.mine_with([far]), [])
        self.assertEqual(len(self.mine_with([far], min_px=30)), 1)

    def test_only_downloaded_skips_missing_images(self):
        self.assertEqual(self.mine_with([PED_AHEAD], only_downloaded=True), [])

    def test_missing_image_breaks_window_instead_of_closing_gap(self):
        ped = ("pedA", "human.pedestrian.adult", [12, 1, 1], [1, 1, 4, 4, 4, 4, 4])
        with tempfile.TemporaryDirectory() as root:
            write_tables(root, [ped], n=7)
            os.makedirs(os.path.join(root, "samples", "CAM_FRONT"))
            for i in (0, 1, 3, 4, 5, 6):  # frame 2 (the evidence frame) is missing
                open(os.path.join(root, "samples", "CAM_FRONT", f"{i}.jpg"), "w").close()
            self.assertEqual(mine.mine(root, only_downloaded=True), [])
            self.assertEqual(len(mine.mine(root)), 1)

    def test_mine_finds_low_to_high_visibility_pedestrian(self):
        seqs = self.mine_with([PED_AHEAD, PED_BEHIND, CAR_AHEAD])
        self.assertEqual([s["instance"] for s in seqs], ["pedA"])  # behind camera / car ignored
        self.assertEqual(seqs[0]["evidence_index"], 2)
        self.assertEqual([f["gt_action"] for f in seqs[0]["frames"]], ["stop"] * 6)  # ~10 m, in lane

    def test_mine_skips_when_another_pedestrian_is_already_clear(self):
        other = ("pedD", "human.pedestrian.adult", [25, -4, 1], [3] * 6)
        self.assertEqual(self.mine_with([PED_AHEAD, other]), [])

    def test_parse(self):
        self.assertEqual(run.parse('```json\n{"present": "no", "confidence": 0.3, "action": "slow"}\n```'),
                         {"present": False, "confidence": 0.3, "action": "slow"})
        self.assertEqual(run.parse("Yes, a pedestrian; I would stop."),
                         {"present": True, "confidence": None, "action": "stop"})
        self.assertEqual(run.parse("unsure")["present"], None)

    def test_run_and_score(self):
        seq = self.mine_with([PED_AHEAD])[0]
        stuck = [answer("no", "proceed")] * 4 + [answer("yes", "stop")] * 2  # fixed 2 frames after evidence
        results = [
            {"model": "m", **run.run_sequence(ScriptedModel(stuck), seq, "stateful", "/data")},
            {"model": "m", **run.run_sequence(ScriptedModel([answer("yes", "stop")] * 6), seq, "stateless", "/data")},
        ]
        s = metrics.summarize(results, n_boot=200)["m"]
        self.assertEqual(s["stateful"]["n_hallucinated"], 1)
        self.assertEqual(s["stateful"]["persistence_rate"], 1.0)
        self.assertEqual(s["stateful"]["latency_median"], 2)
        self.assertEqual(s["stateful"]["post_evidence_accuracy"], 0.5)
        self.assertAlmostEqual(s["stateful"]["unsafe_action_rate"], 4 / 6)
        self.assertEqual(s["stateless"]["n_hallucinated"], 0)
        self.assertEqual(s["stateless"]["safety_balanced_accuracy"], 1.0)
        self.assertEqual(s["stateful_minus_stateless_post_acc"]["mean"], -0.5)

    def test_stateful_sends_history(self):
        seen = []

        class Spy(ScriptedModel):
            def generate(self, messages):
                seen.append(len(messages))
                return super().generate(messages)

        seq = self.mine_with([PED_AHEAD])[0]
        run.run_sequence(Spy(), seq, "stateful", "/data")
        run.run_sequence(Spy(), seq, "stateless", "/data")
        self.assertEqual(seen, [1, 3, 5, 7, 9, 11] + [1] * 6)


if __name__ == "__main__":
    unittest.main()
