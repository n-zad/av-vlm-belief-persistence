"""End-to-end check on a tiny synthetic nuScenes: python -m unittest discover tests"""
import json
import os
import tempfile
import unittest

from belief_eval import metrics, mine, run
from belief_eval.models import ScriptedModel

CAM_ROT = [0.5, -0.5, 0.5, -0.5]  # nuScenes CAM_FRONT: camera z (forward) = ego x


def write_tables(root, objects, n=6, description=""):
    """objects: (instance, category, global_xyz, [visibility per frame])."""
    t = {k: [] for k in ["scene", "sample", "sample_data", "sample_annotation", "instance",
                         "category", "calibrated_sensor", "ego_pose", "sensor"]}
    t["sensor"].append({"token": "sen", "channel": "CAM_FRONT", "modality": "camera"})
    t["calibrated_sensor"].append({"token": "cs", "sensor_token": "sen", "translation": [1.7, 0, 1.5],
                                   "rotation": CAM_ROT,
                                   "camera_intrinsic": [[1266, 0, 816], [0, 1266, 491], [0, 0, 1]]})
    t["scene"].append({"token": "sc", "name": "scene-0001", "first_sample_token": "s0", "description": description})
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
                                           "translation": xyz, "size": [0.6, 0.7, 1.7],
                                           "rotation": [1, 0, 0, 0]})
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
                         {"present": False, "confidence": 0.3, "action": "slow", "parsed_from": "json"})
        # text fallback recovers presence but never the action ("stop sign" is not an action)
        self.assertEqual(run.parse("No pedestrian, but there is a stop sign."),
                         {"present": False, "confidence": None, "action": None, "parsed_from": "text"})
        self.assertEqual(run.parse('{"present": "maybe", "action": "brake"}'),
                         {"present": None, "confidence": None, "action": None, "parsed_from": None})
        self.assertEqual(run.parse("[1, 2]")["parsed_from"], None)  # JSON that is not an object

    def test_run_and_score(self):
        seq = self.mine_with([PED_AHEAD])[0]
        stuck = [answer("no", "proceed")] * 4 + [answer("yes", "stop")] * 2  # fixed 2 frames after evidence
        results = [
            {"model": "m", **run.run_sequence(ScriptedModel(stuck), seq, "stateful", "/data")},
            {"model": "m", **run.run_sequence(ScriptedModel([answer("yes", "stop")] * 6), seq, "stateless", "/data")},
        ]
        s = metrics.summarize(results, n_boot=200)["m"]
        self.assertEqual(s["stateful"]["n_wrong_before"], 1)
        self.assertEqual(s["stateful"]["persistence_rate"], 1.0)
        self.assertEqual(s["stateful"]["latency_median"], 2)
        self.assertEqual(s["stateful"]["post_evidence_accuracy"], 0.5)
        self.assertAlmostEqual(s["stateful"]["unsafe_action_rate"], 4 / 6)
        self.assertEqual(s["stateless"]["n_wrong_before"], 0)
        self.assertEqual(s["stateless"]["safety_balanced_accuracy"], 1.0)
        self.assertEqual(s["stateful_minus_stateless_post_acc"]["mean"], -0.5)
        self.assertIn("latency_s_median", s["stateful"])  # timing recorded for every call
        # stateless sees the pedestrian on the evidence frame; stateful (with history) does not
        self.assertEqual(s["visible_without_history"]["n"], 1)
        self.assertEqual(s["visible_without_history"]["stateful"]["evidence_frame_accuracy"], 0.0)

    def test_figures_render(self):
        from PIL import Image
        from belief_eval import figures
        with tempfile.TemporaryDirectory() as root:
            write_tables(root, [PED_AHEAD])
            seq = mine.mine(root)[0]
            os.makedirs(os.path.join(root, "samples", "CAM_FRONT"))
            for f in seq["frames"]:
                Image.new("RGB", (1600, 900)).save(os.path.join(root, f["image"]))
            stuck = [answer("no", "proceed")] * 4 + [answer("yes", "stop")] * 2
            results = [{"model": "m", **run.run_sequence(ScriptedModel(a), seq, mode, root)}
                       for mode, a in [("stateless", [answer("yes", "stop")] * 6), ("stateful", stuck),
                                       ("planted_no", stuck[2:])]]
            ctrl = {**seq, "id": "scene-0002_control_0", "kind": "control",
                    "frames": [{**f, "gt_present": False} for f in seq["frames"]]}
            results.append({"model": "m", **run.run_sequence(ScriptedModel(), ctrl, "planted_yes", root)})
            out = os.path.join(root, "figs")
            names = ["copying", "examples_m", "planted_belief", "seen_by_size", "visible_without_history",
                     "yes_by_frame"]
            self.assertEqual(figures.make_all(results, {seq["id"]: seq}, out, root),
                             sorted(n + ext for n in names for ext in (".pdf", ".png")))

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

    def test_planted_modes_write_fixed_answers_before_evidence(self):
        seen = []

        class Spy(ScriptedModel):
            def generate(self, messages):
                seen.append(messages)
                return super().generate(messages)

        seq = self.mine_with([PED_AHEAD])[0]  # evidence frame k = 2
        rec = run.run_sequence(Spy([answer("no", "proceed")] * 4), seq, "planted_no", "/data")
        # no model call on the two planted frames; from k on the history holds both planted answers
        self.assertEqual([len(m) for m in seen], [5, 7, 9, 11])
        self.assertEqual(seen[0][1]["content"][0]["text"], run.PLANTED["planted_no"].format(object="pedestrian"))
        self.assertEqual([f["planted"] for f in rec["frames"]], [True, True, False, False, False, False])
        self.assertEqual([f["present"] for f in rec["frames"]], [False] * 6)
        self.assertNotIn("latency_s", rec["frames"][0])
        self.assertIn("latency_s", rec["frames"][2])
        yes = run.run_sequence(ScriptedModel(), seq, "planted_yes", "/data")
        self.assertEqual([f["present"] for f in yes["frames"][:2]], [True, True])

    def test_resume_skips_done_and_refuses_other_config(self):
        seq = self.mine_with([PED_AHEAD])[0]
        with tempfile.TemporaryDirectory() as root:
            out = os.path.join(root, "r.jsonl")
            run.run(ScriptedModel(), "m", [seq], ["stateless"], "/data", out)
            run.run(ScriptedModel(), "m", [seq], ["stateless", "stateful"], "/data", out)
            with open(out) as f:
                recs = [json.loads(line) for line in f]
            self.assertEqual([r["mode"] for r in recs], ["stateless", "stateful"])
            self.assertEqual(recs[0]["config"]["prompt_sha"], run.PROMPT_SHA)

            class Other(ScriptedModel):
                config = {"model": "scripted", "max_side": 768}

            with self.assertRaises(SystemExit):
                run.run(Other(), "m", [seq], ["reconsider"], "/data", out)
            run.run(Other(), "other-model", [seq], ["stateless"], "/data", out)  # other models are fine

    def controls_with(self, objects, n_frames=8, positives=None, description=""):
        positives = positives or [{"scene": "scene-0001", "evidence_index": 3}]
        with tempfile.TemporaryDirectory() as root:
            write_tables(root, objects, n=n_frames, description=description)
            return mine.mine_controls(root, positives, version="v1.0-mini")

    def test_controls_need_six_frames_without_any_pedestrian(self):
        early = ("pedA", "human.pedestrian.adult", [12, 1, 1], [1, 1])  # in view on frames 0-1 only
        controls = self.controls_with([early])
        self.assertEqual([c["id"] for c in controls], ["scene-0001_control_2"])
        c = controls[0]
        self.assertEqual((c["kind"], c["evidence_index"]), ("control", 3))  # k copied from the scene's positive
        self.assertEqual([f["image"] for f in c["frames"]], [f"samples/CAM_FRONT/{i}.jpg" for i in range(2, 8)])
        self.assertTrue(all(f["gt_present"] is False for f in c["frames"]))
        self.assertEqual(self.controls_with([early], n_frames=7), [])  # frames 2-6 are only 5

    def test_controls_reject_partly_visible_boxes_and_cyclists(self):
        # centre projects just left of the image, but the near edge of the box is inside it
        edge = ("pedE", "human.pedestrian.adult", [4, 1.6, 1], [1] * 8)
        self.assertFalse(mine.in_image(
            mine.to_camera([4, 1.6, 1], {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]},
                           {"rotation": CAM_ROT, "translation": [1.7, 0, 1.5]}),
            [[1266, 0, 816], [0, 1266, 491], [0, 0, 1]], 1600, 900))
        self.assertEqual(self.controls_with([edge]), [])
        bike = ("bikeB", "vehicle.bicycle", [15, 0, 1], [1] * 8)
        self.assertEqual(self.controls_with([bike]), [])
        behind = ("pedB", "human.pedestrian.adult", [-10, 0, 1], [4] * 8)
        self.assertEqual(len(self.controls_with([behind])), 1)

    def test_controls_are_matched_on_conditions(self):
        self.assertEqual([mine.scene_condition({"description": d}) for d in
                          ("Night, rain, peds", "Rain, wet road", "Parking lot, peds", "")],
                         ["night", "rain", "clear", "clear"])
        self.assertEqual(mine.matched_counts(["a", "a", "b"], 4), {"a": 3, "b": 1})
        # the positives' mix: 9 night, 11 rain, 125 clear sequences -> 75 controls
        counts = mine.matched_counts(["night"] * 9 + ["rain"] * 11 + ["clear"] * 125, 75)
        self.assertEqual(counts, {"night": 4, "rain": 6, "clear": 65})
        early = ("pedA", "human.pedestrian.adult", [12, 1, 1], [1, 1])
        self.assertEqual(self.controls_with([early], description="Night, rain")[0]["condition"], "night")

    def test_control_records_are_kept_out_of_positive_metrics(self):
        seq = self.mine_with([PED_AHEAD])[0]
        ctrl = {**seq, "id": "c", "kind": "control",
                "frames": [{**f, "gt_present": False} for f in seq["frames"]]}
        results = [{"model": "m", **run.run_sequence(ScriptedModel(), s, "stateless", "/data")} for s in (seq, ctrl)]
        self.assertEqual(results[1]["kind"], "control")
        self.assertEqual(metrics.summarize(results, n_boot=50)["m"]["stateless"]["n_sequences"], 1)

    def test_planted_and_paired_metrics(self):
        seq = self.mine_with([PED_AHEAD])[0]  # k = 2
        run_ = lambda answers, mode: {"model": "m", **run.run_sequence(ScriptedModel(answers), seq, mode, "/data")}
        results = [run_([answer("yes", "stop")] * 6, "stateless"),
                   run_([answer("no", "proceed"), answer("yes", "stop"), answer("yes", "stop")], "planted_no")]
        s = metrics.summarize(results, n_boot=50)["m"]
        p = s["planted_no"]
        self.assertEqual(p["frame_accuracy"], 3 / 4)  # no, yes, yes, then the mock "yes"; planted frames not counted
        self.assertEqual(p["safety_balanced_accuracy"], 0.5)  # 2 of 4 answered frames say "stop" (gt)
        self.assertEqual((p["n_wrong_before"], p["persistence_rate"], p["latency_median"]), (1, 0.0, 1))
        self.assertEqual(p["switches"], {"yes_to_no": 0, "no_to_yes": 1})  # planted "no" -> no -> yes
        self.assertEqual(p["accuracy_by_offset"][0], {"accuracy": 0, "n": 1})
        self.assertEqual(s["visible_without_history"]["planted_no"]["evidence_frame_accuracy"], 0.0)
        self.assertEqual(s["mcnemar_evidence_frame"]["planted_no"],
                         {"n_pairs": 1, "stateless_only": 1, "mode_only": 0, "p_exact": 1.0})

    def test_control_metrics(self):
        seq = self.mine_with([PED_AHEAD])[0]
        ctrl = {**seq, "id": "scene-0002_control_0", "kind": "control",
                "frames": [{**f, "gt_present": False} for f in seq["frames"]]}
        results = [{"model": "m", **run.run_sequence(ScriptedModel(a), ctrl, mode, "/data")} for mode, a in
                   [("stateless", [answer("no", "proceed")] * 6), ("planted_yes", [answer("yes", "slow")] * 4)]]
        c = metrics.summarize(results, n_boot=50)["m"]["controls"]
        self.assertEqual(c["stateless"]["says_present_rate"], 0.0)  # no false positives without history
        self.assertEqual(c["planted_yes"]["persistence_rate"], 1.0)  # the planted "yes" never goes away
        self.assertEqual(c["empty_without_history"]["planted_yes"]["evidence_frame_accuracy"], 0.0)

    def test_mcnemar_and_scene_bootstrap(self):
        self.assertAlmostEqual(metrics.mcnemar_exact(39, 0), 2 / 2 ** 39)
        self.assertEqual(metrics.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(metrics.mcnemar_exact(3, 3), 1.0)
        rng = metrics.random.Random(0)
        # two scenes, one all 0s and one all 1s: resampling scenes gives only means 0, 1/3, 2/3, 1
        ci = metrics.bootstrap_ci([0, 1, 1], 200, rng, ["a", "b", "b"])
        self.assertEqual(ci, [0.0, 1.0])
        self.assertEqual(metrics.scene_of("scene-0003_24082a54_0"), "scene-0003")


if __name__ == "__main__":
    unittest.main()
