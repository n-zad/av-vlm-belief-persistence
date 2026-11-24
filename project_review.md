# Project Review: Belief Persistence Evaluation

*Reviewed 2026-11-24 against commit `9aa0c89`. It covers the code in `belief_eval/`, the committed results in `results/`, the Project Proposal, the instructor feedback (`proposal_feedback.txt`) and the AI Novelty & Feasibility Audit. The proposal, feedback and audit are treated as possibly outdated suggestions, not as fixed requirements.*

## Summary

The project is in good shape. The pipeline is small and readable, the 9 unit tests pass, and `python -m belief_eval score results/*.jsonl` reproduces the README results table exactly.

The headline effect holds up well when examined directly. On the evidence frame (the first frame where the pedestrian is clearly visible):

| | Seen without history, missed with history | Missed without history, seen with history |
|---|---|---|
| Qwen2.5-VL-7B | 39 | 0 |
| LLaVA-OneVision-7B | 32 | 6 |

The main issues are about the experiment design rather than bugs. Issues 1 and 2 below matter most.

## Main issues

### 1. "Hallucination" is defined as exactly the case the feedback warned against

In [metrics.py:19](belief_eval/metrics.py#L19), `hallucinated` means "wrong on frame k−1". By construction, that frame shows the pedestrian at only 0–60% visibility (121 of 145 sequences are at 0–40%). Answering "no" there is a reasonable perceptual call, not a hallucination.

The existing `visible_without_history` metric already does the right thing. Suggested changes:

- Make `visible_without_history` the headline metric.
- Report the discordant pairs from the table above, with a McNemar test.
- Rename "hallucination" to something like "prior negative belief" or "initial miss".

### 2. Belief persistence can't yet be told apart from plain answer copying

Ground truth is always "present" (`gt_present: True` is hardcoded in [mine.py:173](belief_eval/mine.py#L173)), so only the harmful direction of inertia is visible. The data shows the inertia goes both ways:

- **LLaVA with history never changes "yes" to "no"** (0 times). Without history it does so 53 times.
- The copying figure shows 84% (Qwen) and 93% (LLaVA) of answers with history are word-for-word repeats of the previous answer.

Two additions would separate the explanations:

- **Control sequences:** pedestrian-free windows, and windows where the pedestrian leaves the frame. These would also measure false positives. LLaVA says "yes" on 52–59% of pre-evidence frames, which may be guessing.
- **Splitting what the history contains:** right now the history holds earlier images *and* earlier answers. The novelty audit's key claim needs a "same frame, with vs without the wrong answer" comparison. Two new modes would do it:
  - earlier images with no earlier answers
  - earlier answers as text with no earlier images

## Against the instructor feedback

| Feedback point | Status |
|---|---|
| Don't call every initial error a hallucination | ⚠️ Still the definition (see issue 1) |
| Plant a wrong belief, then add stronger and stronger evidence | ❌ Not done. This is the biggest gap |
| Focus on object presence | ✅ Presence only, but the README abstract ([README.md:9](README.md#L9)) still promises attributes, spatial relations and scene state |
| Plan for limited GPU | ✅ Colab T4 with 4-bit quantization |

The planted-belief experiment is cheap with the current design:

1. Add a mode in [run.py:56-58](belief_eval/run.py#L56-L58) that, for t < k, puts a fixed `{"present": "no", ...}` reply into the history instead of the model's own answer.
2. Add a matching planted "yes" mode on pedestrian-free frames as the control.
3. Vary how many planted turns there are (1–4), and group results by pedestrian height on the evidence frame. This gives the "how much evidence or how many frames" curves the feedback asks for.

The mined sequences already have k spread across 1–4 (63 / 33 / 18 / 31 sequences).

## Against the proposal

- ✅ **At least 100 sequences on two VLMs:** 145 per model. Bootstrap confidence intervals and the "reconsider" mitigation are both done. Stateless mode doubles as the "reset history" mitigation.
- ⚠️ **Proposal targets:** "persistence ≤30%" and "latency ≤2 frames" describe model behavior, not project success. They read better as thresholds the results are compared against. As thresholds:
  - Qwen persistence with history is 96% and LLaVA's is 82%.
  - The paired difference is far above the 10-point bar for both models (−29 for Qwen, −18 for LLaVA).
- ⚠️ **Promised manual review of visibility** was not done. This may explain why Qwen without history sees only 50 of 145 evidence frames. Hand-checking about 30 evidence frames would bound the label noise.
- ❌ **Safety-action accuracy is close to meaningless:**
  - 824 of 870 labels are "proceed".
  - `safe_action` ([mine.py:43](belief_eval/mine.py#L43)) only considers pedestrians, because `load_tables` drops every other category. It also counts pedestrians that are 0–40% visible.
  - It should be demoted to an extension, as the feedback suggested.

## Against the novelty audit

As built, the project sits near the audit's 7/10 case. The audit scores it 4/10 if it becomes a general study of hallucinations across frames, and it hasn't. But the audit's key differentiator is "the same later frame with and without the earlier wrong answer in history". That needs the planted-belief mode or the split-history modes. The current comparison changes the whole history at once.

## Statistics

- **Correction latency median:** it is computed only over sequences that were eventually corrected ([metrics.py:48](belief_eval/metrics.py#L48)). Qwen's "median 2" comes from 17 of 139 sequences; 122 were never corrected. The uncorrected ones should be treated as censored (a survival curve, or reported as "> frames remaining"). The number of frames after the evidence frame also varies with k, from 2 to 5.
- **Persistence rate across modes isn't a like-for-like comparison:** it is conditioned on the k−1 answer, and that answer itself depends on the mode. The paired difference avoids this problem.
- **Sequences aren't independent:** the 145 sequences come from 120 scenes, and 59 frames appear in more than one sequence. A bootstrap that resamples by scene would be more honest. The effect on the intervals is likely small.

## Smaller issues

- **4-bit quantization isn't mentioned in the README.** Bitsandbytes also quantizes the vision encoder ([models.py:32-35](belief_eval/models.py#L32-L35)), which could hurt small-object detection. Skipping it is a cheap fix: `llm_int8_skip_modules=["visual"]` for Qwen, `["vision_tower", "multi_modal_projector"]` for LLaVA.
- **The run settings aren't saved with the results.** Each record lacks `max_side`, the 4-bit flag, the prompt version and the transformers version. Resuming will silently mix settings, and the notebooks run an unpinned `pip install -U transformers`.
- **The confidence field means different things per model.** LLaVA outputs 0.0 alongside "no" (it reads it as P(present)), while Qwen outputs 0.9 alongside "no". The yes/no token probabilities would give a real measure of belief strength.
- **Bare "No" replies are counted as unsafe actions.** There are 107 from LLaVA, which accounts for about 4 points of its ~10% unsafe rate. They would be better reported as missing.
- **The prompt says "the image" while up to 6 images are in the context** ([run.py:15](belief_eval/run.py#L15)). "The most recent image" would remove the ambiguity.
- **Text parsing falls back to the first "yes"/"no" anywhere in the reply, and "stop" can match "stop sign".** Parsing failures are currently 0, so this is low impact, but it's worth logging which path was used.
- **[colab_run_llava7b.ipynb](notebooks/colab_run_llava7b.ipynb) is a copy of [colab_eval.ipynb](notebooks/colab_eval.ipynb)** with one line switched.
