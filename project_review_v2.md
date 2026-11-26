# Project Review v2: Gaps Remaining in the Paper

*Reviewed 2026-11-25 against `report/main.tex` at commit `7439841`. The compiled `report/main.pdf` is built from the same source. This follows up on [project_review.md](project_review.md) and covers only the points from that review that the paper leaves out or only partly addresses. All numbers below were recomputed from `results/*.jsonl` and `sequences_trainval.jsonl`.*

## Summary

The paper already handles most of the first review. `visible_without_history` is the headline metric. The failure-analysis section covers control sequences, label noise, safety labels, confidence and format failures. Future work lists split-history modes.

What remains, in priority order:

| # | Issue | Status | Effort to fix |
|---|---|---|---|
| 1 | "Hallucination" still means a "no" on a mostly occluded frame | Partly addressed | Wording only |
| 2 | "Sequences do not overlap" is false, and the bootstrap ignores clustering | Not addressed; contains an error | One sentence + one limitation |
| 3 | Persistence rate compared across modes with different denominators | Not addressed | One sentence, numbers below |
| 4 | No McNemar test / discordant pairs | Not addressed | One sentence or a small table |
| 5 | Belief persistence vs. plain copying, both directions | Partly addressed | One paragraph |
| 6 | Planted-belief experiment from the instructor feedback | Not mentioned | Future-work item |
| 7 | Smaller items (vision-encoder quantization, prompt wording, safety-label rule) | Not mentioned | One clause each |

## 1. "Hallucination" is still defined by the frame before the evidence

**What the paper says.** The title uses "Hallucinated Belief Persistence" ([main.tex:21](report/main.tex#L21)). The abstract studies "a VLM that has *wrongly* reported that no pedestrian is present" ([main.tex:33](report/main.tex#L33)). Table I reports "Wrong at k−1" ([main.tex:173](report/main.tex#L173), [main.tex:178](report/main.tex#L178)). Persistence rate starts from "sequences wrong at k−1" ([main.tex:147](report/main.tex#L147)).

**Why it matters.** On frame k−1 the pedestrian is 0–40% visible in 121 of 145 sequences and 41–60% visible in the other 24. A "no" there is often a defensible perceptual call. This is the exact case the instructor feedback warned against calling a hallucination. The headline metric avoids the problem, but the framing and the persistence metric don't. A reader who only skims the abstract sees the old definition.

**Suggested fix.** Keep the term "hallucination" for the error the paper actually isolates, and name the k−1 answer something neutral.

- Abstract: "a VLM that has reported no pedestrian while the pedestrian was mostly occluded" instead of "has wrongly reported".
- Table I row: "'No' at k−1 (n)" instead of "Wrong at k−1 (n)".
- Add to Metrics, after the persistence-rate definition:

  > At k−1 the pedestrian is at most 40% visible in 121 of 145 sequences, so a "no" there is often a reasonable perceptual call. We call it an *initial miss*, not a hallucination. The hallucination we measure is a "no" on the evidence frame that the same model answers correctly without history, which is what visible-without-history accuracy counts.

- Title: optional. Keeping "Hallucinated" is defensible once the definition above is in the paper.

## 2. "Sequences do not overlap" is false, and intervals ignore clustering

**What the paper says.** "One sequence is kept per pedestrian so sequences do not overlap" ([main.tex:124](report/main.tex#L124)). Intervals are "bootstrap intervals over sequences" ([main.tex:151](report/main.tex#L151)).

**Why it matters.**

- **The sentence is wrong.** 59 images appear in more than one sequence. Two pedestrians in the same scene can share frames.
- **The bootstrap assumes independent sequences.** The 145 sequences come from only 120 scenes, so the intervals are slightly too narrow. With effects this large, the conclusions won't change, but a careful reader will notice the overlap claim.

**Suggested fix.**

- [main.tex:124](report/main.tex#L124): "One sequence is kept per pedestrian. Sequences from the same scene can share frames (59 of the 804 images appear in more than one sequence)."
- Add to the limitations list: "(vii) The 145 sequences come from 120 scenes and are not fully independent; the bootstrap resamples sequences, not scenes, so intervals may be slightly narrow."
- Better, if time allows: resample by scene in `bootstrap_ci` and report those intervals instead.

## 3. Persistence rate is compared across modes with different denominators

**What the paper says.** "History also raises the persistence rate from 67% to 96% for Qwen and from 28% to 82% for LLaVA" ([main.tex:184](report/main.tex#L184)).

**Why it matters.** Each rate is computed only on sequences that are wrong at k−1 *in that mode*, and which sequences qualify depends on the mode:

| | Stateless | Stateful | Reconsider |
|---|---|---|---|
| Qwen, "no" at k−1 | 125 | 139 | 139 |
| LLaVA, "no" at k−1 | 60 | 62 | 62 |

So the two rates describe different sets of sequences.

**Suggested fix.** Report persistence on the sequences that are "no" at k−1 in **both** modes. The conclusion holds:

| | n | Stateless persistence | Stateful persistence |
|---|---|---|---|
| Qwen | 125 | 67% (84/125) | 97% (121/125) |
| LLaVA | 53 | 30% (16/53) | 85% (45/53) |

A replacement sentence:

> On the sequences that both modes miss at k−1 (Qwen n=125, LLaVA n=53), history raises persistence from 67% to 97% for Qwen and from 30% to 85% for LLaVA.

Alternatively, keep Table I as it is and add a note that its persistence rates are conditioned on mode-dependent subsets.

## 4. No significance test on the discordant pairs

**What the paper says.** The visible-without-history result is reported as a proportion with a bootstrap CI ([main.tex:193](report/main.tex#L193)). That metric conditions on stateless being correct, so it never shows the reverse case, where history *fixes* a frame the model missed without it.

**The counts.** On the evidence frame:

| | Stateless ✓, stateful ✗ | Stateless ✗, stateful ✓ | Exact McNemar p |
|---|---|---|---|
| Qwen (stateful) | 39 | 0 | 3.6 × 10⁻¹² |
| LLaVA (stateful) | 32 | 6 | 2.4 × 10⁻⁵ |
| Qwen (reconsider) | 42 | 0 | 4.5 × 10⁻¹³ |
| LLaVA (reconsider) | 32 | 6 | 2.4 × 10⁻⁵ |

**Suggested fix.** Add one sentence after the Fig. 2 paragraph:

> On the evidence frame, history turns a correct stateless answer into a miss in 39 sequences for Qwen and 32 for LLaVA, and does the reverse in 0 and 6 (exact McNemar test, p < 10⁻¹¹ and p < 10⁻⁴).

This shows the effect isn't one-sided by construction, which matters most for LLaVA's 6 reverse cases.

## 5. Belief persistence vs. plain copying is not discussed as a limitation

**What the paper says.** Copying is presented as the mechanism, which is good. The ablation's "after a 'yes', both models stay correct (100%)" ([main.tex:235](report/main.tex#L235)) is presented as evidence that history drives the error.

**What's missing.** The same inertia also works in the other direction, and the paper doesn't say so:

| "yes" → "no" switches between consecutive frames | Stateless | Stateful | Reconsider |
|---|---|---|---|
| Qwen | 36 | 12 | 12 |
| LLaVA | 53 | **0** | **0** |

With history, LLaVA *never* changes "yes" to "no". Because ground truth is always "present", this inertia only shows up as an error in one direction. The current design therefore can't tell "the model holds a belief that the pedestrian is absent" from "the model repeats whatever it said last". The history also holds both earlier images and earlier answers, so it can't tell which of the two causes the copying.

**Suggested fix.** Add a failure-analysis paragraph:

> **Persistence or copying?** Inertia runs both ways: with history, LLaVA never switches from "yes" to "no" (53 times without history), and Qwen does so 12 times (36 without). Since ground truth is always "present", only the harmful direction counts as an error. The current history contains both earlier images and earlier answers, so we cannot yet say whether the model holds a visual belief or simply repeats its last answer. Control sequences and split-history modes (future work) would separate the two.

## 6. The planted-belief experiment is not mentioned

**What the feedback asked for.** Plant a wrong belief, then add stronger and stronger evidence. The first review called this the biggest gap against the feedback. The paper doesn't mention it in limitations or future work ([main.tex:343](report/main.tex#L343)).

**Why it matters.** It is the cleanest test of the novelty audit's key claim: the same later frame, with and without a wrong answer in the history. The current comparison changes the whole history at once. A grader checking the feedback will look for it.

**Suggested fix.** At minimum, add it to future work:

> (6) planting a fixed "no" answer in the history for one to four turns before the evidence frame, and a planted "yes" on pedestrian-free frames, to measure how much evidence it takes to overturn a belief the model did not form itself.

If there is GPU time, it's a small code change: a mode in [run.py:56-58](belief_eval/run.py#L56-L58) that writes a fixed `{"present": "no", ...}` reply into the history for t < k. The ablation by pedestrian height ([main.tex:226-248](report/main.tex#L226-L248)) already provides the "stronger evidence" axis.

## 7. Smaller items

- **4-bit quantization of the vision encoder.** The paper mentions 4-bit weights ([main.tex:158](report/main.tex#L158), limitation v) but not that bitsandbytes also quantizes the vision encoder ([models.py:32-35](belief_eval/models.py#L32-L35)). That could explain part of Qwen's weak small-pedestrian detection (16% under 60 px). Suggested addition to limitation (v): "including the vision encoder, which may reduce small-object detection."
- **Prompt says "the image" with up to six images in context** ([main.tex:128](report/main.tex#L128)). In stateful mode the model could reasonably read "the image" as any image in the conversation. Worth one clause in limitations. A rerun would use "the most recent image".
- **The safety-action rule ignores visibility and non-pedestrians.** `safe_action` ([mine.py:43](belief_eval/mine.py#L43)) only sees pedestrians, because `load_tables` drops every other category. It also counts pedestrians that are 0–40% visible. Limitation (ii) already calls safety accuracy weak evidence. Adding "it considers only pedestrians, including occluded ones" explains why.

## Code and repository issues (no paper change needed)

These points from the first review concern the code and repository, not the paper. They don't change the reported results, but they affect whether someone else can reproduce them. All four are still present in the current code.

- **Run settings aren't saved with the results.**
  - Each result line ([run.py:80](belief_eval/run.py#L80)) records the model name, sequence, mode and answers. It doesn't record `--max-side`, the `--4bit` flag, the prompt text or version, or the `transformers` version.
  - Resuming skips any (sequence, mode) pair already in the output file, matching on model name only ([run.py:74](belief_eval/run.py#L74)). If a run is resumed with a different image size or prompt, the file silently mixes settings.
  - Today the only record that Qwen ran at 1024 px and LLaVA at 768 px is the README and the paper.
  - Fix: write a `config` field into each line (or one header line per file), and have resume refuse to continue if the config differs.
- **`transformers` isn't pinned in the notebooks.**
  - Both notebooks run `pip install -q -U transformers accelerate bitsandbytes`, so every Colab session installs whatever version is newest that day. `requirements.txt` only sets a lower bound (`transformers>=4.56`).
  - Chat templates and image processors for Qwen2.5-VL and LLaVA-OneVision have changed between releases. A later rerun could tokenize the same history differently and give different answers.
  - Fix: pin the exact versions used for the committed runs in both the notebooks and `requirements.txt`, and record them with the results as above.
- **The text parser has a loose fallback.**
  - `parse` ([run.py:23](belief_eval/run.py#L23)) reads the JSON if it can. Otherwise it takes the first standalone "yes"/"no" anywhere in the reply ([run.py:34](belief_eval/run.py#L34)), and the first "proceed"/"slow"/"stop" ([run.py:38](belief_eval/run.py#L38)).
  - In the committed results, the fallback was used for exactly the 107 bare "No" replies from LLaVA. Those parse correctly for presence and give no action. Qwen never needed it. So the current numbers aren't affected.
  - The risk is in future runs. When the JSON is missing or broken, the parser takes whichever matching word comes first. A reply like "No pedestrian, but there is a stop sign" would be scored as action "stop", and "Yes, no pedestrian is visible" would be scored as "present".
  - Fix: store which path was used (`"parsed_from": "json" | "text"`) in each frame so it can be audited, and match actions only inside the JSON or at the start of the reply.
- **The two notebooks are nearly identical.**
  - [colab_run_llava7b.ipynb](notebooks/colab_run_llava7b.ipynb) is a copy of [colab_eval.ipynb](notebooks/colab_eval.ipynb) with the model choice switched to LLaVA (13 vs 14 cells). Unlike the template, it was saved with its outputs (5 cells), and the README now describes it as the saved LLaVA run.
  - Keeping it as a record of the run is reasonable. The cost is that a fix to one notebook won't reach the other.
  - Fix: either rename it to make its role clear (for example `colab_run_llava7b_outputs.ipynb`) and note that `colab_eval.ipynb` is the one to edit, or save the run's outputs to a log file and delete the copy.
