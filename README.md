# Hallucinated Visual Belief Persistence in Sequential Autonomous Vehicle Perception

Course project for CMPE 249: IA Systems (Track 6, Safety and Evaluation), San José State University.

**Team:** Nickzad Bayati, Toney Zhen

- **Paper (PDF):** [report/main.pdf](report/main.pdf) (LaTeX source in [report/](report/))
- **Video demo:** _link to be added_
- **Models:** [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct),
  [llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)
  (used as released; no training or fine-tuning)
- **Dataset:** [nuScenes v1.0](https://www.nuscenes.org/nuscenes#download) (trainval metadata + camera keyframes)

## Abstract

Vision-language models (VLMs) are increasingly used as perception components in autonomous vehicles,
where they are queried over a stream of camera frames. We test whether a VLM that wrongly reports
"no pedestrian" keeps that belief after the pedestrian becomes clearly visible. On 145 nuScenes
front-camera sequences in which a pedestrian goes from 0–40% to 80–100% visible, we query
Qwen2.5-VL-7B and LLaVA-OneVision-7B on every frame without history (stateless), with the conversation
history including their own earlier answers (stateful), and with history plus an instruction to
reconsider. History lowers post-evidence accuracy by 29 and 18 points, and the cause is verbatim
answer copying. The reconsider instruction does not help; only resetting the history does.

## Results

| | Qwen2.5-VL-7B (1024 px) | LLaVA-OneVision-7B (768 px) |
|---|---|---|
| Post-evidence accuracy, stateless → stateful | 38% → 9% | 84% → 65% |
| Stateful − stateless (paired, 95% CI) | −29 pts [−35, −23] | −18 pts [−25, −12] |
| Clear frames seen without history that are still seen with history | 22% [12, 34] of 50 | 73% [65, 81] of 119 |
| Answers identical to the previous answer, stateless / stateful | 31% / 84% | 37% / 93% |
| Reconsider instruction vs plain stateful | no better | no better |

![Seen without history: still seen with history?](figures/visible_without_history.png)

**Limitations:** no pedestrian-free control sequences (false positives are not measured); rule-based
driving-action labels (92% "proceed"); different image sizes per model to fit a 16 GB T4; nuScenes
visibility spans all six cameras and was not reviewed manually; 107 LLaVA replies were a bare "No"
with no action. Details in section 5 of the paper.

## Project structure

```
belief_eval/            evaluation package (python -m belief_eval <command>)
  mine.py               mine sequences from the nuScenes JSON tables (no devkit needed)
  run.py                query a model over sequences in stateless / stateful / reconsider mode
  models.py             Hugging Face VLM wrapper (+ a scripted mock model for tests)
  metrics.py            metrics with bootstrap 95% CIs, incl. visible-without-history
  figures.py            figures as PNG (README) and vector PDF (paper)
  __main__.py           CLI: mine, run, score, figures
notebooks/
  colab_eval.ipynb      run a model on a Colab T4 (setup, smoke test, full run, scores, profiling)
  colab_run_llava7b.ipynb  saved LLaVA run with outputs
results/                raw model answers, one JSON line per (sequence, mode)
figures/                generated figures (.png and .pdf)
report/                 LaTeX paper (IEEE conference format) and bibliography
tests/                  unit and end-to-end tests on a synthetic nuScenes
sequences_trainval.jsonl  the 145 evaluated sequences
needed_images.txt       the 804 CAM_FRONT images those sequences use
```

## Environment setup

Python 3.10+. Mining, scoring, figures and tests run on a CPU:

```bash
pip install numpy pillow matplotlib
```

Model inference needs a CUDA GPU (we used a 16 GB T4) and the rest of `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data

The evaluation needs only 804 front-camera images, not the full dataset.

1. Download the **Trainval metadata** (0.43 GB) and the **camera keyframe** archives of the 10
   Trainval parts from the [nuScenes download page](https://www.nuscenes.org/nuscenes#download)
   (free account required).
2. Extract the metadata into `data/nuscenes/` so that `data/nuscenes/v1.0-trainval/` exists.
3. Extract only the needed images from each part, then delete the archive if you like:

   ```bash
   tar -xzf v1.0-trainval01_keyframes.tgz -C data/nuscenes -T needed_images.txt
   ```

`data/` is git-ignored; nuScenes may not be redistributed.

## Running the evaluation

```bash
# 1. Mine the sequences (CPU, ~40 s; reproduces sequences_trainval.jsonl)
python -m belief_eval mine --dataroot data/nuscenes --version v1.0-trainval --out sequences_trainval.jsonl

# 2. Dry run without a GPU
python -m belief_eval run --dataroot data/nuscenes --sequences sequences_trainval.jsonl --model mock --limit 5 --out results/mock.jsonl

# 3. Inference (GPU); ~2,610 queries per model
python -m belief_eval run --dataroot data/nuscenes --sequences sequences_trainval.jsonl \
  --model Qwen/Qwen2.5-VL-7B-Instruct --4bit --max-side 1024 --out results/results_qwen7b.jsonl
python -m belief_eval run --dataroot data/nuscenes --sequences sequences_trainval.jsonl \
  --model llava-hf/llava-onevision-qwen2-7b-ov-hf --4bit --max-side 768 --out results/results_llava7b.jsonl

# 4. Metrics and figures
python -m belief_eval score results/results_qwen7b.jsonl results/results_llava7b.jsonl
python -m belief_eval figures results/results_qwen7b.jsonl results/results_llava7b.jsonl --dataroot data/nuscenes
```

`run` appends to its output file and skips finished (sequence, mode) pairs, so an interrupted run
resumes with the same command. Each answer records latency, prompt length and peak GPU memory.

**On Colab:** upload the 804 images as `cam_front_images.zip` to `MyDrive/cmpe249/`, open
[notebooks/colab_eval.ipynb](https://colab.research.google.com/github/n-zad/av-vlm-belief-persistence/blob/main/notebooks/colab_eval.ipynb),
choose a T4 runtime and the model in step 1, and run the cells in order.

## Building the paper

```bash
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The paper reads its figures from `../figures/` (or `figures/` next to `main.tex`); regenerate them with step 4 above. On Overleaf,
upload `main.tex`, `references.bib` and a `figures/` folder with the PDFs, then compile with pdfLaTeX.

## Tests

```bash
python -m unittest discover tests
```

## License and attribution

Image crops in `figures/examples_*` are from the nuScenes dataset (Motional), used under
CC BY-NC-SA 4.0 for non-commercial research: <https://www.nuscenes.org/terms-of-use>.
