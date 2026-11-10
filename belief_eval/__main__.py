import argparse
import json

from . import metrics, mine, models, run


def main():
    p = argparse.ArgumentParser(prog="python -m belief_eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine", help="find low->high visibility sequences in nuScenes")
    m.add_argument("--dataroot", required=True)
    m.add_argument("--version", default="v1.0-mini")
    m.add_argument("--category", default="human.pedestrian")
    m.add_argument("--object-name", default="pedestrian")
    m.add_argument("--length", type=int, default=6)
    m.add_argument("--min-px", type=int, default=40, help="min target height in px on the evidence frame")
    m.add_argument("--only-downloaded", action="store_true", help="skip frames whose image is missing")
    m.add_argument("--out", default="sequences.jsonl")

    r = sub.add_parser("run", help="run a VLM over mined sequences")
    r.add_argument("--dataroot", required=True)
    r.add_argument("--sequences", default="sequences.jsonl")
    r.add_argument("--model", default="mock", help="'mock' or a Hugging Face model id")
    r.add_argument("--modes", nargs="+", default=list(run.MODES), choices=run.MODES)
    r.add_argument("--4bit", dest="load_4bit", action="store_true")
    r.add_argument("--max-side", type=int, default=1024, help="longest image side fed to the model; 40 px targets need ~1024")
    r.add_argument("--limit", type=int)
    r.add_argument("--out", default="results.jsonl")

    s = sub.add_parser("score", help="compute metrics from results")
    s.add_argument("results", nargs="+")
    s.add_argument("--n-boot", type=int, default=1000)

    a = p.parse_args()
    if a.cmd == "mine":
        seqs = mine.mine(a.dataroot, a.version, a.category, a.object_name, a.length,
                         a.min_px, a.only_downloaded)
        with open(a.out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x) + "\n" for x in seqs)
        print(f"{len(seqs)} sequences -> {a.out}")
    elif a.cmd == "run":
        with open(a.sequences, encoding="utf-8") as f:
            seqs = [json.loads(line) for line in f][:a.limit]
        model = models.load_model(a.model, a.load_4bit, a.max_side)
        run.run(model, a.model, seqs, a.modes, a.dataroot, a.out)
        print(f"results -> {a.out}")
    else:
        results = []
        for path in a.results:
            with open(path, encoding="utf-8") as f:
                results += [json.loads(line) for line in f]
        print(json.dumps(metrics.summarize(results, a.n_boot), indent=2))


if __name__ == "__main__":
    main()
