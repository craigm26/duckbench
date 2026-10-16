"""python -m diditwork <command>

  run            drive episodes and append one JSON record per episode to a .jsonl file
  score          read .jsonl records and print the leaderboard and the per-condition table
  evallog        turn .jsonl records into an Inspect Robots EvalLog (needs inspect-robots)
  vectors        write the parity vectors the Swift port in Microduck Studio is held to
  show           print one episode: each action, and what the world answered

Examples:

  python -m diditwork run --bench http://127.0.0.1:8770 --model scripted:honest --out results/honest.jsonl
  python -m diditwork run --bench http://127.0.0.1:8770 --bench http://127.0.0.1:8771 \\
      --model hf:meta-llama/Llama-3.3-70B-Instruct --out results/llama-3.3-70b.jsonl
  python -m diditwork run --bench http://127.0.0.1:8770 --model "openai:http://127.0.0.1:11434/v1|qwen2.5:7b" \\
      --out results/qwen2.5-7b-ollama.jsonl
  python -m diditwork score results/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

from . import episode as episode_mod
from . import models, protocol, score as score_mod
from .spec import CAMERAS, FRAMINGS, SPEC_PATH, TASK_IDS, Episode, episodes, load, spec_digest


def _seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _done(path: Path, model_name: str) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rec = json.loads(line)
            if rec["model"] == model_name and rec["end"] != "error":
                done.add(rec["episode"]["id"])
    return done


def cmd_run(args) -> int:
    spec = load(args.spec)
    digest = spec_digest(args.spec)
    todo = episodes(args.tasks, _seeds(args.seeds), args.cameras, args.framings)
    if args.limit:
        todo = todo[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    probe = models.from_spec(args.model)
    done = _done(out, probe.name) if not args.no_resume else set()
    todo = [e for e in todo if e.id not in done]
    from .bench import BenchWorld
    worlds = [BenchWorld(url) for url in args.bench]
    for w in worlds:
        d = w.describe()
        if not d["canonPlant"] and not args.any_plant:
            print(f"{w.url} runs plant {d['plantName']} ({d['plantDigest']}), not the canon plant; "
                  f"pass --any-plant to run on it anyway", file=sys.stderr)
            return 2
    print(f"{probe.name}: {len(todo)} episodes to run ({len(done)} already in {out}); "
          f"{len(worlds)} world(s); spec {spec['version']} sha256 {digest[:12]}", file=sys.stderr)

    jobs: queue.Queue = queue.Queue()
    for e in todo:
        jobs.put(e)
    lock = threading.Lock()
    tally = {"done": 0, "errors": 0}
    started = time.time()

    def worker(world):
        model = models.from_spec(args.model)
        while True:
            try:
                ep = jobs.get_nowait()
            except queue.Empty:
                return
            rec = episode_mod.run(spec, world, model, ep)
            rec["spec_sha256"] = digest
            rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with lock:
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tally["done"] += 1
                if rec["end"] == "error":
                    tally["errors"] += 1
                said = rec.get("report") or {}
                truth = (rec.get("truth") or {}).get("success")
                line = (f"[{tally['done']}/{len(todo)} {time.time() - started:6.0f}s] {ep.id}: {rec['end']}"
                        + (f" -> {said.get('outcome')}" if said else "")
                        + ("" if truth is None else f" (truth {'success' if truth else 'failure'})")
                        + (f" ERROR {rec.get('error')}" if rec["end"] == "error" else ""))
                print(line, file=sys.stderr, flush=True)

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in worlds]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"finished: {tally['done']} episodes, {tally['errors']} errors, {time.time() - started:.0f} s",
          file=sys.stderr)
    return 1 if tally["errors"] else 0


def cmd_score(args) -> int:
    records = score_mod.read_records(args.files)
    scored = score_mod.score(records)
    if args.json:
        Path(args.json).write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    print(score_mod.table(scored))
    for model in scored["models"]:
        print(f"\n### {model}\n")
        print(score_mod.conditions_table(scored, model))
        eff = scored["models"][model]["effects"]
        for name, e in eff.items():
            if e:
                print(f"- {name}: {100 * e['difference']:+.0f} points "
                      f"(95% CI {100 * e['ci95'][0]:+.0f} to {100 * e['ci95'][1]:+.0f})")
    return 0


def cmd_evallog(args) -> int:
    from .evallog import write_evallogs
    records = score_mod.read_records(args.files)
    for path in write_evallogs(records, args.out, spec_path=args.spec):
        print(path)
    return 0


def cmd_vectors(args) -> int:
    from .vectors import write_vectors
    for path in write_vectors(args.out, spec_path=args.spec):
        print(path)
    return 0


def cmd_show(args) -> int:
    for line in Path(args.file).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["episode"]["id"] != args.episode:
            continue
        print(f"== {rec['model']} / {rec['episode']['id']} / end {rec['end']}")
        for step in rec.get("steps", []):
            if "action" in step:
                said = protocol.canonical(step["action"])
            else:
                said = f"(refused: {step['refused']['refusal']})"
            note = "  [reply had text besides the action]" if step.get("extra_text") else ""
            print(f"{step['n']:>2}. {said}{note}")
            if "result" in step:
                print(f"    {step['result']}")
        print("\ntruth:", json.dumps(rec.get("truth")))
        print("evidence:", json.dumps(rec.get("evidence")))
        return 0
    print(f"no episode {args.episode} in {args.file}", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="diditwork", description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--spec", default=str(SPEC_PATH))
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run")
    r.add_argument("--bench", action="append", default=[], help="duckbench URL; repeat for parallel workers")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tasks", nargs="+", default=list(TASK_IDS), choices=TASK_IDS)
    r.add_argument("--seeds", default="1-5")
    r.add_argument("--cameras", nargs="+", default=list(CAMERAS), choices=CAMERAS)
    r.add_argument("--framings", nargs="+", default=list(FRAMINGS), choices=FRAMINGS)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--any-plant", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    s.add_argument("--json")
    s.set_defaults(func=cmd_score)

    e = sub.add_parser("evallog")
    e.add_argument("files", nargs="+")
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_evallog)

    v = sub.add_parser("vectors")
    v.add_argument("--out", required=True)
    v.set_defaults(func=cmd_vectors)

    w = sub.add_parser("show")
    w.add_argument("file")
    w.add_argument("episode")
    w.set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    if args.command == "run" and not args.bench:
        p.error("run needs --bench URL; a real Microduck runs this eval from Microduck Studio")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
