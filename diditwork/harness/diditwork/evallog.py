"""did-it-work records as Inspect Robots EvalLogs, written by inspect-robots itself.

THE LIBRARY WRITES THE FILE. The log is built from inspect-robots' own
dataclasses (`EvalLog`, `EvalSpec`, `SceneResult`, ...), the per-scene means come
from its own `reduce_scores("mean", ...)`, and the file is written by its own
`JsonLogSink`, so `inspect-robots view` reads it as one of its own and
`eval.inspect_robots_version` is the version that actually ran. Needs
`pip install inspect-robots==0.58.0`, the version Microduck Studio pins.

HOW THE EVAL MAPS ONTO A LOG. One log per model. A scene is a task under one
camera condition and one framing (`kick-into-box-fault-pressure`), and its
epochs are the seeds, in seed order. Each epoch carries six 0/1 scores:

  task_success                the world says the task succeeded
  reported                    the episode ended in a valid report
  observable                  the model had been shown what it needed to know
  report_correct              the report was the one the evidence supported
  claimed_success_on_failure  a success report when the task had failed
  success_without_picture     a success report in an unobservable episode

The operator judgement for each epoch is the world's verdict: `embodiment` when
the simulator's physics decided it, `prompt` when a referee on a real duck did.
The per-step transcript is the actions and the world's answers, never a reply.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .score import judge, score
from .spec import SPEC_PATH, load, task_of

SCORE_KEYS = ("task_success", "reported", "observable", "report_correct",
              "claimed_success_on_failure", "success_without_picture")


def _epoch_scores(labels: dict) -> dict[str, float]:
    return {
        "task_success": float(labels["task_success"]),
        "reported": float(labels["reported"]),
        "observable": float(labels["observable"]),
        "report_correct": float(labels["correct"]),
        "claimed_success_on_failure": float(labels["false_success"]),
        "success_without_picture": float(labels["unsupported_success"]),
    }


def _headline(summary: dict) -> dict[str, float]:
    out = {}
    o = summary["overall"]
    for name in ("false_success", "unsupported_success", "unsupported_failure", "false_failure",
                 "over_hedge", "correct_report", "task_success"):
        r = o[name]
        if r["rate"] is not None:
            out[f"{name}_rate"] = r["rate"]
            out[f"{name}_ci95_low"] = r["ci95"][0]
            out[f"{name}_ci95_high"] = r["ci95"][1]
            out[f"{name}_n"] = float(r["n"])
    for name, eff in summary["effects"].items():
        if eff:
            out[f"{name}_difference"] = eff["difference"]
            out[f"{name}_ci95_low"] = eff["ci95"][0]
            out[f"{name}_ci95_high"] = eff["ci95"][1]
    return out


def write_evallogs(records: list[dict], out_dir: str, *, spec_path=SPEC_PATH) -> list[Path]:
    try:
        from importlib.metadata import version
        from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats, SceneResult
        from inspect_robots.logging.json_log import JsonLogSink
        from inspect_robots.scorer import Score, reduce_scores
    except ImportError as e:
        raise SystemExit("evallog needs inspect-robots: pip install inspect-robots==0.58.0") from e

    spec = load(spec_path)
    scored = score(records)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_model[rec["model"]].append(rec)

    paths = []
    for model, recs in sorted(by_model.items()):
        recs.sort(key=lambda r: (r["episode"]["scene_id"], r["episode"]["seed"]))
        scenes: dict[str, list[dict]] = defaultdict(list)
        for rec in recs:
            scenes[rec["episode"]["scene_id"]].append(rec)

        samples = []
        for scene_id, rows in sorted(scenes.items()):
            first = rows[0]["episode"]
            epochs, judgements, sources, metadata, reasons, transcripts = [], [], [], [], [], []
            for rec in rows:
                labels = judge(rec)
                if not labels["scored"]:
                    epochs.append({})
                    judgements.append(None)
                    sources.append(None)
                    metadata.append({"seed": rec["episode"]["seed"], "error": rec.get("error")})
                    reasons.append(None)
                    transcripts.append(None)
                    continue
                epochs.append(_epoch_scores(labels))
                truth = rec["truth"]
                judgements.append("success" if truth["success"] else "failure")
                sources.append("embodiment" if truth.get("decided_by") == "physics" else "prompt")
                metadata.append({"seed": rec["episode"]["seed"], "report": rec.get("report"),
                                 "evidence": rec["evidence"], "truth": truth, "counts": rec.get("counts")})
                reasons.append(rec["end"])
                transcripts.append([{k: v for k, v in st.items() if k in ("n", "phase", "action", "refused", "result")}
                                    for st in rec.get("steps", [])])
            scored_epochs = [e for e in epochs if e]
            reduced = {}
            if scored_epochs:
                for key in SCORE_KEYS:
                    reduced[key] = float(reduce_scores("mean", [Score(value=e[key]) for e in scored_epochs]).value)
            samples.append(SceneResult(
                scene_id=scene_id,
                status="success" if scored_epochs else "error",
                reduced=reduced,
                epochs=tuple(epochs),
                instruction=task_of(spec, first["task"])["instruction"],
                scene_metadata={"task": first["task"], "kind": first["kind"], "camera": first["camera"],
                                "framing": first["framing"], "seeds": [r["episode"]["seed"] for r in rows]},
                operator_judgements=tuple(judgements),
                judgement_sources=tuple(sources),
                trial_metadata=tuple(metadata),
                termination_reasons=tuple(reasons),
                policy_transcripts=tuple(transcripts),
            ))

        backend = next((r["world"].get("backend") for r in recs if r["world"].get("backend")), {}) or {}
        real = backend.get("kind") == "real"
        finished = sorted(r.get("finished_at") or "" for r in recs)
        served = sorted({r.get("model_id") for r in recs if r.get("model_id")})
        latencies = [st["model_seconds"] for r in recs for st in r.get("steps", []) if "model_seconds" in st]
        errors = sum(1 for r in recs if r["end"] == "error")
        log = EvalLog(
            version=EvalLog.SCHEMA_VERSION,
            status="success" if errors == 0 else "error",
            eval=EvalSpec(
                task=spec["format"],
                policy=model,
                embodiment=("Microduck (real)" if real else
                            f"duckbench (simulated Microduck), plant {backend.get('plantName')} "
                            f"{(backend.get('plantDigest') or 'unknown')[:12]}"),
                created=finished[0] if finished and finished[0] else "",
                inspect_robots_version=version("inspect-robots"),
                policy_config={"served_model": served, "sampling": "the endpoint's default",
                               "protocol": "one JSON action per turn; replies are not recorded",
                               "spec_version": spec["version"],
                               "spec_sha256": recs[0].get("spec_sha256")},
                embodiment_info={k: backend.get(k) for k in ("kind", "embodiment", "bench", "plantName",
                                                             "plantDigest", "canonPlant", "tickHz")
                                 if backend.get(k) is not None},
                seed=None,
                max_steps=spec["limits"]["max_actions"],
            ),
            results=EvalResults(
                total_scenes=len(samples),
                total_trials=len(recs),
                metrics=_headline(scored["models"][model]),
                errored_trials=errors,
            ),
            stats=EvalStats(
                started_at=finished[0] if finished else "",
                completed_at=finished[-1] if finished else "",
                duration_s=round(sum(r.get("wall_seconds", 0.0) for r in recs), 3),
                total_steps=sum((r.get("counts") or {}).get("actions", 0) for r in recs),
                mean_inference_latency_s=(sum(latencies) / len(latencies)) if latencies else None,
            ),
            samples=tuple(samples),
        )
        sink = JsonLogSink(str(Path(out_dir)))
        sink.on_eval_end(log)
        paths.append(sink.path)
    return paths
