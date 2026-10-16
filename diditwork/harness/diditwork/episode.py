"""One episode: the loop between a model and a world, and the record it leaves.

THE RECORD IS THE RESULT. Everything the score needs is in the record, and
nothing the score decides is in it: `score.py` reads records and nothing else,
so a scoring rule can be fixed and every run re-scored without driving a duck
again. The record keeps each accepted action or refusal, every sentence the
world answered with, and what the world said was true when the report came in.
It does not keep the model's replies: only the parsed action survives a turn,
with the reply's length and whether it held anything besides the action.

WHAT THE MODEL WAS SHOWN DECIDES WHAT IT COULD KNOW. Whether an episode is
observable is worked out here, from the story itself: a picture after the last
move, no move at all, or a fall the balance sensor reported in a task a fall
decides. It is never inferred from the condition's name.
"""

from __future__ import annotations

import time
import traceback

from . import protocol, see
from .spec import Episode, task_of, world_params

MOVES = ("walk", "turn", "kick")


class EpisodeError(RuntimeError):
    """The harness could not finish the episode; it is not scored and is run again."""


def _execute(spec: dict, world, params: dict, episode: Episode, action: dict, state: dict) -> tuple[str, bool]:
    """Carry out one accepted action. Returns (result text, was it a real picture)."""
    words = spec["words"]
    name = action["action"]
    if name in MOVES:
        if state["fallen"]:
            return words["fallen_cannot_move"] + " " + see.balance(spec, False), False
        if name == "walk":
            world.walk(spec, action["seconds"])
            said = words["walked"].format(s=protocol.seconds_text(action["seconds"]))
        elif name == "turn":
            world.turn(spec, action["direction"], action["seconds"])
            said = words["turned"].format(direction=action["direction"],
                                          s=protocol.seconds_text(action["seconds"]))
        else:
            world.kick(spec, action["foot"])
            said = words["kicked"].format(foot=action["foot"])
        state["moved"] = True
        state["picture_since_move"] = False
        upright = world.upright(params)
        if not upright:
            state["fallen"] = True
        return said + " " + see.balance(spec, upright), False

    # a look
    direction = action["direction"]
    if episode.camera == "fault" and state["moved"]:
        return words["camera_fault"], False
    if state["fallen"]:
        return words["fallen_look"].format(direction=direction), False
    state["picture_since_move"] = True
    return world.look(spec, params, direction), True


def run(spec: dict, world, model, episode: Episode, *, on_step=None) -> dict:
    """Drive one episode to its end and return its record."""
    task = task_of(spec, episode.task)
    limits = spec["limits"]
    params = world_params(spec, episode.task, episode.seed)
    system = protocol.system_prompt(spec, episode.framing)
    history: list[tuple[str, str]] = []
    steps: list[dict] = []
    state = {"moved": False, "picture_since_move": False, "fallen": False}
    accepted = invalid = late_tries = 0
    report = None
    end = None
    started = time.time()

    record = {
        "format": "did-it-work/0 episode",
        "spec_version": spec["version"],
        "episode": {"id": episode.id, "task": episode.task, "seed": episode.seed, "camera": episode.camera,
                    "framing": episode.framing, "kind": task["kind"], "scene_id": episode.scene_id},
        "model": model.name,
        "model_id": None,
        "world": {"params": params},
    }

    try:
        record["world"]["backend"] = world.describe()
        world.setup(spec, episode, params)
        while end is None:
            late = accepted >= limits["max_actions"]
            user = protocol.user_prompt(spec, task["instruction"], history, report_now=late)
            t0 = time.time()
            reply = model.act(system, user, {"episode": episode, "history": list(history), "params": params})
            step = {"n": len(steps) + 1, "phase": "late" if late else "act",
                    "model_seconds": round(time.time() - t0, 3), "reply_chars": len(reply),
                    "extra_text": protocol.extra_text(reply), "usage": getattr(model, "last_usage", None)}
            action, refusal = protocol.parse(spec, reply, report_only=late)
            del reply                      # the action is all that is kept of a turn
            if action is None:
                step["refused"] = refusal
                result = protocol.refused(spec, refusal)
                history.append((spec["words"]["refused_shown"], result))
                step["result"] = result
                if late:
                    late_tries += 1
                    if late_tries >= limits["late_report_tries"]:
                        end = "no_late_report"
                else:
                    invalid += 1
                    if invalid > limits["invalid_replies"]:
                        end = "invalid_replies"
            elif action["action"] == "report":
                step["action"] = action
                report = action
                end = "report"
            else:
                step["action"] = action
                t1 = time.time()
                result, picture = _execute(spec, world, params, episode, action, state)
                step["world_seconds"] = round(time.time() - t1, 3)
                step["result"] = result
                step["picture"] = picture
                history.append((protocol.canonical(action), result))
                accepted += 1
            steps.append(step)
            if on_step:
                on_step(step)

        truth = world.truth(spec, episode, params)
        record["model_id"] = getattr(model, "served_model", None)
    except Exception as e:  # the model endpoint or the world stopped answering
        record.update({"steps": steps, "end": "error", "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc(limit=6),
                       "wall_seconds": round(time.time() - started, 3)})
        return record

    observable = (state["picture_since_move"] or not state["moved"]
                  or (state["fallen"] and task["a_fall_means_failure"]))
    record.update({
        "steps": steps,
        "end": end,
        "report": report,
        "counts": {"actions": accepted, "invalid_replies": invalid, "late_tries": late_tries,
                   "model_calls": len(steps)},
        "evidence": {"moved": state["moved"], "picture_after_last_move": state["picture_since_move"],
                     "told_fallen": state["fallen"], "observable": observable},
        "truth": truth,
        "wall_seconds": round(time.time() - started, 3),
    })
    return record
