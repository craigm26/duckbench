"""The text protocol: what a model is sent each turn, and how its reply is read.

STATELESS ON PURPOSE. Every turn sends the whole story so far in one user
message, under one system message. A chat endpoint, Apple's on-device model and
a 1.5B model on a phone all accept exactly that, so one protocol covers every
model this eval will be pointed at, and nothing depends on a provider's
tool-calling format.

ONE JSON OBJECT, AND ONLY THE ACTION IS KEPT. The model is asked for one JSON
object and no other text. The first balanced {...} in a reply is read as the
action; nothing else in the reply is used, stored or repeated back. A reply with
no object, or with an object this protocol does not accept, is refused with a
sentence that says why, and the story the model reads next shows the refusal,
never the reply.
"""

from __future__ import annotations

import json
import math
import re

ACTIONS = ("walk", "turn", "kick", "look", "report")
OUTCOMES = ("success", "failure", "cannot_tell")


def system_prompt(spec: dict, framing: str) -> str:
    return "\n".join(spec["prompts"]["system"]) + "\n\n" + spec["prompts"]["framing"][framing]


def history_text(spec: dict, history: list[tuple[str, str]]) -> str:
    prompts = spec["prompts"]
    if not history:
        return prompts["history_empty"]
    lines = [prompts["history_heading"]]
    for n, (action, result) in enumerate(history, start=1):
        lines.append(prompts["history_line"].format(n=n, action=action, result=result))
    return "\n".join(lines)


def user_prompt(spec: dict, instruction: str, history: list[tuple[str, str]], *, report_now: bool = False) -> str:
    template = "\n".join(spec["prompts"]["user"])
    text = template.format(instruction=instruction, history=history_text(spec, history))
    if report_now:
        last = text.rfind("\n")
        text = text[:last] + "\n" + spec["words"]["budget"] + text[last:]
    return text


def first_json_object(reply: str) -> tuple[int, int] | None:
    """Where the first balanced {...} in the reply starts and ends, honouring strings and escapes."""
    start = reply.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(reply)):
            c = reply[i]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
            elif c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return start, i + 1
        start = reply.find("{", start + 1)
    return None


_SPACE = " \t\n\r\f\v"


def extra_text(reply: str) -> bool:
    """Did the reply hold anything besides its JSON object? Counted for format compliance, never kept."""
    span = first_json_object(reply)
    if span is None:
        return bool(reply.strip(_SPACE))
    return bool(reply[:span[0]].strip(_SPACE) or reply[span[1]:].strip(_SPACE))


def _not_json(name: str):
    """NaN and Infinity are not JSON; Python's reader would otherwise accept them."""
    raise ValueError(f"{name} is not JSON")


def _double(text: str) -> float:
    """Every JSON number is read as an IEEE 754 double, and one that does not fit is not JSON."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text[:20]} does not fit in a double")
    return value


def depth(text: str) -> int:
    """The deepest nesting of objects and arrays in a JSON text, outside its strings."""
    deepest = level = 0
    in_string = escaped = False
    for c in text:
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c in "{[":
            level += 1
            deepest = max(deepest, level)
        elif c in "}]":
            level -= 1
    return deepest


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def canonical(action: dict) -> str:
    """The action as the story repeats it: `action` first, then its fields, in a fixed order.

    Written out by hand rather than by a JSON encoder, because encoders disagree
    about 1 versus 1.0 and the Swift port has to produce these bytes exactly.
    Seconds and confidence are printed as C's %g prints them. A report is never
    repeated in the story (it ends the episode); its form here is for the record.
    """
    name = action["action"]
    if name == "walk":
        return '{"action": "walk", "seconds": %s}' % seconds_text(action["seconds"])
    if name == "turn":
        return '{"action": "turn", "direction": "%s", "seconds": %s}' % (
            action["direction"], seconds_text(action["seconds"]))
    if name == "kick":
        return '{"action": "kick", "foot": "%s"}' % action["foot"]
    if name == "look":
        return '{"action": "look", "direction": "%s"}' % action["direction"]
    if action.get("confidence") is None:
        return '{"action": "report", "outcome": "%s"}' % action["outcome"]
    return '{"action": "report", "outcome": "%s", "confidence": %s}' % (
        action["outcome"], seconds_text(action["confidence"]))


def parse(spec: dict, reply: str, *, report_only: bool = False) -> tuple[dict | None, dict | None]:
    """(action, None) for an accepted reply; (None, refusal) for a refused one.

    A refusal is {"refusal": key} with the key into the spec's refusals, and for a
    seconds refusal {"about": "walk" or "turn"} as well.


    Fields the protocol does not name are ignored and dropped, so nothing a model
    adds to its action object travels any further than this function.
    """
    commands = spec["commands"]
    span = first_json_object(reply)
    if span is None:
        return None, {"refusal": "no_json"}
    text = reply[span[0]:span[1]]
    if depth(text) > spec["protocol"]["max_depth"]:
        return None, {"refusal": "no_json"}
    try:
        obj = json.loads(text, parse_constant=_not_json, parse_int=_double, parse_float=_double)
    except ValueError:
        return None, {"refusal": "no_json"}
    if not isinstance(obj, dict):
        return None, {"refusal": "not_an_object"}
    name = obj.get("action")
    if name not in ACTIONS:
        return None, {"refusal": "unknown_action"}
    if report_only and name != "report":
        return None, {"refusal": "report_only"}

    if name in ("walk", "turn"):
        lo, hi = commands[name]["seconds"]
        seconds = _number(obj.get("seconds"))
        if seconds is None or not (lo <= seconds <= hi):
            return None, {"refusal": "seconds", "about": name}
        if name == "walk":
            return {"action": "walk", "seconds": seconds}, None
        direction = obj.get("direction")
        if direction not in ("left", "right"):
            return None, {"refusal": "turn_direction"}
        return {"action": "turn", "direction": direction, "seconds": seconds}, None
    if name == "kick":
        foot = obj.get("foot")
        if foot not in ("left", "right"):
            return None, {"refusal": "foot"}
        return {"action": "kick", "foot": foot}, None
    if name == "look":
        direction = obj.get("direction", "ahead")
        if direction not in ("ahead", "left", "right"):
            return None, {"refusal": "look_direction"}
        return {"action": "look", "direction": direction}, None

    outcome = obj.get("outcome")
    if outcome not in OUTCOMES:
        return None, {"refusal": "outcome"}
    confidence = obj.get("confidence")
    if confidence is not None:
        confidence = _number(confidence)
        if confidence is None or not (0.0 <= confidence <= 1.0):
            return None, {"refusal": "confidence"}
    return {"action": "report", "outcome": outcome, "confidence": confidence}, None


def refused(spec: dict, refusal: dict) -> str:
    """The refusal sentence the model reads, from the spec's words for that refusal."""
    why = spec["refusals"][refusal["refusal"]]
    if refusal["refusal"] == "seconds":
        lo, hi = spec["commands"][refusal["about"]]["seconds"]
        why = why.format(action=refusal["about"], lo=f"{lo:g}", hi=f"{hi:g}")
    return spec["words"]["refused"].format(why=why)


def seconds_text(seconds: float) -> str:
    """C's %g: six significant digits, trailing zeros dropped."""
    return "%g" % seconds
