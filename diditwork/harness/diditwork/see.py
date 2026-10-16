"""What the duck's camera and balance sensor say, as the sentences the model reads.

ONE FUNCTION FROM A SCENE TO A SENTENCE, AND NOTHING ELSE. A scene is a plain
dict of where things are; the bench fills it from its ground truth, and a real
duck fills it from what its own detector found. Either way the caption is
built here, so the words a model reads in simulation are the words it would
read on hardware, and the Swift port is held to them by vectors/see.json.

ROUNDING IS SPELLED OUT, NOT INHERITED. Python's round() rounds half to even and
Swift's rounded() rounds half away from zero, so neither is used on anything a
model reads. Bearings round half away from zero to the whole degree; ranges
round half up to the centimetre.
"""

from __future__ import annotations

import math


def round_half_away(value: float) -> int:
    return int(math.floor(abs(value) + 0.5)) * (1 if value >= 0 else -1)


def centimetres(value: float) -> str:
    """A non-negative distance in metres, rounded half up to the centimetre, as text."""
    cm = int(math.floor(value * 100 + 0.5))
    return f"{cm // 100}.{cm % 100:02d}"


def wrap_degrees(angle: float) -> float:
    """Into (-180, 180]."""
    a = math.fmod(angle, 360.0)
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    return a


def relative(duck: dict, point: tuple[float, float]) -> tuple[float, float]:
    """Horizontal range in metres and bearing in degrees (left positive) from the duck's trunk."""
    dx, dy = point[0] - duck["x"], point[1] - duck["y"]
    rng = math.hypot(dx, dy)
    bearing = wrap_degrees(math.degrees(math.atan2(dy, dx) - duck["yaw"]))
    return rng, bearing


def box_rect(box: dict) -> tuple[float, float, float, float]:
    return box["x_min"], box["x_max"], -box["half_width"], box["half_width"]


def step_rect(step: dict) -> tuple[float, float, float, float]:
    return (step["x"] - step["half_depth"], step["x"] + step["half_depth"],
            step["y"] - step["half_width"], step["y"] + step["half_width"])


def inside(rect: tuple[float, float, float, float], x: float, y: float) -> bool:
    return rect[0] <= x <= rect[1] and rect[2] <= y <= rect[3]


def nearest(rect: tuple[float, float, float, float], x: float, y: float) -> tuple[float, float]:
    return min(max(x, rect[0]), rect[1]), min(max(y, rect[2]), rect[3])


def on_step(scene: dict) -> bool:
    """Upright, trunk over the step's footprint, and high enough to be standing on it."""
    step, duck = scene.get("step"), scene["duck"]
    if not step or not duck["upright"]:
        return False
    high = duck["z"] >= scene["stand_height"] + 0.75 * step["top"]
    return high and inside(step_rect(step), duck["x"], duck["y"])


def bearing_words(words: dict, bearing: float) -> str:
    degrees = round_half_away(bearing)
    if degrees == 0:
        return words["ahead"]
    side = "left" if degrees > 0 else "right"
    if abs(degrees) == 1:
        return words["bearing_one"].format(side=side)
    return words["bearing"].format(deg=abs(degrees), side=side)


def ball_in_box(spec: dict, ball: list[float], box: dict) -> str:
    """inside, outside or edge, from the ball's centre against the box's outline."""
    x_min, x_max, y_min, y_max = box_rect(box)
    depth = min(ball[0] - x_min, x_max - ball[0], ball[1] - y_min, y_max - ball[1])
    if abs(depth) < spec["perception"]["box_edge_band_m"]:
        return "edge"
    return "inside" if depth > 0 else "outside"


def recognised(spec: dict, scene: dict, direction: str) -> list[tuple[str, str]]:
    """(kind, phrase) for every thing in this picture, in the spec's order."""
    camera, words = spec["camera"], spec["words"]
    pan = camera["pan_deg"][direction]
    duck = scene["duck"]

    def in_view(rng: float, bearing: float) -> bool:
        return rng <= camera["max_range_m"] and abs(wrap_degrees(bearing - pan)) <= camera["half_fov_deg"]

    found: list[tuple[str, str]] = []
    for kind in spec["perception"]["order"]:
        thing = scene.get(kind)
        if not thing:
            continue
        if kind in ("ball", "cone"):
            rng, bearing = relative(duck, (thing[0], thing[1]))
            if in_view(rng, bearing):
                found.append((kind, words[kind].format(r=centimetres(rng), b=bearing_words(words, bearing))))
        elif kind == "box":
            rect = box_rect(thing)
            if inside(rect, duck["x"], duck["y"]):
                found.append((kind, words["box_around"]))
                continue
            rng, bearing = relative(duck, nearest(rect, duck["x"], duck["y"]))
            if in_view(rng, bearing):
                found.append((kind, words["box"].format(r=centimetres(rng), b=bearing_words(words, bearing))))
        elif kind == "step":
            if on_step(scene):
                found.append((kind, words["on_step"]))
                continue
            rect = step_rect(thing)
            rng, bearing = relative(duck, nearest(rect, duck["x"], duck["y"]))
            if in_view(rng, bearing):
                height = str(round_half_away(thing["top"] * 100))
                found.append((kind, words["step"].format(r=centimetres(rng),
                                                         b=bearing_words(words, bearing), h=height)))
    return found


def caption(spec: dict, scene: dict, direction: str) -> str:
    """The camera's whole answer to one look."""
    words = spec["words"]
    heading = words["camera_heading"].format(direction=direction)
    found = recognised(spec, scene, direction)
    if not found:
        return heading + words["nothing_seen"]
    text = heading + "; ".join(phrase for _, phrase in found) + "."
    kinds = {kind for kind, _ in found}
    if "ball" in kinds and "box" in kinds:
        relation = ball_in_box(spec, scene["ball"], scene["box"])
        text += " " + words[{"inside": "ball_in_box", "outside": "ball_out_of_box",
                             "edge": "ball_on_box_edge"}[relation]]
    return text


def balance(spec: dict, upright: bool) -> str:
    return spec["words"]["upright" if upright else "fallen"]
