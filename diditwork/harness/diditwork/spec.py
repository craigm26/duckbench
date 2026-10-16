"""did-it-work/0, read from spec/did-it-work-0.json, and the seeded worlds it describes.

THE SPEC FILE IS THE ONE SOURCE OF TRUTH. Every sentence the model reads, every
limit an action is checked against and every range a world is drawn from lives
in that JSON, so the Python harness here and the Swift one in Microduck Studio
read the same bytes rather than two copies that can drift. Nothing in this
module restates a number the file holds.

SEEDS ARE SHA-256, NOT A RANDOM GENERATOR. A world is a function of
(task, seed) and nothing else, computed the same way in any language that has
SHA-256: the first eight bytes of SHA-256('<task>|<seed>|<name>') as a
big-endian integer, over 2**64. Python's `random` and Swift's generators do
not agree with each other, and a seed that draws a different world on the
phone than on the desk would be two evaluations wearing one name.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = PACKAGE_ROOT / "spec" / "did-it-work-0.json"

TASK_IDS = ("kick-into-box", "walk-to-cone", "climb-step")
CAMERAS = ("ok", "fault")
FRAMINGS = ("neutral", "pressure")

# Where the things a task does not use are put. The ball is a permanent body and
# cannot leave the arena, so it waits in the far corner behind the duck, where no
# task takes it; every prop the task does not use (the spare cone and the three
# cubes the canon plant compiles in front of the duck) stands outside the walls,
# in a row from PARK_X along x.
PARKED_BALL = (-1.30, -1.30)
PARK_X, PARK_DX, PARK_Y = 3.0, 0.2, 3.0


def load(path: Path | str = SPEC_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    if spec.get("format") != "did-it-work/0":
        raise ValueError(f"{path} is not a did-it-work/0 spec")
    return spec


def spec_digest(path: Path | str = SPEC_PATH) -> str:
    """SHA-256 of the spec with CRLF read as LF, so a Windows checkout names the same spec.

    Git on Windows (core.autocrlf) writes this file with CRLF; JSON never holds a
    raw carriage return inside a string, so folding CRLF to LF changes nothing the
    spec says and only undoes what the checkout did.
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def task_of(spec: dict, task_id: str) -> dict:
    for task in spec["tasks"]:
        if task["id"] == task_id:
            return task
    raise KeyError(f"no task {task_id!r} in {spec['format']}")


def uniform(task_id: str, seed: int, name: str) -> float:
    digest = hashlib.sha256(f"{task_id}|{seed}|{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def draw(task_id: str, seed: int, name: str, lo: float, hi: float) -> float:
    return lo + uniform(task_id, seed, name) * (hi - lo)


def sign(task_id: str, seed: int, name: str) -> float:
    return 1.0 if uniform(task_id, seed, name) < 0.5 else -1.0


@dataclass(frozen=True)
class Episode:
    """One run of one task: which world, which camera, which framing."""

    task: str
    seed: int
    camera: str
    framing: str

    def __post_init__(self) -> None:
        if self.task not in TASK_IDS:
            raise ValueError(f"unknown task {self.task!r}")
        if self.camera not in CAMERAS:
            raise ValueError(f"unknown camera condition {self.camera!r}")
        if self.framing not in FRAMINGS:
            raise ValueError(f"unknown framing {self.framing!r}")
        if self.seed < 0:
            raise ValueError("a seed is a non-negative integer")

    @property
    def id(self) -> str:
        return f"{self.task}-s{self.seed:03d}-{self.camera}-{self.framing}"

    @property
    def scene_id(self) -> str:
        """The scene an EvalLog files this episode under: everything but the seed."""
        return f"{self.task}-{self.camera}-{self.framing}"


def r6(value: float) -> float:
    """Six decimals: well below anything physical here, and stable in JSON."""
    return round(value, 6)


def world_params(spec: dict, task_id: str, seed: int) -> dict:
    """The world a (task, seed) pair describes, in the world frame.

    The duck stands at the origin facing +x after a reset, so "ahead" is +x and
    "left" is +y.
    """
    world = task_of(spec, task_id)["world"]
    if task_id == "kick-into-box":
        ahead = draw(task_id, seed, "ball_ahead", *world["ball_ahead_m"])
        side = draw(task_id, seed, "ball_side", *world["ball_side_m"])
        side *= sign(task_id, seed, "ball_side_sign")
        box = world["box"]
        return {
            "ball": [r6(ahead), r6(side)],
            "box": {"x_min": box["x_min"], "x_max": box["x_max"], "half_width": box["half_width"]},
            "kicking_foot": "left" if side > 0 else "right",
        }
    if task_id == "walk-to-cone":
        rng = draw(task_id, seed, "cone_range", *world["cone_range_m"])
        bearing = draw(task_id, seed, "cone_bearing", *world["cone_bearing_deg"])
        bearing *= sign(task_id, seed, "cone_bearing_sign")
        a = math.radians(bearing)
        return {
            "cone": [r6(rng * math.cos(a)), r6(rng * math.sin(a))],
            "cone_range": r6(rng),
            "cone_bearing_deg": r6(bearing),
            "arrive_m": world["arrive_m"],
        }
    if task_id == "climb-step":
        x = draw(task_id, seed, "step_x", *world["step_x_m"])
        return {
            "step": {
                "x": r6(x),
                "y": world["step_y_m"],
                "top": world["step_top_m"],
                "half_depth": world["step_half_depth_m"],
                "half_width": world["step_half_width_m"],
            },
            "stand_height": world["stand_height_m"],
        }
    raise KeyError(task_id)


def episodes(tasks=TASK_IDS, seeds=range(1, 6), cameras=CAMERAS, framings=FRAMINGS) -> list[Episode]:
    """Every combination, ordered so that a partial run still covers every condition."""
    out = []
    for seed in seeds:
        for task in tasks:
            for camera in cameras:
                for framing in framings:
                    out.append(Episode(task, seed, camera, framing))
    return out
