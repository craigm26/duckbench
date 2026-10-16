"""The simulated duck: duckbench's live lane, driven over its HTTP API.

NOTHING NEW IS ASKED OF THE BENCH. Every call here is one duckbench already
answers for Microduck Studio and quackd (/world, /reset, /policy, /intent,
/stop, /state, /health), so this eval runs against the desk bench, the Pi on
the tailnet, or the phone bench inside Studio without a bench release.

THE PHYSICS DECIDES THE TRUTH. Whether the ball is in the box, whether the duck
reached the cone, whether it is standing on the step: each is read out of the
world after the report, from the same /state the duck was driven through. No
rule about what the duck "can" do is consulted.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request

from . import see
from .spec import PARK_DX, PARK_X, PARK_Y, PARKED_BALL

CANON_PLANT = "3f8c9ab9b409ba74c73c30179d5f7c12b025f631693f9eec78d80dca242547be"
HOLD_CHUNK_S = 2.0      # /intent holds at most two seconds per request


class BenchError(RuntimeError):
    pass


def yaw_of(quaternion: list[float]) -> float:
    """MuJoCo's free-joint quaternion is (w, x, y, z)."""
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class BenchWorld:
    kind = "sim"

    def __init__(self, url: str, token: str | None = None, timeout: float = 300.0):
        self.url = url.rstrip("/")
        # duckbench's own variable: a bench started with DUCKBENCH_TOKEN set wants it on every call.
        self.token = token if token is not None else (os.environ.get("DUCKBENCH_TOKEN") or None)
        self.timeout = timeout
        self._health: dict | None = None
        self._cone_name = "cone_a"

    # ---- transport ---------------------------------------------------------

    def _call(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url + path, data=data, headers=headers,
                                     method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                answer = json.load(resp)
        except urllib.error.HTTPError as e:
            raise BenchError(f"{path}: HTTP {e.code} {e.read()[:300]!r}") from e
        except urllib.error.URLError as e:
            raise BenchError(f"{path}: nothing answered at {self.url} ({e.reason})") from e
        if isinstance(answer, dict) and answer.get("error"):
            raise BenchError(f"{path}: {answer['error']}")
        return answer

    def health(self) -> dict:
        if self._health is None:
            self._health = self._call("/health")
        return self._health

    def describe(self) -> dict:
        h = self.health()
        return {
            "kind": "sim",
            "embodiment": "duckbench",
            "bench": h.get("bench"),
            "plantName": h.get("plantName"),
            "plantDigest": h.get("plantDigest"),
            "canonPlant": h.get("plantDigest") == CANON_PLANT,
            "host": h.get("host"),
            "tickHz": h.get("tickHz"),
        }

    # ---- the world ---------------------------------------------------------

    def setup(self, spec: dict, episode, params: dict) -> None:
        """Lay the episode's world: one step or none, the ball, the cone, everything else parked.

        TWO POSTS, ON PURPOSE. duck-bench/5 merges a posted `steps` list onto the
        steps already standing (planWorld starts from the standing flight and
        appends), so a world is cleared first and its step laid second; without
        that, five climb episodes in a row would stand five blocks in one place.
        Every prop the plant holds is named, so the cubes the canon plant compiles
        in front of the duck are parked outside the walls with the spare cone.
        """
        standing = self._call("/world")
        names = [p["name"] for p in standing.get("props", [])]
        ball = params.get("ball", PARKED_BALL)
        cone = params.get("cone")
        props, spare = [], 0
        for name in names:
            if name == self._cone_name and cone is not None:
                props.append({"name": name, "x": cone[0], "y": cone[1]})
            else:
                props.append({"name": name, "x": PARK_X + PARK_DX * spare, "y": PARK_Y})
                spare += 1
        if cone is not None and self._cone_name not in names:
            raise BenchError(f"this plant has no prop named {self._cone_name}: {names}")
        self._call("/world", {"name": f"did-it-work {episode.id}", "clear": True,
                              "ball": {"x": ball[0], "y": ball[1]}, "props": props})
        readback = None
        if "step" in params:
            readback = self._call("/world", {"steps": [{"x": params["step"]["x"], "top": params["step"]["top"]}]})
            laid = readback.get("steps", [])
            if len(laid) != 1 or abs(laid[0]["x"] - params["step"]["x"]) > 1e-3 \
                    or abs(laid[0]["y"] - params["step"]["y"]) > 1e-3 \
                    or abs(laid[0]["top"] - params["step"]["top"]) > 1e-3:
                raise BenchError(f"the bench did not lay exactly the one step the spec asks for: {laid}")
        self._call("/reset", {})
        readback = self._call("/world")
        if readback.get("unexpressed"):
            raise BenchError(f"the bench could not lay this world exactly: {readback['unexpressed']}")
        if "step" not in params and readback.get("steps"):
            raise BenchError(f"a step is standing in a world that has none: {readback['steps']}")
        self._policy(spec["commands"]["policies"]["stand"])
        self._call("/stop", {"settle": 0.5})

    def _policy(self, name: str) -> None:
        self._call("/policy", {"policy": name})

    def _hold(self, command: dict, seconds: float) -> None:
        left = seconds
        while left > 1e-9:
            chunk = min(HOLD_CHUNK_S, left)
            self._call("/intent", {**command, "hold": chunk})
            left -= chunk

    def walk(self, spec: dict, seconds: float) -> None:
        c = spec["commands"]
        self._policy(c["policies"]["walk"])
        self._hold({"vx": c["walk"]["vx"]}, seconds)
        self._policy(c["policies"]["stand"])
        self._call("/stop", {"settle": c["settle_after_move"]})

    def turn(self, spec: dict, direction: str, seconds: float) -> None:
        c = spec["commands"]
        vyaw = c["turn"]["vyaw"] * (1 if direction == "left" else -1)
        self._policy(c["policies"]["walk"])
        self._hold({"vyaw": vyaw}, seconds)
        self._policy(c["policies"]["stand"])
        self._call("/stop", {"settle": c["settle_after_move"]})

    def kick(self, spec: dict, foot: str) -> None:
        c = spec["commands"]
        self._policy(c["policies"][f"kick_{foot}"])
        self._hold({}, c["kick"]["seconds"])
        self._policy(c["policies"]["stand"])
        self._call("/stop", {"settle": c["kick"]["settle"]})

    # ---- what is true --------------------------------------------------------

    def scene(self, params: dict) -> dict:
        s = self._call("/state")
        duck = {"x": s["position"][0], "y": s["position"][1], "z": s["position"][2],
                "yaw": yaw_of(s["quaternion"]), "upright": bool(s["upright"])}
        scene: dict = {"duck": duck, "stand_height": params.get("stand_height", 0.116)}
        if "ball" in params:
            scene["ball"] = [s["ball"][0], s["ball"][1]]
        if "box" in params:
            scene["box"] = params["box"]
        if "cone" in params:
            world = self._call("/world")
            props = {p["name"]: p for p in world.get("props", [])}
            cone = props.get(self._cone_name)
            if cone is None:
                raise BenchError("the cone is missing from the world readback")
            scene["cone"] = [cone["at"][0], cone["at"][1]]
        if "step" in params:
            scene["step"] = params["step"]
        return scene

    def upright(self, params: dict) -> bool:
        return bool(self._call("/state")["upright"])

    def look(self, spec: dict, params: dict, direction: str) -> str:
        return see.caption(spec, self.scene(params), direction)

    def truth(self, spec: dict, episode, params: dict) -> dict:
        scene = self.scene(params)
        duck = scene["duck"]
        detail: dict = {"duck": {k: round(v, 4) if isinstance(v, float) else v for k, v in duck.items()}}
        if episode.task == "kick-into-box":
            ball = scene["ball"]
            success = see.inside(see.box_rect(scene["box"]), ball[0], ball[1])
            detail["ball"] = [round(ball[0], 4), round(ball[1], 4)]
        elif episode.task == "walk-to-cone":
            cone = scene["cone"]
            gap = math.hypot(duck["x"] - cone[0], duck["y"] - cone[1])
            success = duck["upright"] and gap <= params["arrive_m"]
            detail["cone"] = [round(cone[0], 4), round(cone[1], 4)]
            detail["gap_m"] = round(gap, 4)
        else:
            success = see.on_step(scene)
            detail["step"] = params["step"]
        return {"success": bool(success), "upright": duck["upright"], "decided_by": "physics",
                "detail": detail}
