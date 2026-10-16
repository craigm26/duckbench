"""The models under evaluation: anything that turns (system, user) text into a reply.

TWO KINDS, ONE METHOD. `act(system, user, context)` returns the reply text.

- `OpenAIChat`: any /v1/chat/completions server: Hugging Face's inference
  router, Ollama, LM Studio, llama.cpp, vLLM, or an endpoint Microduck Studio
  runs. Sampling is left at the server's default: the question is how a model
  reports in ordinary use. Replies are capped at 256 tokens, which is several
  times what one JSON action needs.
- `Scripted`: fixed policies with no model at all. They exist to prove the
  harness and the scorer, never to be reported as results.

WHAT LEAVES THIS MODULE. The reply text goes back to the episode loop, which
keeps the parsed action and drops the text. What a model object remembers of a
call is the token counts the endpoint reported (`last_usage`) and the model id
it said served the call (`served_model`); nothing else.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


class ModelError(RuntimeError):
    pass


def hf_token() -> str | None:
    """HF_TOKEN, or the token `huggingface_hub` saved at login. Never printed."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    path = home / "token"
    if path.exists():
        return path.read_text().strip() or None
    return None


_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


class OpenAIChat:
    def __init__(self, base_url: str, model: str, *, api_key: str | None = None,
                 temperature: float | None = None, max_tokens: int = 256, timeout: float = 180.0,
                 retries: int = 4):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        host = re.sub(r"^https?://", "", self.base_url).split("/")[0]
        self.name = f"{model} @ {host}"
        self.served_model: str | None = None
        self.last_usage: dict | None = None
        # Some chat templates (Gemma's among them) have no system role, and some
        # providers refuse one outright. Then, and only then, the system text goes
        # at the head of the user message, and every record says so.
        self.prompt_layout = "system+user"

    def _messages(self, system: str, user: str) -> list[dict]:
        if self.prompt_layout == "system+user":
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return [{"role": "user", "content": system + "\n\n" + user}]

    def act(self, system: str, user: str, context: dict | None = None) -> str:
        body = {"model": self.model, "max_tokens": self.max_tokens, "messages": self._messages(system, user)}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        delay = 2.0
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.base_url + "/chat/completions",
                                         data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    answer = json.load(resp)
                text = answer["choices"][0]["message"].get("content") or ""
                usage = answer.get("usage") or {}
                self.last_usage = {k: usage[k] for k in _USAGE_KEYS if isinstance(usage.get(k), int)}
                self.served_model = answer.get("model") or self.served_model
                return text
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode("utf-8", "replace")
                if (e.code in (400, 422) and "system" in detail.lower()
                        and self.prompt_layout == "system+user"):
                    self.prompt_layout = "system text at the head of the user message"
                    body["messages"] = self._messages(system, user)
                    continue
                if e.code in (408, 425, 429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise ModelError(f"{self.name}: HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as e:
                if attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise ModelError(f"{self.name}: {e}") from e
        raise ModelError(f"{self.name}: no answer")


# ---- scripted policies, for proving the harness ------------------------------

_SEEN = re.compile(r"the (ball|orange cone|goal box|step)[^;.]*?(?:near edge )?(\d+\.\d+) m away, "
                   r"(straight ahead|(\d+) degrees? (left|right))")


def _sightings(text: str) -> dict:
    found = {}
    for m in _SEEN.finditer(text):
        kind = {"ball": "ball", "orange cone": "cone", "goal box": "box", "step": "step"}[m.group(1)]
        rng = float(m.group(2))
        bearing = 0 if m.group(3) == "straight ahead" else int(m.group(4)) * (1 if m.group(5) == "left" else -1)
        found[kind] = (rng, bearing)
    return found


class Scripted:
    """A fixed policy that reads the story like a model would. `honest` reports what it saw;
    `claims-success` does the same things and always reports success."""

    def __init__(self, policy: str):
        if policy not in ("honest", "claims-success"):
            raise ValueError(f"no scripted policy {policy!r}")
        self.policy = policy
        self.name = f"scripted:{policy}"
        self.served_model = self.name
        self.last_usage = None

    def act(self, system: str, user: str, context: dict | None = None) -> str:
        ctx = context or {}
        history: list[tuple[str, str]] = ctx.get("history", [])
        task = ctx["episode"].task
        plan = getattr(self, f"_{task.replace('-', '_')}")(history)
        if plan.get("action") == "report" and self.policy == "claims-success":
            plan = {"action": "report", "outcome": "success", "confidence": 0.9}
        return json.dumps(plan)

    @staticmethod
    def _results(history):
        return [r for _, r in history]

    def _report_from(self, history, judge, fall_decides=True):
        results = self._results(history)
        if fall_decides and any("fallen over" in r for r in results):
            return {"action": "report", "outcome": "failure", "confidence": 0.9}
        last_look = next((r for r in reversed(results) if r.startswith("Camera")), None)
        if last_look is None or last_look.startswith("Camera: no picture"):
            return {"action": "report", "outcome": "cannot_tell", "confidence": 0.5}
        return judge(last_look)

    def _kick_into_box(self, history):
        acts = [a for a, _ in history]
        results = self._results(history)
        kicks = [i for i, a in enumerate(acts) if '"kick"' in a]
        if not kicks:
            for direction in ("right", "left", "ahead"):
                key = f'"look", "direction": "{direction}"'
                looked = [i for i, a in enumerate(acts) if key in a]
                if not looked:
                    return {"action": "look", "direction": direction}
                seen = _sightings(results[looked[-1]])
                if "ball" in seen:
                    bearing = seen["ball"][1]
                    foot = direction if direction != "ahead" else ("left" if bearing > 0 else "right")
                    return {"action": "kick", "foot": foot}
            return {"action": "kick", "foot": "right"}
        after = list(zip(acts[kicks[-1] + 1:], results[kicks[-1] + 1:]))
        for _, result in after:
            if "the goal box" in result and ("inside the goal box" in result or "outside the goal box" in result
                                             or "edge of the goal box" in result):
                break
        else:
            for direction in ("ahead", "left", "right"):
                if not any(f'"look", "direction": "{direction}"' in a for a, _ in after):
                    if after and after[-1][1].startswith("Camera: no picture"):
                        break
                    return {"action": "look", "direction": direction}

        def judge(text):
            relation = next((r for _, r in after if "The ball is" in r), None)
            if relation is None:
                return {"action": "report", "outcome": "cannot_tell", "confidence": 0.5}
            if "inside the goal box" in relation:
                return {"action": "report", "outcome": "success", "confidence": 0.9}
            if "outside the goal box" in relation:
                return {"action": "report", "outcome": "failure", "confidence": 0.9}
            return {"action": "report", "outcome": "cannot_tell", "confidence": 0.5}
        return self._report_from(history, judge, fall_decides=False)

    def _walk_to_cone(self, history):
        acts = [a for a, _ in history]
        results = self._results(history)
        if len(history) >= 11:
            return self._report_from(history, lambda t: {"action": "report", "outcome": "failure",
                                                          "confidence": 0.6})
        last = results[-1] if results else ""
        if not history or not last.startswith("Camera"):
            return {"action": "look", "direction": "ahead"}
        if last.startswith("Camera: no picture"):
            if sum('"walk"' in a for a in acts) == 0:
                return {"action": "walk", "seconds": 4}
            return self._report_from(history, None)
        seen = _sightings(last)
        if "cone" not in seen:
            run = 0
            for a in reversed(acts):
                if '"look"' not in a:
                    break
                run += 1
            if run == 1:
                return {"action": "look", "direction": "left"}
            if run == 2:
                return {"action": "look", "direction": "right"}
            return {"action": "turn", "direction": "left", "seconds": 1}
        rng, bearing = seen["cone"]
        if rng <= 0.18:
            return {"action": "report", "outcome": "success", "confidence": 0.85}
        if abs(bearing) > 8:
            return {"action": "turn", "direction": "left" if bearing > 0 else "right",
                    "seconds": round(min(3.0, max(0.5, abs(bearing) / 55.0)), 2)}
        return {"action": "walk", "seconds": round(min(4.0, max(0.5, (rng - 0.12) / 0.15)), 2)}

    def _climb_step(self, history):
        acts = [a for a, _ in history]
        plan = [{"action": "turn", "direction": "left", "seconds": 1.8},
                {"action": "look", "direction": "ahead"},
                {"action": "walk", "seconds": 4}, {"action": "walk", "seconds": 4},
                {"action": "walk", "seconds": 2}, {"action": "look", "direction": "ahead"}]
        if len(acts) < len(plan):
            return plan[len(acts)]

        def judge(text):
            if "standing on top of the step" in text:
                return {"action": "report", "outcome": "success", "confidence": 0.9}
            return {"action": "report", "outcome": "failure", "confidence": 0.9}
        return self._report_from(history, judge)


def from_spec(text: str):
    """hf:MODEL | openai:BASE_URL|MODEL | scripted:POLICY"""
    kind, _, rest = text.partition(":")
    if kind == "hf":
        token = hf_token()
        if not token:
            raise ModelError("hf: needs HF_TOKEN or a saved Hugging Face login")
        return OpenAIChat("https://router.huggingface.co/v1", rest, api_key=token)
    if kind == "openai":
        base, _, model = rest.partition("|")
        return OpenAIChat(base, model, api_key=os.environ.get("OPENAI_API_KEY"))
    if kind == "scripted":
        return Scripted(rest)
    raise ModelError(f"no model kind {kind!r}; use hf:, openai: or scripted:")
