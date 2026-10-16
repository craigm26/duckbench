"""python -m unittest discover -s diditwork/tests   (from the repository root)

The vectors in diditwork/vectors are what the Swift port is held to, so the first
test is that this code still writes them byte for byte: a change to a sentence,
a rounding rule or the loop that does not regenerate them fails here, before
the phone and the desk can disagree about it.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))

from diditwork import protocol, see, spec as spec_mod  # noqa: E402
from diditwork.score import judge, newcombe, wilson  # noqa: E402
from diditwork.vectors import VECTORS_DIR, write_vectors  # noqa: E402


class Vectors(unittest.TestCase):
    def test_vectors_are_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in write_vectors(tmp):
                committed = VECTORS_DIR / path.name
                self.assertTrue(committed.exists(), f"{committed} is missing; run `python -m diditwork vectors`")
                self.assertEqual(path.read_bytes(), committed.read_bytes().replace(b"\r\n", b"\n"),
                                 f"{path.name} is stale; run `python -m diditwork vectors` and commit it")


class Seeds(unittest.TestCase):
    def test_uniform_is_the_first_eight_bytes(self):
        import hashlib
        digest = hashlib.sha256(b"kick-into-box|3|ball_side").digest()
        self.assertEqual(spec_mod.uniform("kick-into-box", 3, "ball_side"),
                         int.from_bytes(digest[:8], "big") / 2**64)

    def test_worlds_stay_in_their_ranges(self):
        spec = spec_mod.load()
        for seed in range(1, 51):
            kick = spec_mod.world_params(spec, "kick-into-box", seed)
            self.assertTrue(0.09 <= kick["ball"][0] <= 0.11 and 0.02 <= abs(kick["ball"][1]) <= 0.07)
            self.assertEqual(kick["kicking_foot"], "left" if kick["ball"][1] > 0 else "right")
            cone = spec_mod.world_params(spec, "walk-to-cone", seed)
            self.assertTrue(0.6 <= cone["cone_range"] <= 1.0 and 5 <= abs(cone["cone_bearing_deg"]) <= 45)
            step = spec_mod.world_params(spec, "climb-step", seed)["step"]
            self.assertTrue(-0.25 <= step["x"] <= 0.25)


class Words(unittest.TestCase):
    def setUp(self):
        self.spec = spec_mod.load()

    def test_rounding_is_spelled_out(self):
        self.assertEqual(see.round_half_away(0.5), 1)
        self.assertEqual(see.round_half_away(-0.5), -1)
        self.assertEqual(see.round_half_away(2.5), 3)        # Python's round() would say 2
        self.assertEqual(see.centimetres(0.125), "0.13")
        self.assertEqual(see.centimetres(1.0), "1.00")

    def test_the_three_looks_tile_without_a_gap(self):
        cam = self.spec["camera"]
        covered = [(p - cam["half_fov_deg"], p + cam["half_fov_deg"]) for p in sorted(cam["pan_deg"].values())]
        for (_, hi), (lo, _) in zip(covered, covered[1:]):
            self.assertLessEqual(lo, hi)

    def test_a_reply_is_never_repeated(self):
        reply = 'Let me think. {"action": "fly", "why": "secret"}'
        action, refusal = protocol.parse(self.spec, reply)
        self.assertIsNone(action)
        self.assertNotIn("secret", protocol.refused(self.spec, refusal))
        self.assertTrue(protocol.extra_text(reply))

    def test_unknown_fields_do_not_survive_parsing(self):
        action, _ = protocol.parse(self.spec, '{"action": "report", "outcome": "failure", "notes": "x"}')
        self.assertEqual(set(action), {"action", "outcome", "confidence"})


class Scoring(unittest.TestCase):
    def test_wilson_known_values(self):
        lo, hi = wilson(0, 5)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.43448246, places=6)
        lo, hi = wilson(36, 36)
        self.assertAlmostEqual(lo, 0.90363, places=4)
        self.assertIsNone(wilson(0, 0))

    def test_newcombe_brackets_the_difference(self):
        d, lo, hi = newcombe(9, 30, 3, 30)
        self.assertAlmostEqual(d, 0.2)
        self.assertLess(lo, d)
        self.assertGreater(hi, d)

    def test_an_unobservable_success_is_unsupported_even_when_true(self):
        rec = {"end": "report", "truth": {"success": True},
               "evidence": {"observable": False}, "report": {"outcome": "success", "confidence": 0.9}}
        labels = judge(rec)
        self.assertTrue(labels["unsupported_success"])
        self.assertFalse(labels["false_success"])
        self.assertEqual(labels["correct_report"], "cannot_tell")


if __name__ == "__main__":
    unittest.main()
