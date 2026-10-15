# Duck Cards closed on odometry: what one Walk and one Turn card do

`../cardloop.mjs`, 2026-10-15. The question: can a Duck Card's Walk mean "one board square" and
its Turn "a quarter turn" on a duck, if the card closes on what robotd reports rather than
running for two seconds? Measured on this bench's canon plant (`scene.mjb`, Pollen's MJCF with
their servo class, sha256 `3f8c9ab9b409`) under `alpha_walking.onnx`, through the bench's own
live lane (`/intent`, one tick at a time). **A simulator, not a duck.**

## What the controller reads, and what scores it

- **Control** sees only what robotd's `robot.state` carries: `odom.position` and `odom.yaw`, and
  whether the duck is up. Odometry is computed the way Pollen's `odometry` crate computes it
  (contact anchor on the ALPHA16 sole grid, the foot's pose from forward kinematics of the
  reported joints on a second copy of the plant, rotated by the trunk orientation as the IMU's,
  anchor switch at 10 mm for 2 ticks), not read from the root. Commands go at 20 Hz and act on a
  state 40 ms old; `STATE_EVERY=5` holds each state for 100 ms, which is Studio's 10 Hz
  subscription.
- **Scoring** uses the free joint (ground truth): where the duck ended against the board rule at
  30 cm a square, in the row's own frame, and the heading against the board's.

The gap that matters most: the odometry here is fed exact joint angles (to 1e-4 rad) and an exact
trunk orientation. A real IMU's yaw drifts (no magnetometer) and real servos read with noise, and
real feet slip differently from MuJoCo's contacts. In this plant odometry ended rows within
2.6 cm of the truth (median 1.1 cm); nobody has measured that on a duck.

## The spec

`../duck-card-loop.json`, copied byte for byte from duck-studio's
`StudioKit/Sources/StudioKit/Resources/DuckCards/duck-card-loop.json`, which Studio
(`DuckCardLoop`) and the card reader (`cardloop.py`) both run. docs/PACKS.md in duck-studio
("Closed-loop cards") says it in words. **No card commands more than Studio's stick envelope**
(`DuckDrive.maxForward` 0.3, `maxBackward` 0.3, `maxTurn` 1.5), because nobody has driven a real
duck past it; both readers' tests and `check_duck_packs.py` hold the spec to that. The numbers: a
30 cm square; walk at 0.25, 0.30, 0.30 (Slow, Normal, Fast), back at 0.30; heading gain 1.5,
cross-track gain 3.0, steering clamped to 0.6; stop 1.5 cm early (7 cm walking back); give up
after 6 s; turn on the spot at 1.5, stop 4 degrees early, settle 0.5 s, done within 6 degrees,
else up to 2 trims; 0.5 s still between cards.

## The result

Seven rows, ten starts each (0 to 0.9 s of standing first, so the gait starts at different
phases), at both state rates: 140 rows, 700 cards.

| row | cards |
|---|---|
| course | F F L F R F |
| square | F R F R F R F R |
| uturn | F L L F |
| back | F F B |
| speeds | F(slow) F(fast) L(slow) F R(fast) B(slow) |
| turns | L L R R |
| walk4 | F F F F |

| | 50 Hz state | 10 Hz state | both |
|---|---|---|---|
| falls | 0 of 70 | 0 of 70 | **0 of 140** |
| Walk forward and Turn cards that did not arrive | 0 of 330 | 0 of 330 | 0 of 660 |
| Walk back cards that did not arrive | 20 of 20 | 20 of 20 | **40 of 40** |
| position error, rows without Walk back | median 2.7, max 5.7 cm | median 2.8, max 6.1 cm | median 2.7, **max 6.1 cm** (100 rows) |
| position error, rows with Walk back | median 30.5, max 31.4 cm | median 30.7, max 32.9 cm | median 30.6, max 32.9 cm (40 rows) |
| heading error, rows without Walk back | median 2.2, max 9.4 degrees | median 3.4, max 9.6 | median 2.7, **max 9.6** |
| heading error, rows with Walk back | median 1.7, max 8.3 | median 2.7, max 10.4 | median 1.9, max 10.4 |
| one Walk forward card, ground truth | median 30.5 cm [25.6 to 35.2] | median 30.7 cm [24.8 to 34.0] | |
| one Turn card, from the card's own start | median 90.3 degrees [78.8 to 97.7] | median 91.6 [78.3 to 98.8] | |
| seconds per Walk forward card | median 2.5, max 2.9 (3.5 at Slow) | median 2.5, max 2.8 (3.5 at Slow) | |
| seconds per Turn card | median 2.4, max 2.6 | median 2.4, max 3.8 | |

**Walk back does not walk at the stick's 0.3.** Every one of the 40 Walk back cards marched on
the spot (at most 0.3 cm), ran out its 6 s and ended "out of time", which the reader and Studio
say ("The card ran out of time before it got there."); the duck did not fall, and the row went
on a square short. The canon plant walks backward only from about 0.45, which this spec will
not send. Walk forward at 0.30 instead of 0.35 ends rows a little further off (max 6.1 cm
against 4.6); 0.25 (Slow) still walks, at about 8.6 cm/s.

Plus 0.5 s still after every card. A Turn card's own angle runs past 90 degrees when it takes
back the heading the card before it left (the board is what it aims at), which is why single
cards spread more than rows end.

Files: `results-50hz.json`, `results-10hz.json` (every row, every card).
`first-spec-results-*.json` are the same rows under the first spec (0.35 forward, 0.45 back),
which ended within 4.6 cm (6.4 with Walk back): kept as a measurement, not shipped, because
those speeds are past the stick.

## Choosing the square, and the controller

`sweep-square-and-gains.log` (and `.json`): an earlier spec (back stop 1.5 cm early, trims
stopping 8 degrees short), ten starts per row, square sizes 20, 25, 30 and 40 cm, and variants
at 25 cm. Rows course / square / uturn / back, end-of-row error median/max in cm:

| square | course | square | uturn | back |
|---|---|---|---|---|
| 20 cm | 1.6 / 2.4 | 2.3 / 3.5 | 2.8 / 4.4 | 5.8 / 8.2 |
| 25 cm | 1.4 / 2.6 | 2.0 / 3.6 | 2.6 / 3.4 | 6.4 / 7.8 |
| 30 cm | 1.9 / 2.5 | 2.5 / 3.4 | 2.4 / 3.8 | 6.5 / 8.1 |
| 40 cm | 2.5 / 2.7 | 3.2 / 4.0 | 2.0 / 2.8 | 7.0 / 7.5 |

No falls in any of them. The error does not grow with the square, so the square was chosen for
the floor: 30 cm is every starter pack's printed square (one card, one taped square), holds a
25 cm duck with room, and a Walk card takes about two seconds. At 25 cm: walking at 0.30 or 0.45
instead of 0.35, gains halved, the turn stopping 2 degrees short, made no clear difference;
**turning at 1.2 failed** (34 cards in 40 rows never arrived: below about 1.2 the duck does not
turn on the spot). Walking back coasted further than forward, so `sweep-30cm-trims-and-back.log`
tried stopping it 7 cm early: back-row error went from 3.5/5.6 to 1.9/4.3 cm. Trims stopping
3 rather than 8 degrees short changed nothing (no turn needed a trim), and 3 was kept because a
trim that stops wider than the tolerance can never fix anything.

## The open-loop facts behind it

`probe.json` (`node cardloop.mjs probe`): on the canon plant, forward commands below about 0.25
march on the spot (0.2 moves 3 mm/s), 0.25 walks 8.6 cm/s, 0.3 12 cm/s, 0.45 19.5 cm/s, veering
3 to 17 degrees in 3 s; backward needs about 0.45 (0.3 does not move; 0.45 walks 24 cm/s); on the
spot, yaw 0.5 turns 5 degrees and 1.0 turns 12 and stops (the saturation `walk_to.py` saw), but
1.5 turns 94 degrees in 2 s and moves the duck 2 cm. Odometry tracked forward walking to 1 or 2
percent.

`canon-timed.json` (`node cardloop.mjs canon`): `cardsize.mjs`'s timed cards (2.0 s at Studio's
twists, rows of four, ten starts) on the canon plant instead of the retired `scene.xml`. Slow and
Normal move nothing (walks 0 to 1 cm, turns 0 to 52 degrees, median 11 or less); Fast Walk forward
26 cm [22.6 to 26.9]; Fast turns 88 and 95 degrees; Walk back does not move at any speed; no
falls in 120 rows. `cardsize.mjs`'s numbers (14 cm, falling turns) are the invented plant's.

## Not measured

- Pollen's duck-sim (`robotd --sim` with its own MuJoCo), where robotd itself computes `odom` and
  switches to its standing policy inside the command dead zone. Here the walking policy keeps
  running at a zero command between cards.
- A real duck: IMU yaw drift, servo noise, foot slip on a classroom floor, battery sag.
- A Walk back that walks: at the stick's 0.3 the canon plant does not move backward. Whether a
  real duck does is not measured.
