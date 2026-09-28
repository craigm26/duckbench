// shoot_test.mjs — the shot's contract, then one live shot per outcome shape.
//   node shoot_test.mjs                 (pure checks only)
//   BENCH=http://localhost:8797 node shoot_test.mjs   (and live shots)
import assert from 'node:assert/strict';
import { checkParams, shootCells, DEFAULTS, LIMITS, PITCH, PHASES } from './shoot_score.mjs';

// Every default is inside its own range, and a parameter nobody has is refused.
for (const [k, [lo, hi]] of Object.entries(LIMITS)) {
  assert.ok(DEFAULTS[k] >= lo && DEFAULTS[k] <= hi, `${k} default outside its limits`);
}
assert.throws(() => checkParams({ vibes: 1 }), /no parameter "vibes"/);
assert.equal(checkParams({ kickDist: 9 }).kickDist, LIMITS.kickDist[1], 'clamped, not refused');
assert.equal(checkParams({ foot: 'left' }).foot, 'left');
assert.throws(() => checkParams({ foot: 'both' }));

// Nine core cells and two harder ones, all inside the room and short of the goal line.
const cells = shootCells();
assert.equal(cells.filter(c => c.tier === 'core').length, 9);
assert.equal(cells.length, 11);
for (const c of cells) assert.ok(c.ball.x < PITCH.goal.x && Math.abs(c.ball.y) < PITCH.walls);
assert.ok(PITCH.goal.x < PITCH.walls, 'the goal line is inside the room');
console.log('shoot_test: pure checks ok');

if (process.env.BENCH) {
  const shot = async (ball, params = {}, sensing = 'state') => {
    const r = await fetch(`${process.env.BENCH}/shoot`, { method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ cell: { ball }, params, sensing }) });
    return r.json();
  };
  const a = await shot({ x: 0.3, y: 0 });
  assert.ok(['goal', 'wide', 'short', 'fell', 'never kicked'].includes(a.outcome), a.outcome);
  assert.equal(a.frames.length, a.roots.length);
  assert.equal(a.frames.length, a.ball.length, 'one ball position per frame');
  assert.ok(a.phases.every(p => PHASES.includes(p.phase)));
  assert.equal(a.goal, a.outcome === 'goal');
  assert.equal(a.plantDigest.length, 64, 'the canon plant, unchanged');
  // Deterministic under perfect sensing: the same shot twice is the same shot.
  const b = await shot({ x: 0.3, y: 0 });
  assert.deepEqual(b.ball[b.ball.length - 1], a.ball[a.ball.length - 1]);
  const refused = await shot({ x: 5, y: 0 });
  assert.match(refused.error, /inside the room/);
  console.log(`shoot_test: live ok (centre-far shot: ${a.outcome}, ${a.seconds} s)`);
}
