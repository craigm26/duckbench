// cardloop: Duck Cards' Walk and Turn closed on what robotd reports, measured
// on this bench's canon plant (scene.mjb, Pollen's own MJCF with their servo
// class) under alpha_walking, through the bench's own live lane.
//
// cardsize.mjs measured the open-loop cards: a timed Walk forward carries the
// duck about 14 cm and turns it 11 to 15 degrees; a timed Turn lands anywhere
// and topples it. (It ran on scene.xml, the invented floor-and-box-soles plant
// with kp 9 servos, which this repo replaced as its canon on 2026-08-27; see
// the `canon` mode below for the same open-loop cards on the canon plant.)
//
// Here a Walk card walks until robotd's odometry says the duck has covered one
// duck square along the card's heading, holding that heading and steering back
// onto the line; a Turn card turns until the yaw is a quarter turn from the
// card's heading, arcing forward because the gait does not turn on the spot,
// then stops, settles and trims.
//
// WHAT THE CONTROLLER MAY READ. Only what robotd's `robot.state` carries:
// `odom.position` and `odom.yaw` (contact odometry, Pollen's `odometry` crate,
// whose heading is the trunk IMU's yaw) and whether the duck is up. Odometry is
// computed here the way that crate computes it (class `Odom`), from the joint
// angles and the trunk orientation the bench reports, not read from the
// simulator's root position. The root position (ground truth) only scores.
//
// THE BOARD FRAME. A row remembers where the duck should be on the board: its
// start pose, then one square or one quarter turn per card, in odometry's frame.
// Each card aims at the board pose, not at "one more square from wherever the
// last card left the duck", so an overshoot or an arc is taken back by the next
// card instead of adding up. This is the spec in duck-studio's docs/PACKS.md
// ("Closed-loop cards"); SPEC below is its numbers, from duck-card-loop.json.
//
// Usage (one process; the Pi reboots under load, so run it niced):
//   nice -n 19 node cardloop.mjs probe            # open-loop facts the controller needs
//   nice -n 19 node cardloop.mjs canon            # cardsize's timed cards, on the canon plant
//   VARIANTS='[...]' DS=0.2,0.3 ROWS=course nice -n 19 node cardloop.mjs sweep > out.json
//   D=0.3 nice -n 19 node cardloop.mjs rows > out.json   # SPEC, every row, ten starts
import load from 'mujoco';
import fs from 'node:fs';
import { nodeBench } from './duckbench-node.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const DT = 1 / C.tickHz;
const DEG = 180 / Math.PI;
const wrap = a => { while (a > Math.PI) a -= 2 * Math.PI; while (a <= -Math.PI) a += 2 * Math.PI; return a; };
const yawOf = ([w, x, y, z]) => Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));

// ------------------------------------------------------------ the bench
const bench = await nodeBench();
const ask = (path, body = {}) => bench.handle(new URL('http://bench.local' + path), body);
const POLICY = process.env.POLICY || 'alpha_walking.onnx';

// ------------------------------------------------------------ odometry
// Pollen's `odometry` crate (pollen-robotics/microduck, odometry/src/lib.rs),
// step for step: one sole point is the ground anchor at world Z = 0; the trunk
// is placed from it by that foot's pose in the trunk frame (forward kinematics
// of the measured joints) rotated by the IMU's orientation; a different sole
// point more than 10 mm below the anchor's plane for 2 ticks becomes the anchor,
// keeping its current world X/Y. The candidates are Pollen's ALPHA16 grid,
// in each foot site's frame, which is the production set (odometry/src/anchors.rs).
// The plant is Pollen's MJCF, so the `left_foot` and `right_foot` sites are
// theirs too. Heading is the IMU's yaw, as robotd's. The FK runs on a second
// copy of the plant with the trunk at the origin, which is what the
// `kinematics` crate computes from the same MJCF.
const ALPHA16 = {
  left: [[-0.0183, -0.01951, 0.005894], [-0.0015, -0.019659, 0.001516], [0.0155, -0.019659, 0.002082], [0.031699, -0.019061, 0.005531],
    [-0.0185, -0.00696, 0.002891], [-0.0015, -0.00696, -0.000607], [0.0155, -0.00696, -0.000573], [0.0325, -0.00696, 0.002863],
    [-0.0185, 0.00574, 0.003709], [-0.0015, 0.00574, 0.0005], [0.0155, 0.00574, 0.000474], [0.0325, 0.00574, 0.003683],
    [-0.0183, 0.01829, 0.007998], [-0.0015, 0.018439, 0.003859], [0.0155, 0.018439, 0.004318], [0.031699, 0.017841, 0.007574]],
  right: [[-0.0183, -0.01829, 0.007998], [-0.0015, -0.018439, 0.003859], [0.0155, -0.018439, 0.00432], [0.031699, -0.017841, 0.007574],
    [-0.0185, -0.00574, 0.003712], [-0.0015, -0.00574, 0.0005], [0.0155, -0.00574, 0.000474], [0.0325, -0.00574, 0.003681],
    [-0.0185, 0.00696, 0.002893], [-0.0015, 0.00696, -0.000607], [0.0155, 0.00696, -0.000573], [0.0325, 0.00696, 0.002863],
    [-0.0183, 0.01951, 0.005894], [-0.0015, 0.019659, 0.001516], [0.0155, 0.019659, 0.00208], [0.031699, 0.019061, 0.005531]],
};
const SWITCH_MARGIN = -0.010, SWITCH_CONFIRM_TICKS = 2;

const mj = await load();
mj.FS.writeFile('/fk.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const fkModel = mj.MjModel.mj_loadBinary('/fk.mjb', new mj.MjVFS());
const fk = new mj.MjData(fkModel);
const byName = (n, read, name) => { for (let i = 0; i < n; i++) if (read(i) === name) return i; throw new Error('missing ' + name); };
const FK_FREE = fkModel.jnt_qposadr[byName(fkModel.njnt, i => fkModel.jnt(i).name, 'trunk_base_freejoint')];
const FK_JOINTS = C.jointNames.filter(n => n !== 'mouth').map(n => fkModel.jnt_qposadr[byName(fkModel.njnt, i => fkModel.jnt(i).name, n)]);
const FK_SITES = ['left_foot', 'right_foot'].map(n => byName(fkModel.nsite, i => fkModel.site(i).name, n));

function quatRot([w, x, y, z], v) {
  const n = Math.hypot(w, x, y, z); w /= n; x /= n; y /= n; z /= n;
  return [
    (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - w * z) * v[1] + 2 * (x * z + w * y) * v[2],
    2 * (x * y + w * z) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - w * x) * v[2],
    2 * (x * z - w * y) * v[0] + 2 * (y * z + w * x) * v[1] + (1 - 2 * (x * x + y * y)) * v[2]];
}
/** Both foot sites' poses in the trunk frame, from the joint angles. */
function feetInTrunk(joints) {
  fk.qpos[FK_FREE] = 0; fk.qpos[FK_FREE + 1] = 0; fk.qpos[FK_FREE + 2] = 0;
  fk.qpos[FK_FREE + 3] = 1; fk.qpos[FK_FREE + 4] = 0; fk.qpos[FK_FREE + 5] = 0; fk.qpos[FK_FREE + 6] = 0;
  for (let k = 0; k < 14; k++) fk.qpos[FK_JOINTS[k]] = joints[k];
  mj.mj_kinematics(fkModel, fk);
  return FK_SITES.map(s => ({ p: [0, 1, 2].map(r => fk.site_xpos[s * 3 + r]), m: Array.from(fk.site_xmat.slice(s * 9, s * 9 + 9)) }));
}
const point = (pose, c) => [0, 1, 2].map(r => pose.p[r] + pose.m[r * 3] * c[0] + pose.m[r * 3 + 1] * c[1] + pose.m[r * 3 + 2] * c[2]);

class Odom {
  constructor() { this.foot = 0; this.local = ALPHA16.left[5]; this.xy = [0, 0]; this.pos = [0, 0, 0]; this.yaw = 0; this.pending = null; this.ticks = 0; this.init = true; }
  reproject(q, feet) {
    const c = quatRot(q, point(feet[this.foot], this.local));
    this.pos = [this.xy[0] - c[0], this.xy[1] - c[1], -c[2]];
  }
  update(joints, q) {
    const feet = feetInTrunk(joints);
    if (this.init) { const a = quatRot(q, point(feet[this.foot], this.local)); this.xy = [a[0], a[1]]; this.init = false; }
    this.reproject(q, feet);
    let lowest = -SWITCH_MARGIN, best = null;
    for (const foot of [0, 1]) for (const c of (foot ? ALPHA16.right : ALPHA16.left)) {
      const r = quatRot(q, point(feet[foot], c));
      const z = this.pos[2] + r[2];
      if (z < lowest) { lowest = z; best = [foot, c, [this.pos[0] + r[0], this.pos[1] + r[1]]]; }
    }
    if (!best) { this.pending = null; this.ticks = 0; }
    else {
      if (this.pending && this.pending[0] === best[0]) this.ticks++; else { this.pending = best; this.ticks = 1; }
      if (this.ticks >= SWITCH_CONFIRM_TICKS) {
        [this.foot, this.local, this.xy] = this.pending;
        this.reproject(q, feet); this.pending = null; this.ticks = 0;
      }
    }
    this.yaw = yawOf(q);
  }
}

// ------------------------------------------------------------ the controller
// The spec (duck-studio docs/PACKS.md). Pure: the board pose the card starts
// from, robotd's odometry and the seconds into the card -> a twist, or done.
// Ported number for number to cardreader/duckcards/cardloop.py and StudioKit's
// DuckCardLoop.swift; the vectors in cardreader/vectors/cardloop_*.json pin it.
// duck-card-loop.json is duck-studio's StudioKit/Sources/StudioKit/Resources/DuckCards/
// duck-card-loop.json, copied: the shipped spec is what was measured.
export const SPEC = JSON.parse(fs.readFileSync(process.env.SPEC || 'duck-card-loop.json', 'utf8'));
const speedOf = (spec, card, label = 'normal') =>
  card === 'F' ? spec.walk.speed[label] : card === 'B' ? spec.walk.back_speed[label] : spec.turn.yaw_rate[label];

function walkStep(cfg, board, sign, D, rate, st, t) {
  const dx = st.x - board.x, dy = st.y - board.y;
  const along = sign * (dx * Math.cos(board.h) + dy * Math.sin(board.h));
  const across = -dx * Math.sin(board.h) + dy * Math.cos(board.h);
  if (along >= D - (sign > 0 ? cfg.stop_early_m : cfg.back_stop_early_m)) return { done: 'reached' };
  if (t >= cfg.max_s) return { done: 'out of time' };
  const e = wrap(board.h - st.yaw);
  let w = cfg.heading_gain * e - sign * cfg.cross_gain * across;
  w = Math.max(-cfg.max_yaw, Math.min(cfg.max_yaw, w));
  return { vx: sign * rate, vyaw: w };
}

// A turn: turn until within stop_early of the target; hold still settle_s; then
// done if within tolerance, else up to `trims` trims (the same, at trim_rate).
class Turn {
  constructor(cfg, board, dir, rate) { this.cfg = cfg; this.rate = rate; this.target = wrap(board.h + dir * Math.PI / 2); this.phase = 'turn'; this.trimsLeft = cfg.trims; this.since = 0; }
  step(st, t) {
    const c = this.cfg, e = wrap(this.target - st.yaw), ed = Math.abs(e) * DEG, way = e >= 0 ? 1 : -1;
    if (t >= c.max_s) return { done: 'out of time' };
    if (this.phase === 'turn' || this.phase === 'trim') {
      const trim = this.phase === 'trim';
      const lead = trim ? c.trim_stop_early_deg : c.stop_early_deg;
      if (ed <= lead) { this.phase = 'settle'; this.since = t; return { vx: 0, vyaw: 0 }; }
      return { vx: trim ? c.trim_forward : c.forward, vyaw: way * (trim ? c.trim_rate : this.rate) };
    }
    if (t - this.since < c.settle_s) return { vx: 0, vyaw: 0 };
    if (ed <= c.tolerance_deg) return { done: 'reached' };
    if (this.trimsLeft <= 0) return { done: 'trimmed out' };
    this.trimsLeft--; this.phase = 'trim';
    return { vx: c.trim_forward, vyaw: way * c.trim_rate };
  }
}

// ------------------------------------------------------------ one row
const CONTROL_DT = 0.05;       // the reader sends robot.move at 20 Hz
const LATENCY_TICKS = 2;       // and acts on a state 40 ms old
// How often robotd's state reaches the controller, in ticks: 1 is every tick
// (50 Hz, the card reader's own subscription asks for 20); 5 is Studio's 10 Hz
// `robot.subscribe` (DuckSubscription.watchingRateHz).
const STATE_EVERY = Number(process.env.STATE_EVERY || 1);

let state = null;
async function tick(vx, vyaw) { state = await ask('/intent', { vx, vyaw, hold: DT }); return state; }

async function start(wait) {
  state = await ask('/reset');
  await ask('/policy', { policy: POLICY });
  const odom = new Odom();
  for (let i = 0; i < Math.round((0.5 + wait) / DT); i++) { await tick(0, 0); odom.update(state.joints, state.quaternion); }
  return odom;
}
const truth = () => ({ x: state.position[0], y: state.position[1], yaw: yawOf(state.quaternion), up: state.upright });

async function runRow(cards, spec, D, wait) {
  const odom = await start(wait);
  const hist = [];
  let tickN = 0, held = null;
  const sample = () => {
    if (tickN++ % STATE_EVERY === 0 || !held) held = { x: odom.pos[0], y: odom.pos[1], yaw: odom.yaw };
    hist.push(held); if (hist.length > 10) hist.shift();
  };
  sample();
  const seen = () => hist[Math.max(0, hist.length - 1 - LATENCY_TICKS)];
  const T0 = truth(), O0 = { ...seen() };
  let board = { x: O0.x, y: O0.y, h: O0.yaw };     // where the duck should be, odometry's frame
  let sq = { x: 0, y: 0, h: 0 };                     // the same in squares, the row's own frame
  let fell = false, simT = 0;
  const per = [];
  const step = async (vx, vyaw) => { await tick(vx, vyaw); odom.update(state.joints, state.quaternion); sample(); simT += DT; return state.upright; };
  const runFor = async (stepper, maxS) => {
    let t = 0, cmd = [0, 0], next = 0;
    for (;;) {
      if (t >= next - 1e-9) {
        const ans = stepper(seen(), t);
        if (ans.done) return { done: ans.done, t };
        cmd = [ans.vx, ans.vyaw];
        next += CONTROL_DT;
      }
      if (!(await step(cmd[0], cmd[1]))) return { done: 'fell', t };
      t += DT;
      if (t > maxS + 5) return { done: 'runaway', t };
    }
  };
  const still = async s => { for (let i = 0; i < Math.round(s / DT); i++) if (!(await step(0, 0))) return false; return true; };
  for (const label of cards) {
    // A card is F, B, L or R, with an optional speed: 'F:fast'.
    const [card, speed = 'normal'] = label.split(':');
    const rate = speedOf(spec, card, speed);
    const a = truth(), oa = { ...seen() };
    let r;
    if (card === 'F' || card === 'B') {
      const sign = card === 'F' ? 1 : -1;
      r = await runFor((st, t) => walkStep(spec.walk, board, sign, D, rate, st, t), spec.walk.max_s);
      board = { x: board.x + sign * D * Math.cos(board.h), y: board.y + sign * D * Math.sin(board.h), h: board.h };
      sq = { x: sq.x + sign * Math.round(Math.cos(sq.h)), y: sq.y + sign * Math.round(Math.sin(sq.h)), h: sq.h };
    } else {
      const dir = card === 'L' ? 1 : -1;
      const turn = new Turn(spec.turn, board, dir, rate);
      r = await runFor((st, t) => turn.step(st, t), spec.turn.max_s);
      board = { ...board, h: wrap(board.h + dir * Math.PI / 2) };
      sq = { ...sq, h: wrap(sq.h + dir * Math.PI / 2) };
    }
    if (r.done === 'fell' || r.done === 'runaway') { fell = true; per.push({ card: label, done: r.done, s: +r.t.toFixed(2) }); break; }
    const ok = await still(spec.between_s);
    const b = truth(), ob = seen();
    per.push({ card: label, done: r.done, s: +r.t.toFixed(2),
      moved_cm: +(100 * Math.hypot(b.x - a.x, b.y - a.y)).toFixed(1),
      odom_moved_cm: +(100 * Math.hypot(ob.x - oa.x, ob.y - oa.y)).toFixed(1),
      turned_deg: +(wrap(b.yaw - a.yaw) * DEG).toFixed(1) });
    if (!ok) { fell = true; break; }
  }
  if (!fell) fell = !(await still(1.0));
  // Scored in the row's own frame (truth) against the board rule, D a square.
  const T = truth();
  const dx = T.x - T0.x, dy = T.y - T0.y;
  const rx = dx * Math.cos(T0.yaw) + dy * Math.sin(T0.yaw), ry = -dx * Math.sin(T0.yaw) + dy * Math.cos(T0.yaw);
  const ex = rx - sq.x * D, ey = ry - sq.y * D;
  const O = seen();
  const odx = O.x - O0.x, ody = O.y - O0.y;
  const orx = odx * Math.cos(O0.yaw) + ody * Math.sin(O0.yaw), ory = -odx * Math.sin(O0.yaw) + ody * Math.cos(O0.yaw);
  return {
    fell, cards: per,
    error_cm: +(100 * Math.hypot(ex, ey)).toFixed(1),
    error_x_cm: +(100 * ex).toFixed(1), error_y_cm: +(100 * ey).toFixed(1),
    heading_error_deg: +(wrap(T.yaw - T0.yaw - sq.h) * DEG).toFixed(1),
    odom_vs_truth_cm: +(100 * Math.hypot(orx - rx, ory - ry)).toFixed(1),
    odom_yaw_vs_truth_deg: +(wrap((O.yaw - O0.yaw) - (T.yaw - T0.yaw)) * DEG).toFixed(2),
    seconds: +simT.toFixed(2),
  };
}

// ------------------------------------------------------------ modes
export const ROWS = {
  course: ['F', 'F', 'L', 'F', 'R', 'F'],
  square: ['F', 'R', 'F', 'R', 'F', 'R', 'F', 'R'],
  uturn: ['F', 'L', 'L', 'F'],
  back: ['F', 'F', 'B'],
  walk4: ['F', 'F', 'F', 'F'],
  speeds: ['F:slow', 'F:fast', 'L:slow', 'F', 'R:fast', 'B:slow'],
  turns: ['L', 'L', 'R', 'R'],
};
const STARTS = Number(process.env.STARTS || 10);
const median = xs => { const s = [...xs].sort((a, b) => a - b); return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null; };
const max = xs => xs.length ? Math.max(...xs) : null;

export function summarise(runs) {
  const ok = runs.filter(r => !r.fell);
  const cards = ok.flatMap(r => r.cards);
  const walks = cards.filter(c => /^[FB]/.test(c.card)), turns = cards.filter(c => /^[LR]/.test(c.card));
  const r1 = v => v == null ? null : +v.toFixed(1);
  return {
    rows: runs.length, falls: runs.length - ok.length,
    error_cm: { median: r1(median(ok.map(r => r.error_cm))), max: r1(max(ok.map(r => r.error_cm))) },
    heading_deg: { median: r1(median(ok.map(r => Math.abs(r.heading_error_deg)))), max: r1(max(ok.map(r => Math.abs(r.heading_error_deg)))) },
    odom_vs_truth_cm: { median: r1(median(ok.map(r => r.odom_vs_truth_cm))), max: r1(max(ok.map(r => r.odom_vs_truth_cm))) },
    walk_s: { median: r1(median(walks.map(c => c.s))), max: r1(max(walks.map(c => c.s))) },
    turn_s: { median: r1(median(turns.map(c => c.s))), max: r1(max(turns.map(c => c.s))) },
    not_reached: cards.filter(c => c.done !== 'reached').length,
    row_s: { median: r1(median(ok.map(r => r.seconds))), max: r1(max(ok.map(r => r.seconds))) },
  };
}

async function measure(label, spec, D, rows) {
  const out = { label, D_cm: Math.round(D * 100), spec, rows: {}, summary: {} };
  for (const name of rows) {
    const runs = [];
    for (let s = 0; s < STARTS; s++) runs.push(await runRow(ROWS[name], spec, D, s * 0.1));
    out.rows[name] = runs;
    const S = out.summary[name] = summarise(runs);
    process.stderr.write(`${label} D=${(D * 100).toFixed(0)} ${name}: falls ${S.falls}/${S.rows} err ${S.error_cm.median}/${S.error_cm.max} cm` +
      ` hdg ${S.heading_deg.median}/${S.heading_deg.max} deg odom ${S.odom_vs_truth_cm.median}/${S.odom_vs_truth_cm.max} cm` +
      ` walk ${S.walk_s.median}/${S.walk_s.max}s turn ${S.turn_s.median}/${S.turn_s.max}s unreached ${S.not_reached}\n`);
  }
  return out;
}

const mode = process.argv[2] || 'rows';
const clone = o => JSON.parse(JSON.stringify(o));
const head = { plant: bench.plant.name, plant_digest: bench.plant.digest, policy: POLICY, starts: STARTS,
  control_hz: 1 / CONTROL_DT, latency_ticks: LATENCY_TICKS, state_hz: C.tickHz / STATE_EVERY, odometry: 'Pollen odometry crate, ALPHA16, on FK of reported joints + reported trunk quaternion',
  date: new Date().toISOString().slice(0, 10) };
if (mode === 'probe') {
  // Open loop, to set the controller: speed, drift and coast per vx; yaw per (vx, vyaw).
  const res = { ...head, walk: [], turn: [] };
  for (const vx of (process.env.VXS || '0.15,0.2,0.25,0.3,0.35,0.45,-0.2,-0.3,-0.45').split(',').map(Number)) {
    const runs = [];
    for (let s = 0; s < 5; s++) {
      const odom = await start(s * 0.1);
      const a = truth(), oa = [...odom.pos]; let up = true;
      for (let i = 0; i < 150; i++) { await tick(vx, 0); odom.update(state.joints, state.quaternion); up = up && state.upright; }
      const b = truth(), ob = [...odom.pos];
      for (let i = 0; i < 25; i++) { await tick(0, 0); up = up && state.upright; }
      const c = truth();
      runs.push({ m_per_s: +(Math.hypot(b.x - a.x, b.y - a.y) / 3).toFixed(3), odom_m_per_s: +(Math.hypot(ob[0] - oa[0], ob[1] - oa[1]) / 3).toFixed(3),
        yaw_deg: +(wrap(b.yaw - a.yaw) * DEG).toFixed(1), coast_cm: +(100 * Math.hypot(c.x - b.x, c.y - b.y)).toFixed(1), up });
    }
    res.walk.push({ vx, runs });
    process.stderr.write(`walk vx ${vx}: ${JSON.stringify(runs.map(r => [r.m_per_s, r.odom_m_per_s, r.yaw_deg, r.coast_cm, r.up]))}\n`);
  }
  for (const vx of (process.env.TVXS || '0,0.2,0.3,0.4').split(',').map(Number)) for (const w of (process.env.WS || '0.5,1.0,1.5').split(',').map(Number)) {
    const runs = [];
    for (let s = 0; s < 3; s++) {
      await start(s * 0.1);
      const a = truth(); let up = true;
      for (let i = 0; i < 100; i++) { await tick(vx, w); up = up && state.upright; }
      const b = truth();
      for (let i = 0; i < 25; i++) { await tick(0, 0); up = up && state.upright; }
      const c = truth();
      runs.push({ deg_in_2s: +(wrap(b.yaw - a.yaw) * DEG).toFixed(1), moved_cm: +(100 * Math.hypot(b.x - a.x, b.y - a.y)).toFixed(1), coast_deg: +(wrap(c.yaw - b.yaw) * DEG).toFixed(1), up });
    }
    res.turn.push({ vx, vyaw: w, runs });
    process.stderr.write(`turn vx ${vx} vyaw ${w}: ${JSON.stringify(runs.map(r => [r.deg_in_2s, r.moved_cm, r.coast_deg, r.up]))}\n`);
  }
  console.log(JSON.stringify(res));
} else if (mode === 'canon') {
  // cardsize.mjs's timed cards (2.0 s, Studio's twists) on the canon plant, rows of four.
  const SHARE = { slow: 1 / 3, normal: 2 / 3, fast: 1 };
  const TW = { F: s => [0.3 * s, 0], B: s => [-0.3 * s, 0], L: s => [0, 1.5 * s], R: s => [0, -1.5 * s] };
  const res = { ...head, runs: [] };
  for (const card of Object.keys(TW)) for (const speed of Object.keys(SHARE)) {
    const cards = [];
    let falls = 0;
    for (let s = 0; s < STARTS; s++) {
      await start(s * 0.1);
      for (let i = 0; i < 4; i++) {
        const a = truth(); let up = true;
        for (let k = 0; k < 100; k++) { await tick(...TW[card](SHARE[speed])); up = up && state.upright; }
        const b = truth();
        const dx = b.x - a.x, dy = b.y - a.y;
        cards.push({ along_cm: +(100 * (dx * Math.cos(a.yaw) + dy * Math.sin(a.yaw))).toFixed(1), turn_deg: +(wrap(b.yaw - a.yaw) * DEG).toFixed(1), up });
        if (!up) { falls++; break; }
      }
    }
    res.runs.push({ card, speed, cards, falls });
    const up = cards.filter(c => c.up);
    process.stderr.write(`${card} ${speed}: along median ${median(up.map(c => c.along_cm))} cm [${Math.min(...up.map(c => c.along_cm))}, ${Math.max(...up.map(c => c.along_cm))}]` +
      ` turn median ${median(up.map(c => c.turn_deg))} [${Math.min(...up.map(c => c.turn_deg))}, ${Math.max(...up.map(c => c.turn_deg))}] falls ${falls}/${STARTS}\n`);
  }
  console.log(JSON.stringify(res));
} else if (mode === 'sweep') {
  const result = { ...head, runs: [] };
  const variants = JSON.parse(process.env.VARIANTS || 'null') || [{ label: 'spec' }];
  const Ds = (process.env.DS || '0.20,0.25,0.30,0.40').split(',').map(Number);
  const rows = (process.env.ROWS || 'course').split(',');
  for (const v of variants) {
    const spec = clone(SPEC);
    Object.assign(spec.walk, v.walk || {}); Object.assign(spec.turn, v.turn || {});
    if (v.between_s != null) spec.between_s = v.between_s;
    // A plain number for a speed table sets all three of its speeds.
    for (const [k, o] of [['speed', spec.walk], ['back_speed', spec.walk], ['yaw_rate', spec.turn]])
      if (typeof o[k] === 'number') o[k] = { slow: o[k], normal: o[k], fast: o[k] };
    for (const D of (v.D ? [v.D] : Ds)) result.runs.push(await measure(v.label, spec, D, rows));
  }
  console.log(JSON.stringify(result));
} else {
  const D = Number(process.env.D || SPEC.square_m);
  // The spec's own square: what the board rule is scored against.
  const rows = (process.env.ROWS || Object.keys(ROWS).join(',')).split(',');
  console.log(JSON.stringify({ ...head, ...(await measure('spec', clone(SPEC), D, rows)) }));
}
