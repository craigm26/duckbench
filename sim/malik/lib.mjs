// malik/run.mjs — "Can you prompt an LLM to output the high frequency control
// commands for a legged robot in varying terrain?"  (J. Malik, 2026-09-07)
//
// Three controllers, one plant (duckbench scene.mjb: Pollen Microduck, MuJoCo,
// 0.005 s timestep, 50 Hz control = 4 substeps per tick, position servos
// kp 0.55 N·m/rad, forcerange ±0.6405 N·m), four terrains.
//
//   A  open-loop     the LLM (Claude) wrote a parametric gait by hand and it is
//                    expanded into a literal 50 Hz table of 14 joint targets
//                    (commands_open_loop.csv). No sensor is read after t=0.
//   B  feedback      the same gait plus an LLM-written balance law: projected
//                    gravity and gyro feed hip pitch/roll offsets every tick.
//   C  policy        Pollen's alpha_walking.onnx (PPO, mjlab), 61-D proprio
//                    observation, run through the bench's canonical path.
//
// A and B are tuned on FLAT floor only (small grids, reported), frozen, then
// evaluated on every terrain with 10 perturbed starts each. C is not tuned.
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { declaredDefaultPose } from '../onnx_meta.mjs';
import { makeLoop } from '../../site/duckloop.mjs';
import { findStairJoints, clearStairs, placeSteps, STAIR_Y, STEP_HALF_DEPTH } from '../../site/stairs.js';

export const NODE = process.version;
const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, LO, HI, buildObs, projectedGravity, command, findDuckJoints } = makeLoop(C);
const J = C.jointNames.filter(n => n !== 'mouth');
const ix = n => J.indexOf(n);
const L = { hy: ix('left_hip_yaw'), hr: ix('left_hip_roll'), hp: ix('left_hip_pitch'), kn: ix('left_knee'), an: ix('left_ankle') };
const R = { hy: ix('right_hip_yaw'), hr: ix('right_hip_roll'), hp: ix('right_hip_pitch'), kn: ix('right_knee'), an: ix('right_ankle') };

const mj = await load();
const MJB = 'scene.mjb';
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync(MJB)));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);
const STAIRS = findStairJoints(model);            // isolate: true (step-step collision off)
let GYRO = -1;
for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
const SUB = Math.round(1 / C.tickHz / model.opt.timestep);
const TICKS = 300, SETTLE = 25;

const stand = await ort.InferenceSession.create('./BEST_alpha_stand.onnx');
const standRef = declaredDefaultPose('./BEST_alpha_stand.onnx', HOME) ?? HOME;
const walk = await ort.InferenceSession.create('./alpha_walking.onnx');
const walkRef = declaredDefaultPose('./alpha_walking.onnx', HOME) ?? HOME;

// ───────────────────────────────────────────── terrains (block tops, metres)
// Blocks are 340 mm deep; laid 200 mm apart they overlap, so the profile is
// the max of overlapping tops: a solid sequence of ledges the duck walks along.
const blocks = tops => tops.map((top, i) => ({ x: 0.30 + 0.20 * i, top }));
const ramp = r => [0,1,2,3,4,5,6].map(i => ({ x: 0.30 + 0.20 * i, top: r * (i + 1) }));
const TERRAINS = {
  flat:      { steps: [], note: 'floor only; the stair bank parked 5 m below' },
  ledge10:   { steps: blocks([0.010,0.010,0.010,0.010,0.010,0.010,0.010]), note: 'one 10 mm step up at x = 0.13 m, then flat' },
  ledge15:   { steps: blocks([0.015,0.015,0.015,0.015,0.015,0.015,0.015]), note: 'one 15 mm step up at x = 0.13 m, then flat' },
  ledge20:   { steps: blocks([0.020,0.020,0.020,0.020,0.020,0.020,0.020]), note: 'one 20 mm step up at x = 0.13 m, then flat' },
  rough5:    { steps: blocks([0.005, 0.010, 0.005, 0.010, 0.005, 0.010, 0.005]), note: 'alternating 5 mm up/down ledges every 200 mm from x = 0.13 m' },
  rough10:   { steps: blocks([0.010, 0.020, 0.010, 0.020, 0.010, 0.020, 0.010]), note: 'alternating 10 mm up/down ledges every 200 mm from x = 0.13 m' },
  ramp5:     { steps: ramp(0.005), note: 'a flight rising 5 mm every 200 mm, to 35 mm' },
  ramp10:    { steps: ramp(0.010), note: 'a flight rising 10 mm every 200 mm, to 70 mm' },
};
function profile(steps, x) {   // ground height under x, for the report
  let h = 0;
  for (const b of steps) if (Math.abs(x - b.x) <= STEP_HALF_DEPTH) h = Math.max(h, b.top);
  return h;
}

// ───────────────────────────────────────────── the plant loop
function lcg(seed) { let s = (seed * 7919 + 17) >>> 0; return () => ((s = (1664525 * s + 1013904223) >>> 0) / 2 ** 32) * 2 - 1; }
function reset(seed, steps) {
  mj.mj_resetData(model, data);
  const rand = seed ? lcg(seed) : null;
  const f = D.freeQpos;
  data.qpos[f] = rand ? 0.002 * rand() : 0;
  data.qpos[f + 1] = STAIR_Y + (rand ? 0.002 * rand() : 0);
  data.qpos[f + 2] = 0.125; data.qpos[f + 3] = 1; data.qpos[f + 4] = data.qpos[f + 5] = data.qpos[f + 6] = 0;
  for (let k = 0; k < 14; k++) { data.qpos[D.qpos[k]] = HOME[k] + (rand ? 0.02 * rand() : 0); data.ctrl[k] = HOME[k]; }
  if (steps.length) placeSteps(data, STAIRS, steps); else clearStairs(data, STAIRS);
  mj.mj_forward(model, data);
}
const yawOf = q => Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
const BANK_HALF = 0.17;
const quat = () => [data.qpos[D.freeQpos + 3], data.qpos[D.freeQpos + 4], data.qpos[D.freeQpos + 5], data.qpos[D.freeQpos + 6]];
function sense() {
  const jp = [], jv = [];
  for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
  return { gyro: [data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]], grav: projectedGravity(quat()), jp, jv };
}
function setCtrl(target) { for (let k = 0; k < 14; k++) data.ctrl[k] = Math.min(Math.max(target[k], LO[k]), HI[k]); }
function step(steps) { if (steps.length) placeSteps(data, STAIRS, steps); for (let s = 0; s < SUB; s++) mj.mj_step(model, data); }
async function policyTick(session, ref, last, cmd) {
  const s = sense();
  const obs = buildObs(s.gyro, s.grav, s.jp, s.jv, last, cmd, ref);
  const out = await session.run({ obs: new ort.Tensor('float32', obs, [1, 61]) });
  const a = Array.from(out.actions.data);
  setCtrl(a.map((v, k) => ref[k] + v));           // the bench canon: scale 1.0, no filter
  return a;
}

// ───────────────────────────────────────────── controllers A and B (LLM-authored)
// Gait: period 1/f. Left leg swings on phase [0, π), right on [π, 2π).
//   hip pitch (left sign):  -A·cos(φ)  → foot back at φ=0, forward at φ=π
//   knee     (left sign):   +K·sin(φ) during swing, 0 in stance (lifts the foot)
//   ankle:   knee − hip deltas (keeps the sole flat), from the sign probe
//   hip roll: ±Rl·sin(φ) shifts weight over the stance foot
//   lean: a constant forward hip-pitch bias so the trunk's mass leads the feet
// Right-leg deltas are mirrored (its HOME angles are the negatives of the left).
function gaitPose(p, t, fb = null) {
  const phi = 2 * Math.PI * p.f * t;
  const N = p.neutral ?? HOME;                       // measured standing targets, or HOME
  const ramp = p.rampIn ? Math.min(1, t / p.rampIn) : 1;   // amplitudes fade in
  const leg = (ph, sgn) => {
    const swing = Math.sin(ph) > 0 ? Math.sin(ph) : 0;
    let hip = ramp * (-p.A * Math.cos(ph) - p.lean);
    let kn = ramp * p.K * swing;
    let roll = ramp * p.Rl * Math.sin(ph);
    if (fb) { hip += fb.hip; roll += fb.roll * sgn; }
    return { hip: sgn * hip, kn: sgn * kn, an: sgn * (kn - hip), roll: sgn * roll };
  };
  const l = leg(phi, 1), r = leg(phi + Math.PI, -1);
  const q = N.slice();
  q[L.hp] += l.hip; q[L.kn] += l.kn; q[L.an] += l.an; q[L.hr] += l.roll;
  q[R.hp] += r.hip; q[R.kn] += r.kn; q[R.an] += r.an; q[R.hr] += r.roll;
  return q;
}
// Balance law: projected gravity g (trunk frame, world −z). Upright g = [0,0,−1].
// g[0] > 0 means the trunk is pitched forward (gravity has a +x component in the
// body frame), g[1] > 0 means rolled toward +y (left). Gyro damps the rates.
function balance(p, s) {
  const gx = s.grav[0], gy = s.grav[1];
  return { hip: p.kp * gx - p.kd * s.gyro[1], roll: p.kr * gy - p.kd * s.gyro[0] };
}

// ───────────────────────────────────────────── one episode
async function episode({ kind, params, terrain, seed, trace = false }) {
  const steps = TERRAINS[terrain].steps;
  reset(seed, steps);
  let last = new Array(14).fill(0);
  for (let t = 0; t < SETTLE; t++) { last = await policyTick(stand, standRef, last, command({})); step(steps); }
  const f = D.freeQpos;
  const x0 = data.qpos[f];
  let fellAt = -1, maxX = x0, tr = [];
  const ctrlSum = new Array(14).fill(0); let ctrlN = 0;
  let cmd = command({ vx: params.vx ?? 0 });
  let leftBankAt = -1, yAtFall = null;
  for (let t = 0; t < TICKS; t++) {
    const time = t / C.tickHz;
    if (kind === 'policy') {
      if (params.headingHold) {   // a planner-level command: steer back to the bank's centre line
        const y = data.qpos[f + 1] - STAIR_Y, psi = yawOf(quat());
        const vyaw = Math.max(-0.5, Math.min(0.5, -params.headingHold.ky * y - params.headingHold.kpsi * psi));
        cmd = command({ vx: params.vx, vyaw });
      }
      last = await policyTick(walk, walkRef, last, cmd);
    }
    else if (kind === 'stand') last = await policyTick(stand, standRef, last, command({}));
    else if (kind === 'open_loop') setCtrl(gaitPose(params, time));
    else if (kind === 'feedback') setCtrl(gaitPose(params, time, balance(params, sense())));
    step(steps);
    if (t >= 100) { for (let k = 0; k < 14; k++) ctrlSum[k] += data.ctrl[k]; ctrlN++; }
    const g = projectedGravity(quat());
    const x = data.qpos[f], z = data.qpos[f + 2];
    maxX = Math.max(maxX, x);
    if (fellAt < 0 && (g[2] > -0.7 || z < 0.06)) { fellAt = t; yAtFall = +(data.qpos[f + 1] - STAIR_Y).toFixed(4); }
    if (leftBankAt < 0 && Math.abs(data.qpos[f + 1] - STAIR_Y) > BANK_HALF) leftBankAt = t;
    if (trace) tr.push([+x.toFixed(4), +(data.qpos[f + 1] - STAIR_Y).toFixed(4), +z.toFixed(4), +g[2].toFixed(3)]);
  }
  const g = projectedGravity(quat());
  const z = data.qpos[f + 2];
  const standing = g[2] < -0.9 && z > 0.09;
  return { kind, terrain, seed, travelled: +(data.qpos[f] - x0).toFixed(4), maxX: +maxX.toFixed(4),
           fellAt: fellAt < 0 ? null : +(fellAt / C.tickHz).toFixed(2), standing, yAtFall,
           leftBankAt: leftBankAt < 0 ? null : +(leftBankAt / C.tickHz).toFixed(2), yEnd: +(data.qpos[f + 1] - STAIR_Y).toFixed(4),
           groundAtEnd: profile(steps, data.qpos[f]), trace: trace ? tr : undefined,
           meanCtrl: ctrlSum.map(v => +(v / ctrlN).toFixed(4)) };
}


export { C, J, HOME, TERRAINS, TICKS, SETTLE, SUB, MJB, model, STAIR_Y, profile, gaitPose, episode };
