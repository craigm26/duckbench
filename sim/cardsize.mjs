// cardsize: how far one Duck Card carries the duck, in this bench's MuJoCo
// model of Pollen's Microduck with alpha_walking.
//
// A card in Studio is DuckIntentPlan.defaultSeconds (2.0 s) of one twist:
// vx = 0.3 m/s x share for a walk, vyaw = 1.5 rad/s x share for a turn, with
// share 1/3, 2/3, 1 for Slow, Normal, Fast (SequenceProposal.twist). A row
// runs its cards back to back, so this lays rows of four identical cards and
// measures each card's own displacement in the duck's frame at that card's
// start: along (forward +), across (left +), and the heading change. The model
// is deterministic, so the spread comes from ten starts that differ by a short
// stand of 0 to 0.9 s before the row, which puts the gait at different phases.
//
// Usage: node cardsize.mjs > cardsize.json   (one process, about a minute)
import load from 'mujoco';
import * as ort from 'onnxruntime-node';
import fs from 'node:fs';
import { makeLoop } from '../site/duckloop.mjs';

const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, buildObs, gaitTargets, projectedGravity, command } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/scene.xml', fs.readFileSync('scene.xml', 'utf8'));
const model = mj.MjModel.mj_loadXML('/scene.xml');
const data = new mj.MjData(model);
const session = await ort.InferenceSession.create('./alpha_walking.onnx');
const inputName = session.inputNames[0];
const yawOf = q => Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
const wrap = a => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };

const SECONDS = 2.0, SETTLE = 1.0, CARDS = 4, STARTS = 10;
const SHARE = { slow: 1 / 3, normal: 2 / 3, fast: 1 };
const TWIST = {
  walk_forward: s => ({ vx: 0.3 * s }),
  walk_backward: s => ({ vx: -0.3 * s }),
  turn_left: s => ({ vyaw: 1.5 * s }),
  turn_right: s => ({ vyaw: -1.5 * s }),
};

let la, prev;
function pose() {
  const q = [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]];
  return { x: data.qpos[0], y: data.qpos[1], z: data.qpos[2], yaw: yawOf(q) };
}
async function tick(cmd) {
  const q = [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]];
  const jp = [], jv = [];
  for (let k = 0; k < 14; k++) { jp.push(data.qpos[7 + k]); jv.push(data.qvel[6 + k]); }
  const gyro = [data.sensordata[0], data.sensordata[1], data.sensordata[2]];
  const obs = buildObs(gyro, projectedGravity(q), jp, jv, la, cmd);
  const out = await session.run({ [inputName]: new ort.Tensor('float32', obs, [1, 61]) });
  la = Array.from(out[session.outputNames[0]].data);
  prev = gaitTargets(la, prev);
  for (let k = 0; k < 14; k++) data.ctrl[k] = prev[k];
  for (let s = 0; s < 4; s++) mj.mj_step(model, data);
}
async function hold(opts, secs) {
  const cmd = command(opts);
  for (let t = 0; t < Math.round(secs * C.tickHz); t++) await tick(cmd);
}

async function row(card, speed, wait) {
  mj.mj_resetData(model, data);
  data.qpos[2] = 0.1231; data.qpos[3] = 1;
  for (let i = 0; i < 14; i++) { data.qpos[7 + i] = HOME[i]; data.ctrl[i] = HOME[i]; }
  mj.mj_forward(model, data);
  la = new Array(14).fill(0); prev = null;
  await hold({}, SETTLE + wait);
  const out = [];
  for (let i = 0; i < CARDS; i++) {
    const a = pose();
    await hold(TWIST[card](SHARE[speed]), SECONDS);
    const b = pose();
    const dx = b.x - a.x, dy = b.y - a.y;
    out.push({
      along: dx * Math.cos(a.yaw) + dy * Math.sin(a.yaw),
      across: -dx * Math.sin(a.yaw) + dy * Math.cos(a.yaw),
      turn_deg: wrap(b.yaw - a.yaw) * 180 / Math.PI,
      upright: b.z > 0.07,
    });
  }
  // The duck stops after the row: how far does it coast?
  const a = pose();
  await hold({}, 1.0);
  const b = pose();
  return { cards: out, coast: Math.hypot(b.x - a.x, b.y - a.y) };
}

const result = { model: 'scene.xml + alpha_walking.onnx', seconds: SECONDS, cards_per_row: CARDS, starts: STARTS, runs: [] };
for (const card of Object.keys(TWIST)) {
  for (const speed of Object.keys(SHARE)) {
    for (let s = 0; s < STARTS; s++) {
      const r = await row(card, speed, s * 0.1);
      result.runs.push({ card, speed, wait: +(s * 0.1).toFixed(1), ...r });
    }
    process.stderr.write(`${card} ${speed} done\n`);
  }
}
console.log(JSON.stringify(result));
