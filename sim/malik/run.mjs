// malik/run.mjs — see lib.mjs for the plant, the controllers and the episode.
import fs from 'node:fs';
import { C, J, TERRAINS, TICKS, SETTLE, SUB, MJB, model, STAIR_Y, profile, gaitPose, episode, NODE } from './lib.mjs';
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const N = JSON.parse(fs.readFileSync('malik/neutral.json', 'utf8')).neutral;
async function evaluate(kind, params, terrain, seeds, trace = false) {
  const rows = [];
  for (const seed of seeds) rows.push(await episode({ kind, params, terrain, seed, trace }));
  return rows;
}
// Tuning score: an episode that stands to the end is worth 100 + metres; one
// that falls is worth the seconds it survived plus the metres it made.
const score = rows => mean(rows.map(r => r.standing ? 100 + r.travelled : r.fellAt + r.travelled));

// ───────────────────────────────────────────── tune A and B on flat (grids, all reported)
const TUNE_SEEDS = [0, 1, 2];
const EVAL_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const tuning = { open_loop: [], feedback: [] };
let bestA = null;
for (const f of [1.5, 2.0, 3.0]) for (const A of [0.05, 0.10, 0.15]) for (const K of [0.15, 0.30]) for (const Rl of [-0.08, -0.04, 0.04, 0.08]) {
  const p = { f, A, K, Rl, lean: 0, rampIn: 1, neutral: N };
  const rows = await evaluate('open_loop', p, 'flat', TUNE_SEEDS);
  const s = score(rows);
  tuning.open_loop.push({ params: { f, A, K, Rl }, score: +s.toFixed(3), standing: rows.filter(r => r.standing).length, travelled: +mean(rows.map(r => r.travelled)).toFixed(4), fellAt: rows.map(r => r.fellAt) });
  if (!bestA || s > bestA.score) bestA = { params: p, score: s };
}
process.stderr.write(`best A ${JSON.stringify({ ...bestA.params, neutral: undefined })} score ${bestA.score.toFixed(3)}\n`);
let bestB = null;
for (const kp of [-0.6, -0.4, -0.2]) for (const kr of [-0.3, 0, 0.3]) for (const kd of [0, 0.02, 0.05]) {
  const p = { ...bestA.params, kp, kr, kd };
  const rows = await evaluate('feedback', p, 'flat', TUNE_SEEDS);
  const s = score(rows);
  tuning.feedback.push({ params: { kp, kr, kd }, score: +s.toFixed(3), standing: rows.filter(r => r.standing).length, travelled: +mean(rows.map(r => r.travelled)).toFixed(4), fellAt: rows.map(r => r.fellAt) });
  if (!bestB || s > bestB.score) bestB = { params: p, score: s };
}
process.stderr.write(`best B ${JSON.stringify({ ...bestB.params, neutral: undefined })} score ${bestB.score.toFixed(3)}\n`);

// ───────────────────────────────────────────── the literal command table (A)
const csv = ['t_s,' + J.join(',')];
for (let t = 0; t < TICKS; t++) csv.push((t / C.tickHz).toFixed(2) + ',' + gaitPose(bestA.params, t / C.tickHz).map(v => v.toFixed(4)).join(','));
fs.writeFileSync('malik/commands_open_loop.csv', csv.join('\n') + '\n');

// ───────────────────────────────────────────── evaluation
const HH = { ky: 2.0, kpsi: 1.0 };
const controllers = [
  { id: 'A_open_loop', kind: 'open_loop', params: bestA.params, label: 'A. LLM open-loop 50 Hz table' },
  { id: 'A_hold', kind: 'open_loop', params: { f: 2, A: 0, K: 0, Rl: 0, lean: 0, neutral: N }, label: 'A0. hold the measured standing targets (control)' },
  { id: 'B_feedback', kind: 'feedback', params: bestB.params, label: 'B. LLM gait + LLM balance law' },
  { id: 'B_hold', kind: 'feedback', params: { f: 2, A: 0, K: 0, Rl: 0, lean: 0, neutral: N, kp: bestB.params.kp, kr: bestB.params.kr, kd: bestB.params.kd }, label: 'B0. balance law only, no gait (control)' },
  { id: 'C_policy', kind: 'policy', params: { vx: 0.3 }, label: 'C. PPO policy, vx 0.3 m/s' },
  { id: 'C_policy_hh', kind: 'policy', params: { vx: 0.3, headingHold: HH }, label: 'C+. PPO policy, vx 0.3 m/s, heading hold on the bank centre line' },
  { id: 'C_policy_vx0.2', kind: 'policy', params: { vx: 0.2 }, label: 'C at vx 0.2 m/s (below its dead zone)' },
];
const results = [];
for (const c of controllers) for (const terrain of Object.keys(TERRAINS)) {
  const rows = await evaluate(c.kind, c.params, terrain, EVAL_SEEDS, true);
  const standing = rows.filter(r => r.standing).length;
  process.stderr.write(`${c.id.padEnd(16)} ${terrain.padEnd(8)} standing ${standing}/10  travelled ${mean(rows.map(r => r.travelled)).toFixed(3)} m  leftBank ${rows.filter(r => r.leftBankAt != null).length}\n`);
  results.push({ controller: c.id, label: c.label, kind: c.kind, params: { ...c.params, neutral: undefined }, terrain, standing,
                 leftBank: rows.filter(r => r.leftBankAt != null).length,
                 travelledMean: +mean(rows.map(r => r.travelled)).toFixed(4),
                 travelledMedian: +rows.map(r => r.travelled).sort((a, b) => a - b)[5].toFixed(4), episodes: rows });
}
const out = {
  question: 'Can you prompt an LLM to output the high frequency control commands for a legged robot in varying terrain?',
  date: new Date().toISOString(), node: NODE,
  plant: { file: MJB, timestep: model.opt.timestep, substeps: SUB, tickHz: C.tickHz, nq: model.nq, nu: model.nu, servo: 'position servos, kp 0.55 N·m/rad, forcerange ±0.6405 N·m (measured 2026-09-06)' },
  protocol: { settleTicks: SETTLE, ticks: TICKS, seconds: TICKS / C.tickHz, spawnY: STAIR_Y, bankHalfWidth: 0.17,
              perturbation: '±2 mm trunk, ±0.02 rad per joint, seed 0 nominal', tuneSeeds: TUNE_SEEDS, evalSeeds: EVAL_SEEDS,
              fell: 'projected-gravity z > −0.7 or trunk z < 60 mm at any tick', standing: 'projected-gravity z < −0.9 and trunk z > 90 mm at the last tick',
              stepIsolation: 'step-step collision off (site/stairs.js isolateSteps); step-duck contact unchanged',
              headingHold: HH },
  neutral: N,
  terrains: Object.fromEntries(Object.entries(TERRAINS).map(([k, v]) => [k, { steps: v.steps, note: v.note, profile: Array.from({ length: 41 }, (_, i) => [+(i * 0.05).toFixed(2), profile(v.steps, i * 0.05)]) }])),
  tuning, best: { open_loop: { ...bestA, params: { ...bestA.params, neutral: undefined } }, feedback: { ...bestB, params: { ...bestB.params, neutral: undefined } } }, results,
};
fs.writeFileSync('malik/results.json', JSON.stringify(out));
console.log('wrote malik/results.json and malik/commands_open_loop.csv');
