import fs from 'node:fs';
import { episode, TERRAINS, HOME, J } from './lib.mjs';
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
async function show(label, kind, params, terrain, seeds = [0,1,2], trace = false) {
  const rows = []; for (const seed of seeds) rows.push(await episode({ kind, params, terrain, seed, trace }));
  console.log(label.padEnd(44), 'standing', rows.filter(r => r.standing).length + '/' + rows.length,
    'travelled', mean(rows.map(r => r.travelled)).toFixed(3), 'fellAt', rows.map(r => r.fellAt).join(','),
    trace ? ' x@fall ' + rows.map(r => r.fellAt == null ? '-' : r.trace[Math.round(r.fellAt*50)][0].toFixed(3)).join(',') : '');
  return rows;
}
const ledge = h => [0,1,2,3,4,5,6].map(i => ({ x: 0.30 + 0.20*i, top: h }));
Object.assign(TERRAINS, {
  ledge15: { steps: ledge(0.015) }, ledge20: { steps: ledge(0.020) }, ledge25: { steps: ledge(0.025) },
  rough5: { steps: [0.005,0.010,0.005,0.010,0.005,0.010,0.005].map((top,i)=>({x:0.30+0.20*i, top})) },
  ramp5: { steps: [0,1,2,3,4,5,6].map(i => ({ x: 0.30 + 0.20*i, top: 0.005*(i+1) })) },
});
console.log('--- the standing policy, steady state, flat');
const st = await show('stand policy flat', 'stand', {}, 'flat', [0,1,2]);
const N = st[0].meanCtrl.map((_, k) => +mean(st.map(r => r.meanCtrl[k])).toFixed(4));
console.log('measured neutral targets vs HOME:'); J.forEach((n, k) => console.log('  ' + n.padEnd(16), N[k].toFixed(4), ' HOME', HOME[k].toFixed(4), ' delta', (N[k]-HOME[k]).toFixed(4)));
fs.writeFileSync('malik/neutral.json', JSON.stringify({ neutral: N, home: HOME, joints: J, how: 'mean data.ctrl over ticks 100-299 of BEST_alpha_stand.onnx on flat, seeds 0-2' }));
console.log('--- hold the measured neutral, open loop');
await show('A hold neutral (amplitudes 0)', 'open_loop', { f: 2, A: 0, K: 0, lean: 0, Rl: 0, neutral: N }, 'flat', [0,1,2], true);
await show('B hold neutral + balance kp .3 kr .3 kd .02', 'feedback', { f: 2, A: 0, K: 0, lean: 0, Rl: 0, neutral: N, kp: 0.3, kr: 0.3, kd: 0.02 }, 'flat');
await show('B hold neutral + balance kp -.3 kr -.3 kd .02', 'feedback', { f: 2, A: 0, K: 0, lean: 0, Rl: 0, neutral: N, kp: -0.3, kr: -0.3, kd: 0.02 }, 'flat');
await show('B hold neutral + balance kp .3 kr -.3', 'feedback', { f: 2, A: 0, K: 0, lean: 0, Rl: 0, neutral: N, kp: 0.3, kr: -0.3, kd: 0.02 }, 'flat');
await show('B hold neutral + balance kp -.3 kr .3', 'feedback', { f: 2, A: 0, K: 0, lean: 0, Rl: 0, neutral: N, kp: -0.3, kr: 0.3, kd: 0.02 }, 'flat');
await show('A gait on neutral, ramp 1 s (A .15 K .3)', 'open_loop', { f: 2, A: 0.15, K: 0.3, lean: 0, Rl: 0.06, neutral: N, rampIn: 1 }, 'flat', [0,1,2], true);
console.log('--- policy ledge limit, vx 0.3');
for (const t of ['ledge15', 'ledge20', 'ledge25', 'rough5', 'ramp5']) await show(`C vx 0.3 ${t}`, 'policy', { vx: 0.3 }, t, [0,1,2], true);
