import { episode, TERRAINS, TICKS } from './lib.mjs';
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
async function show(label, kind, params, terrain, seeds = [0,1,2], trace = false) {
  const rows = []; for (const seed of seeds) rows.push(await episode({ kind, params, terrain, seed, trace }));
  console.log(label.padEnd(44), 'standing', rows.filter(r => r.standing).length + '/' + rows.length,
    'travelled', mean(rows.map(r => r.travelled)).toFixed(3), 'fellAt', rows.map(r => r.fellAt).join(','),
    trace ? ' y@end ' + rows.map(r => r.trace[r.trace.length-1][1].toFixed(3)).join(',') + ' x@fall ' + rows.map(r => r.fellAt == null ? '-' : r.trace[Math.round(r.fellAt*50)][0].toFixed(3)).join(',') : '');
}
TERRAINS.flush = { steps: [0,1,2,3,4,5,6].map(i => ({ x: 0.30 + 0.20*i, top: 0.0 })), note: 'blocks flush with the floor' };
TERRAINS.mm5 = { steps: [0,1,2,3,4,5,6].map(i => ({ x: 0.30 + 0.20*i, top: 0.005 })), note: 'one 5 mm ledge then flat' };
TERRAINS.mm10 = { steps: [0,1,2,3,4,5,6].map(i => ({ x: 0.30 + 0.20*i, top: 0.010 })), note: 'one 10 mm ledge then flat' };
console.log('--- hand gait controls');
await show('A hold HOME (all amplitudes 0) flat', 'open_loop', { f: 2, A: 0, K: 0, lean: 0, Rl: 0 }, 'flat');
await show('A tiny (A .05 K .1 Rl .03) flat', 'open_loop', { f: 2, A: 0.05, K: 0.1, lean: 0, Rl: 0.03 }, 'flat');
await show('A roll only (Rl .06) flat', 'open_loop', { f: 2, A: 0, K: 0, lean: 0, Rl: 0.06 }, 'flat');
await show('A knee only (K .3) flat', 'open_loop', { f: 2, A: 0, K: 0.3, lean: 0, Rl: 0 }, 'flat');
await show('A hip only (A .15) flat', 'open_loop', { f: 2, A: 0.15, K: 0, lean: 0, Rl: 0 }, 'flat');
console.log('--- policy');
for (const vx of [0.2, 0.25, 0.3, 0.4]) await show(`C vx ${vx} flat`, 'policy', { vx }, 'flat', [0,1,2], true);
for (const t of ['flush', 'mm5', 'mm10', 'rough10']) await show(`C vx 0.3 ${t}`, 'policy', { vx: 0.3 }, t, [0,1,2], true);
