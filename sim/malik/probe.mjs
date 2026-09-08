// Kinematic sign probe: which way does each left-leg joint move the left foot?
import load from 'mujoco';
import fs from 'node:fs';
import { makeLoop } from '../../site/duckloop.mjs';
const C = JSON.parse(fs.readFileSync('duckkit-constants.json', 'utf8'));
const { HOME, findDuckJoints } = makeLoop(C);
const mj = await load();
mj.FS.writeFile('/s.mjb', new Uint8Array(fs.readFileSync('scene.mjb')));
const model = mj.MjModel.mj_loadBinary('/s.mjb', new mj.MjVFS());
const data = new mj.MjData(model);
const D = findDuckJoints(model);
const bodies = []; for (let b = 0; b < model.nbody; b++) bodies.push(model.body(b).name);
console.log('bodies:', bodies.join(', '));
const sites = []; for (let s = 0; s < model.nsite; s++) sites.push(model.site(s).name);
console.log('sites:', sites.join(', '));
const geoms = []; for (let g = 0; g < model.ngeom; g++) { const n = model.geom(g).name; if (/foot|toe|sole/i.test(n)) geoms.push(n + '@' + model.body(model.geom_bodyid[g]).name); }
console.log('foot-ish geoms:', geoms.join(', '));
const names = C.jointNames.filter(n => n !== 'mouth');
function footPos(siteName) {
  const b = sites.indexOf(siteName);
  return [data.site_xpos[3*b], data.site_xpos[3*b+1], data.site_xpos[3*b+2]];
}
function setPose(delta) {
  mj.mj_resetData(model, data);
  data.qpos[D.freeQpos+2] = 0.3; data.qpos[D.freeQpos+3] = 1;
  for (let k = 0; k < 14; k++) data.qpos[D.qpos[k]] = HOME[k] + (delta[k] || 0);
  mj.mj_forward(model, data);
}
const footBody = 'left_foot';
console.log('using foot body', footBody);
setPose([]); const base = footPos(footBody); console.log('home foot', base.map(v => v.toFixed(4)));
for (const j of ['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle']) {
  const k = names.indexOf(j); const d = new Array(14).fill(0); d[k] = 0.3;
  setPose(d); const p = footPos(footBody);
  console.log(j.padEnd(15), '+0.3 rad -> dx', ((p[0]-base[0])*1000).toFixed(1), 'dy', ((p[1]-base[1])*1000).toFixed(1), 'dz', ((p[2]-base[2])*1000).toFixed(1), 'mm');
}
