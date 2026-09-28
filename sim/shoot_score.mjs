// shoot_score.mjs — walk to a ball, line up, kick it at a goal. In physics.
//
// THE QUESTION IT ANSWERS. "What's the best way to get the duck to walk to a
// soccer ball and kick it towards a net?" Pollen ships the pieces and not the
// join: a walker that cannot kick, and two kick networks that cannot walk and
// were trained on a ball 90 mm in front of the toe (they score 0/14 on /chase,
// challenge-ball/README.md). A shot is the join, and this is one: a small
// state machine that drives Pollen's own networks, closed-loop, at the control
// rate, and switches network at the moment the ball is where the kick expects
// it.
//
//   APPROACH  walk (alpha_walking) to a point behind the ball, on the line
//             from the goal through the ball, offset for the kicking foot
//   ALIGN     turn on the spot to face the goal
//   DRIBBLE   while the ball is further than `shootRange` from the goal, walk
//             through it toward the goal: the walker pushes it along. Pollen's
//             kick sends a ball at most ~0.67 m on this plant (measured
//             2026-09-28), so from midfield a kick alone cannot score.
//   CREEP     walk in until the ball is `kickDist` ahead of the trunk
//   KICK      swap to ball_kick_left/right for `kickTicks`
//   RECOVER   the standing policy, neutral command, while the ball rolls
//
// Every number the controller decides by is a PARAMETER, and the parameters
// are what the app's "train a duck to shoot" searches over. The networks are
// Pollen's and are never changed.
//
// THE GOAL IS SCORED, NOT SIMULATED. The canon plant (scene.mjb) is a 3 x 3 m
// room with four walls and no goal, and changing it would move the plant
// digest every published challenge is pinned to. So the goal is a LINE: a
// goal is the ball's centre crossing x = GOAL.x with |y| below the half-mouth
// and z below the crossbar. The posts are drawn by the app, not collided with;
// a ball that would have hit a post counts as wide or in by where its centre
// crossed. The back wall at x = 1.5 m is real and stops everything.
//
// SENSING. `sensing: 'state'` reads the ball's position from the simulator
// (perfect perception). `sensing: 'camera'` gives the controller only what the
// head camera could: the ball's bearing and range when it is inside the
// camera's 26 degree horizontal field of view and within 2.5 m, with noise, and
// nothing otherwise, in which case the duck turns to look for it. The camera
// model is a stand-in for Pollen's duck-detect (sim/duckvision.py measured the
// field of view); no image is rendered here.

import { gravityXYSquared, rotate, yawOf } from './reward_math.mjs';

/** The pitch, in the canon plant's world frame. Shared with the app (ShootPitch). */
export const PITCH = Object.freeze({
  goal: { x: 1.30, halfWidth: 0.30, height: 0.25 },
  duck: { x: -0.70, y: 0.0, yaw: 0.0 },
  // Where the arena walls stand (scene_physics.xml): wall_e is behind the goal.
  walls: 1.5,
});

/** The kick networks were trained with the ball this far in front of the toe. */
export const KICK_TRAINED_M = 0.09;

/** The default controller: a hand-written first guess, not a tuned one. */
export const DEFAULTS = Object.freeze({
  shootRange: 0.55,    // m from the goal line: dribble until the ball is this close
  standoff: 0.22,      // m behind the ball, on the goal line, where APPROACH aims
  footSide: 0.05,      // m to the side of the ball the kicking foot lines up on
  arriveTol: 0.07,     // m: APPROACH is done inside this
  alignTol: 10,        // degrees of heading error ALIGN accepts
  kickDist: 0.10,      // m: CREEP stops when the ball is this far ahead of the trunk
  approachSpeed: 0.30, // m/s: the walker's dead band sits near 0.25
  turnGain: 1.6,       // rad/s of turn per rad of bearing
  kickTicks: 60,       // control ticks the kick network drives
  dribbleAim: 0.10,    // m past the ball, toward the goal, the dribble steers at
  minTurn: 0.5,        // rad/s: ALIGN never turns slower than this, or it creeps forever
  foot: 'auto',        // 'left', 'right' or 'auto' (the side the ball is on)
});

export const LIMITS = Object.freeze({
  shootRange: [0.25, 1.5], standoff: [0.10, 0.45], footSide: [0.0, 0.08], arriveTol: [0.03, 0.15],
  alignTol: [3, 30], kickDist: [0.08, 0.30], approachSpeed: [0.25, 0.30],
  turnGain: [0.5, 3.0], kickTicks: [20, 120], dribbleAim: [0.0, 0.40], minTurn: [0.2, 1.0],
});

export const PHASES = ['approach', 'align', 'dribble', 'creep', 'kick', 'recover'];

/**
 * The grid a shooter is measured on: nine ball spots in front of a duck that
 * always starts at the same place facing the goal. Three depths by three
 * lateral offsets, so the shot has to be aimed from off-axis as well as dead
 * ahead. `core` is the nine; the extended set adds two harder angles.
 */
export function shootCells({ core = false } = {}) {
  const cells = [];
  for (const bx of [-0.30, 0.00, 0.30]) {
    for (const by of [-0.30, 0.00, 0.30]) cells.push({ ball: { x: bx, y: by }, tier: 'core' });
  }
  if (!core) {
    cells.push({ ball: { x: 0.10, y: 0.55 }, tier: 'extended' });
    cells.push({ ball: { x: 0.10, y: -0.55 }, tier: 'extended' });
  }
  return cells;
}

/** Clamp and fill a parameter object; unknown keys are refused by name. */
export function checkParams(p = {}) {
  const out = { ...DEFAULTS };
  for (const [k, v] of Object.entries(p || {})) {
    if (!(k in DEFAULTS)) throw new Error(`shoot has no parameter "${k}"`);
    if (k === 'foot') {
      if (!['left', 'right', 'auto'].includes(v)) throw new Error('foot is left, right or auto');
      out.foot = v; continue;
    }
    const n = +v;
    if (!Number.isFinite(n)) throw new Error(`${k} must be a number`);
    const [lo, hi] = LIMITS[k];
    out[k] = Math.min(Math.max(n, lo), hi);
  }
  out.kickTicks = Math.round(out.kickTicks);
  return out;
}

const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const r4 = v => Math.round(v * 10000) / 10000;

/**
 * The rig. `ctx` is what duckbench-core hands /chase's rig, plus three more
 * actors: `walk`, `kickLeft`, `kickRight` ({ run, reference }).
 */
export function makeShootRig(ctx) {
  const { mj, model, data, D, HOME, LO, HI, buildObs, projectedGravity, command,
          tickHz, stand, walk, kickLeft, kickRight } = ctx;
  const DT = 1 / tickHz;

  const BALL = (() => {
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_type[j] !== 0) continue;
      const adr = model.jnt_qposadr[j];
      if (adr === D.freeQpos) continue;
      if (model.body(model.jnt_bodyid[j]).name === 'ball') {
        return { adr, dof: model.jnt_dofadr[j] };
      }
    }
    return null;
  })();
  if (!BALL) return null;

  let GYRO = -1;
  for (let i = 0; i < model.nsensor; i++) if (model.sensor(i).name === 'imu_ang_vel') GYRO = model.sensor(i).adr;
  if (GYRO < 0) throw new Error('this plant has no imu_ang_vel sensor');

  const TIMESTEP = (() => {
    mj.mj_resetData(model, data); mj.mj_step(model, data);
    const t = data.time; mj.mj_resetData(model, data); return t;
  })();
  const SUBSTEPS = Math.round(DT / TIMESTEP);

  /** A seeded generator, so a camera run is reproducible. */
  function rng(seed) {
    let s = (seed >>> 0) || 1;
    return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  }

  /**
   * ONE SHOT. Returns the clip (duck frames and roots at the control rate),
   * the ball's path, the phases with their start times, and the outcome.
   */
  async function runShot(cell, params, { seconds = 30, sensing = 'state', seed = 1 } = {}) {
    const P = checkParams(params);
    const f = D.freeQpos;
    const random = rng(seed);
    const gauss = () => { const u = Math.max(random(), 1e-9), v = random();
                          return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

    // ---- the world at the start: the duck at its spot, facing the goal
    mj.mj_resetData(model, data);
    data.qpos[f] = PITCH.duck.x; data.qpos[f + 1] = PITCH.duck.y; data.qpos[f + 2] = 0.120;
    data.qpos[f + 3] = Math.cos(PITCH.duck.yaw / 2); data.qpos[f + 4] = 0;
    data.qpos[f + 5] = 0; data.qpos[f + 6] = Math.sin(PITCH.duck.yaw / 2);
    for (let k = 0; k < 14; k++) { data.qpos[D.qpos[k]] = HOME[k]; data.ctrl[k] = HOME[k]; }
    const placeBall = (x, y) => {
      data.qpos[BALL.adr] = x; data.qpos[BALL.adr + 1] = y; data.qpos[BALL.adr + 2] = 0.05;
      data.qpos[BALL.adr + 3] = 1; for (let k = 4; k < 7; k++) data.qpos[BALL.adr + k] = 0;
      for (let k = 0; k < 6; k++) data.qvel[BALL.dof + k] = 0;
    };
    placeBall(cell.ball.x, cell.ball.y);
    mj.mj_forward(model, data);

    let la = new Array(14).fill(0);
    const quat = () => [data.qpos[f + 3], data.qpos[f + 4], data.qpos[f + 5], data.qpos[f + 6]];
    const step = async (actor, cmd) => {
      const jp = [], jv = [];
      for (let k = 0; k < 14; k++) { jp.push(data.qpos[D.qpos[k]]); jv.push(data.qvel[D.dof[k]]); }
      const obs = buildObs([data.sensordata[GYRO], data.sensordata[GYRO + 1], data.sensordata[GYRO + 2]],
                           projectedGravity(quat()), jp, jv, la, command(cmd), actor.reference ?? HOME);
      la = Array.from(await actor.run(obs));
      const ref = actor.reference ?? HOME;
      for (let k = 0; k < 14; k++) data.ctrl[k] = Math.min(Math.max(ref[k] + la[k], LO[k]), HI[k]);
      for (let s = 0; s < SUBSTEPS; s++) mj.mj_step(model, data);
    };

    // ---- the settle, as every clip on this bench opens (ball held in place)
    for (let t = 0; t < 25; t++) { await step(stand, {}); placeBall(cell.ball.x, cell.ball.y); }
    mj.mj_forward(model, data);

    const frames = [], roots = [], ballPath = [], phases = [];
    let phase = 'approach', phaseTick = 0, kickActor = null, kickFoot = null, settled = 0;
    let lastSeen = null, outcome = null, outcomeTick = null, atKick = null;
    const setPhase = (p, t) => { phase = p; phaseTick = t; phases.push({ phase: p, t: r4(t * DT) }); };
    phases.push({ phase: 'approach', t: 0 });

    const ticks = Math.round(seconds * tickHz);
    for (let t = 0; t < ticks; t++) {
      const x = data.qpos[f], y = data.qpos[f + 1], yaw = yawOf(quat());
      const bx = data.qpos[BALL.adr], by = data.qpos[BALL.adr + 1];

      // ---- what the controller knows about the ball this tick
      let seen = { x: bx, y: by };
      if (sensing === 'camera') {
        const dx = bx - x, dy = by - y, range = Math.hypot(dx, dy);
        const bearing = wrap(Math.atan2(dy, dx) - yaw);
        if (Math.abs(bearing) <= (13 * Math.PI) / 180 && range <= 2.5) {
          const nb = bearing + gauss() * (1.5 * Math.PI / 180);
          const nr = range * (1 + gauss() * 0.05);
          seen = { x: x + nr * Math.cos(yaw + nb), y: y + nr * Math.sin(yaw + nb) };
          lastSeen = { ...seen, t };
        } else {
          seen = lastSeen && t - lastSeen.t < tickHz * 1.5 ? lastSeen : null;
        }
      }

      // ---- the plan, from what is known
      const gx = PITCH.goal.x, gy = 0;
      let cmd = {}, actor = walk;
      if (phase === 'kick') {
        actor = kickActor;
        if (t - phaseTick >= P.kickTicks) setPhase('recover', t);
      } else if (phase === 'recover') {
        actor = stand;
      } else if (!seen) {
        // LOOKING FOR IT: turn toward where it was last, or left.
        cmd = { vx: 0, vy: 0, vyaw: 0.9 };
      } else {
        const ux0 = gx - seen.x, uy0 = gy - seen.y, un = Math.hypot(ux0, uy0) || 1;
        const ux = ux0 / un, uy = uy0 / un;               // ball → goal
        const nx = -uy, ny = ux;                          // left of that line
        let foot = P.foot;
        if (foot === 'auto') {
          // The side of the line the duck is already on decides the foot: a
          // duck left of the line lines up left of it, which puts the ball on
          // its right, which is the right foot's.
          const side = (x - seen.x) * nx + (y - seen.y) * ny;
          foot = side >= 0 ? 'right' : 'left';
        }
        // A LEFT-FOOT KICK WANTS THE BALL ON THE DUCK'S LEFT, so the trunk
        // lines up to the RIGHT of the ball-to-goal line (negative `n`).
        const lateral = foot === 'left' ? -P.footSide : P.footSide;
        const tx = seen.x - ux * P.standoff + nx * lateral;
        const ty = seen.y - uy * P.standoff + ny * lateral;
        const heading = Math.atan2(uy, ux);

        if (phase === 'approach') {
          const dx = tx - x, dy = ty - y, d = Math.hypot(dx, dy);
          const b = wrap(Math.atan2(dy, dx) - yaw);
          if (d < P.arriveTol) setPhase('align', t);
          else {
            const vyaw = Math.max(-1.2, Math.min(1.2, P.turnGain * b));
            cmd = { vx: Math.abs(b) > 1.0 ? 0 : P.approachSpeed * Math.max(0.6, Math.cos(b)), vy: 0, vyaw };
          }
        }
        if (phase === 'align') {
          const e = wrap(heading - yaw);
          if (Math.abs(e) < (P.alignTol * Math.PI) / 180) { settled++; } else settled = 0;
          const far = PITCH.goal.x - seen.x > P.shootRange;
          if (settled >= 5) { settled = 0; setPhase(far ? 'dribble' : 'creep', t); }
          else {
            const w = Math.max(-1.0, Math.min(1.0, P.turnGain * e));
            cmd = { vx: 0, vy: 0, vyaw: Math.abs(w) < P.minTurn ? Math.sign(e) * P.minTurn : w };
          }
          // Drifted away while turning: go back.
          if (Math.hypot(tx - x, ty - y) > P.arriveTol * 2.5) setPhase('approach', t);
        }
        if (phase === 'dribble') {
          // Walk THROUGH the ball, aiming at a point just past it on the line
          // to the goal; the push is the walker's own stride.
          const lx = Math.cos(yaw) * (seen.x - x) + Math.sin(yaw) * (seen.y - y);
          const ly = -Math.sin(yaw) * (seen.x - x) + Math.cos(yaw) * (seen.y - y);
          if (PITCH.goal.x - seen.x <= P.shootRange) {
            setPhase('approach', t);                    // close enough: set up the kick
          } else if (Math.abs(ly) > 0.09 || lx < -0.02 || lx > 0.45) {
            setPhase('approach', t);                    // lost it: get behind it again
          } else {
            const aimX = seen.x + ux * P.dribbleAim, aimY = seen.y + uy * P.dribbleAim;
            const b = wrap(Math.atan2(aimY - y, aimX - x) - yaw);
            cmd = { vx: P.approachSpeed, vy: 0, vyaw: Math.max(-0.8, Math.min(0.8, P.turnGain * b)) };
          }
        }
        if (phase === 'creep') {
          // The ball in the duck's own frame.
          const lx = Math.cos(yaw) * (seen.x - x) + Math.sin(yaw) * (seen.y - y);
          const ly = -Math.sin(yaw) * (seen.x - x) + Math.cos(yaw) * (seen.y - y);
          const want = foot === 'left' ? P.footSide : -P.footSide;
          if (lx <= P.kickDist) {
            kickFoot = foot;
            // WHERE THE BALL TRULY WAS, in the duck's frame, as the kick began.
            const tx_ = Math.cos(yaw) * (bx - x) + Math.sin(yaw) * (by - y);
            const ty_ = -Math.sin(yaw) * (bx - x) + Math.cos(yaw) * (by - y);
            atKick = { ahead_m: r4(tx_), left_m: r4(ty_), heading_err_deg: r4(wrap(heading - yaw) * 180 / Math.PI) };
            kickActor = foot === 'left' ? kickLeft : kickRight;
            setPhase('kick', t);
            actor = kickActor;
          } else if (Math.abs(ly - want) > 0.08 || lx > P.standoff + 0.15) {
            setPhase('approach', t);
          } else {
            const e = wrap(heading - yaw);
            cmd = { vx: P.approachSpeed, vy: 0, vyaw: Math.max(-0.8, Math.min(0.8, P.turnGain * e)) };
          }
        }
      }

      await step(actor, phase === 'kick' || phase === 'recover' ? {} : cmd);

      // ---- record, and judge the ball
      const after = [];
      for (let k = 0; k < 14; k++) after.push(r4(Math.min(Math.max(data.qpos[D.qpos[k]], LO[k]), HI[k])));
      frames.push(after);
      roots.push([data.qpos[f], data.qpos[f + 1], data.qpos[f + 2], data.qpos[f + 3],
                  data.qpos[f + 4], data.qpos[f + 5], data.qpos[f + 6]].map(r4));
      const nbx = data.qpos[BALL.adr], nby = data.qpos[BALL.adr + 1], nbz = data.qpos[BALL.adr + 2];
      ballPath.push([r4(nbx), r4(nby), r4(nbz)]);

      if (!outcome && nbx >= PITCH.goal.x) {
        outcome = Math.abs(nby) <= PITCH.goal.halfWidth && nbz <= PITCH.goal.height ? 'goal' : 'wide';
        outcomeTick = t;
      }
      if (!outcome && projectedGravity(quat())[2] > -0.5) { outcome = 'fell'; outcomeTick = t; }
      // A kicked ball that has stopped short ends the shot.
      if (!outcome && phase === 'recover') {
        const sp = Math.hypot(data.qvel[BALL.dof], data.qvel[BALL.dof + 1]);
        if (t - phaseTick > tickHz * 0.5 && sp < 0.02) { outcome = 'short'; outcomeTick = t; }
      }
      // A little after the verdict, so the picture shows the ball arrive.
      if (outcome && t - outcomeTick >= tickHz * 0.6) break;
    }
    if (!outcome) outcome = phase === 'recover' || phase === 'kick' ? 'short' : 'never kicked';

    const end = ballPath[ballPath.length - 1] || [cell.ball.x, cell.ball.y, 0.05];
    // HOW FAR FROM IN: 0 for a goal; otherwise the distance from where the
    // ball ended (or crossed) to the nearest point of the mouth. The search
    // reads this to tell a near miss from a duck that never kicked.
    let miss = 0;
    if (outcome !== 'goal') {
      const cy = Math.max(-PITCH.goal.halfWidth, Math.min(PITCH.goal.halfWidth, end[1]));
      miss = Math.hypot(PITCH.goal.x - Math.min(end[0], PITCH.goal.x), end[1] - cy);
    }
    return {
      cell, params: P, sensing, seconds: r4(frames.length * DT),
      outcome, goal: outcome === 'goal', miss_m: r4(miss), foot: kickFoot, atKick,
      kicked: phases.some(p => p.phase === 'kick'),
      phases, hz: tickHz, frames, roots, ball: ballPath,
    };
  }

  return { runShot, SUBSTEPS };
}
