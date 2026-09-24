// The policy's forward pass, over the canonical parameter bytes.
//
// WHY THIS EXISTS AT ALL. A bench in a browser has MuJoCo — it compiles to
// WebAssembly — and it does not have onnxruntime in any form that is worth
// shipping: onnxruntime-web is a 14 MB WASM blob that wants threads and SIMD
// and a cross-origin-isolated page, which is three more things to get right on
// an iPhone than the network itself is. And the network is FOUR MATRIX
// MULTIPLIES. 61→512→256→128→14, ELU between them, nothing else: no
// convolution, no attention, no batching, no dynamic shapes. 197,774 numbers
// and about 200,000 multiply-adds a tick.
//
// IT DOES NOT PARSE ONNX, AND THAT IS THE POINT. The bytes it reads are
// `DuckPolicy.canonicalParameterBytes` — duckkit's own definition of what a
// policy IS, stripped of producer strings, initializer names and graph order:
// the normalizer mean, then the normalizer standard deviation, then for each
// layer outermost-first its weights and then its biases, every value a
// little-endian IEEE-754 binary32. That layout is already the thing DuckEvidence
// fingerprints a policy by, so the phone runs exactly the numbers the app
// attested to, and a second ONNX reader — the classic place for two
// implementations to quietly disagree — never gets written.
//
// THE ALPHA SHAPE CARRIES NO HEADER; EVERY OTHER SHAPE DOES. Until duckkit
// 1.36 `DuckPolicy.load` refused any architecture but 61→512→256→128→14, so the
// shape was a constant on both sides. It now also loads narrower students of
// the same graph (craigm26/duckbatch's 61→128→128→14, 26,254 parameters), and
// their identity bytes — `DuckPolicy.canonicalIdentityBytes`, v2 — begin with
// the shape: the ASCII `DPv2`, the layer count, then each layer's inputs and
// outputs, all little-endian uint32, then the v1 bytes. The alpha shape's bytes
// are unchanged (v1, no header), so every .bin already on disk still loads. A
// file whose length disagrees with its shape is refused, never reshaped.

/** The alpha shape, outermost first, as `DuckPolicy.expectedWidths` states it. */
export const WIDTHS = [[61, 512], [512, 256], [256, 128], [128, 14]];
export const OBS_WIDTH = 61;
export const ACTION_WIDTH = 14;
/** 61 + 61 + Σ(in·out + out) = 197,896 floats, 791,584 bytes: the ALPHA shape's size. */
export const FLOAT_COUNT = 2 * OBS_WIDTH + WIDTHS.reduce((n, [i, o]) => n + i * o + o, 0);

/** The bounds duckkit's `DuckPolicy.shapeProblem` enforces, mirrored. */
export const MAX_HIDDEN_LAYERS = 4, MAX_LAYER_WIDTH = 1024, MAX_PARAMETERS = 1_000_000;

const floatsFor = widths => 2 * OBS_WIDTH + widths.reduce((n, [i, o]) => n + i * o + o, 0);

/** duckkit's `shapeProblem`, in JS: why these widths are not a policy, or null. */
export function shapeProblem(widths) {
  if (!widths.length) return 'it has no layers';
  const hidden = widths.length - 1;
  if (hidden < 1 || hidden > MAX_HIDDEN_LAYERS) {
    return `it has ${hidden} hidden layers; between 1 and ${MAX_HIDDEN_LAYERS} are supported`;
  }
  if (widths[0][0] !== OBS_WIDTH) return `its first layer takes ${widths[0][0]} inputs, not the ${OBS_WIDTH}-float observation`;
  if (widths[hidden][1] !== ACTION_WIDTH) return `its last layer gives ${widths[hidden][1]} outputs, not the ${ACTION_WIDTH} policy joints`;
  for (let i = 0; i < hidden; i++) {
    if (widths[i][1] !== widths[i + 1][0]) return `layer ${i} gives ${widths[i][1]} outputs but layer ${i + 1} takes ${widths[i + 1][0]}`;
  }
  for (const [i, [a, b]] of widths.entries()) {
    if (a < 1 || b < 1 || a > MAX_LAYER_WIDTH || b > MAX_LAYER_WIDTH) return `layer ${i} is ${a} to ${b}; widths run from 1 to ${MAX_LAYER_WIDTH}`;
  }
  const params = widths.reduce((n, [a, b]) => n + a * b + b, 0);
  if (params > MAX_PARAMETERS) return `it has ${params} parameters; at most ${MAX_PARAMETERS} are supported`;
  return null;
}

/**
 * The shape these bytes declare, and where the floats start: the v2 header if
 * there is one, the alpha shape if there is not. Throws only on a v2 header
 * that is cut short; a shape the rule refuses comes back with its problem.
 */
export function readShape(u8) {
  const magic = u8.byteLength >= 8 && u8[0] === 0x44 && u8[1] === 0x50 && u8[2] === 0x76 && u8[3] === 0x32; // "DPv2"
  if (!magic) return { widths: WIDTHS, offset: 0, scheme: 'canonical-parameter-bytes-v1', problem: null };
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const count = view.getUint32(4, true);
  if (count > MAX_HIDDEN_LAYERS + 1 || u8.byteLength < 8 + count * 8) {
    return { widths: [], offset: 0, scheme: 'canonical-parameter-bytes-v2',
             problem: `the shape header declares ${count} layers and is cut short or implausible` };
  }
  const widths = [];
  for (let l = 0; l < count; l++) widths.push([view.getUint32(8 + 8 * l, true), view.getUint32(12 + 8 * l, true)]);
  return { widths, offset: 8 + 8 * count, scheme: 'canonical-parameter-bytes-v2', problem: shapeProblem(widths) };
}

/** Why these bytes are not a policy this bench runs, or null. The one check every caller uses. */
export function policyByteProblem(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const shape = readShape(u8);
  if (shape.problem) return shape.problem;
  const want = floatsFor(shape.widths) * 4 + shape.offset;
  if (u8.byteLength !== want) {
    return `policy parameters are ${u8.byteLength} bytes; a ${[shape.widths[0][0], ...shape.widths.map(w => w[1])].join('-')} network needs ${want}`;
  }
  return null;
}

/**
 * Read the canonical bytes into the arrays a forward pass wants.
 *
 * Little-endian is asserted rather than assumed: `DataView` would be the
 * portable-but-slow way, and every machine this runs on is little-endian, so
 * the fast path is taken and the assumption is checked once.
 */
export function loadParameters(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const problem = policyByteProblem(u8);
  if (problem) throw new Error(problem);
  const shape = readShape(u8);
  // 1.0f is 0x3F800000; on a little-endian host that 0x3f is the LAST byte.
  // Written the other way round first, this check fired on the Pi and refused
  // every policy — which is at least the failure that says so out loud.
  if (new Uint8Array(new Float32Array([1]).buffer)[0] === 0x3f) {
    // A big-endian host would read every weight byte-reversed and produce a
    // duck that falls over for reasons no log would explain.
    throw new Error('this host is big-endian: the canonical bytes are little-endian float32');
  }
  // COPIED, NOT VIEWED. A Uint8Array from `fetch` is not guaranteed to start on
  // a four-byte boundary, and Float32Array over an unaligned offset throws.
  // And copied BYTE BY BYTE into a fresh buffer, not via `u8.slice(...).buffer`:
  // in Node, `fs.readFileSync` hands back a Buffer, and `Buffer.slice` is a VIEW
  // whose `.buffer` is the whole underlying allocation from byte 0 — so reading
  // floats past a header that way starts at the header. It stayed hidden while
  // every file started with its floats; the first `DPv2` student exposed it.
  const all = new Float32Array((u8.byteLength - shape.offset) / 4);
  new Uint8Array(all.buffer).set(u8.subarray(shape.offset));
  let at = 0;
  const take = n => all.subarray(at, at += n);
  const mean = take(OBS_WIDTH), std = take(OBS_WIDTH);
  const layers = shape.widths.map(([inputs, outputs]) => ({
    inputs, outputs, weights: take(inputs * outputs), biases: take(outputs),
  }));
  for (let i = 0; i < std.length; i++) {
    if (!(Number.isFinite(std[i]) && std[i] !== 0)) {
      throw new Error(`normalizer std[${i}] is ${std[i]} — dividing by it would poison every inference`);
    }
  }
  return { mean, std, layers, widths: shape.widths, scheme: shape.scheme };
}

/**
 * (obs − mean) / std, then the MLP. ELU between the layers and NOT after the
 * last one: the final Gemm's output is the action, and squashing it would clip
 * every negative joint command to −1.
 */
export function forward(params, observation) {
  const { mean, std, layers } = params;
  let x = new Float32Array(OBS_WIDTH);
  for (let i = 0; i < OBS_WIDTH; i++) x[i] = (observation[i] - mean[i]) / std[i];
  for (let l = 0; l < layers.length; l++) {
    const { weights, biases, inputs, outputs } = layers[l];
    const out = new Float32Array(outputs);
    for (let row = 0; row < outputs; row++) {
      // The same row-major `[outputs][inputs]` order duckkit accumulates in, so
      // the two cannot drift from one another in the last bits of a float.
      const base = row * inputs;
      let acc = 0;
      for (let col = 0; col < inputs; col++) acc += weights[base + col] * x[col];
      const v = acc + biases[row];
      // ELU, α = 1: identity above zero, exp(x) − 1 below. Not on the last layer.
      out[row] = (l < layers.length - 1 && v < 0) ? Math.expm1(v) : v;
    }
    x = out;
  }
  return x;
}

/** A policy, ready to run: the shape `duckbench-core.mjs` asks `makeSession` for. */
export function makeForwardSession(bytes, name) {
  const params = loadParameters(bytes);
  return { name, run: observation => forward(params, observation) };
}

/**
 * `gain ⊙ action + offset`, ABSORBED INTO THE LAST LAYER — a transcription of
 * duckkit's `DuckPolicyWriter.folding`, which is the definition of what a gain
 * and a trim mean on a Microduck policy.
 *
 * WHY IT IS HERE AND NOT WHERE THE SEARCH IS. `/tune` scores a candidate the
 * app will later fold into a file with the Swift writer, and the only way those
 * two can be the same network is for the bench to apply the identical
 * arithmetic. The last Gemm is the last op in the graph — no ELU after it — so
 * its output IS the action, and
 *
 *     a' = gain ⊙ (W·h + b) + offset = (diag(gain)·W)·h + (gain ⊙ b + offset)
 *
 * is another Gemm of the same shape. Row `j` of `W` scaled by `gain[j]`, bias
 * `j` scaled and shifted. Fold anywhere else and an ELU sits in the way, and
 * ELU does not commute with a scale.
 *
 * EVERY ROUNDING IS float32, IN THE SAME ORDER SWIFT DOES THEM, and this is
 * the whole reason `Math.fround` appears three times in six lines. Swift's
 * `weights[i] *= Float(gain[j])` rounds the gain to binary32 FIRST and then
 * multiplies two binary32s; JavaScript would multiply the binary32 weight by a
 * full binary64 gain and round once at the store, which is a different number
 * in the last bit for a gain like 1.07 that binary32 cannot hold. The bias is
 * two operations in Swift — a multiply, then an add — so it is two roundings
 * here as well, rather than one rounding of a fused expression.
 *
 * THE FIRST THREE LAYERS ARE SHARED, NOT COPIED. A search that reallocated
 * 197,774 floats per candidate would spend most of its time in the allocator,
 * and nothing here writes to them.
 *
 * `gain` and `offset` are indexed by POLICY SLOT — fourteen wide, mouth
 * excluded, because there is no row of `W` that belongs to the mouth.
 */
export function foldParameters(params, gain, offset) {
  const last = params.layers[params.layers.length - 1];
  if (gain.length !== last.outputs || offset.length !== last.outputs) {
    throw new Error(`the gain and the trim must be ${last.outputs} wide, mouth excluded`);
  }
  for (let j = 0; j < last.outputs; j++) {
    if (!Number.isFinite(gain[j]) || !Number.isFinite(offset[j])) {
      throw new Error('the gain or the trim holds something that is not a number: a fold is '
                    + 'arithmetic on every weight in the last layer, and one NaN in it makes a '
                    + 'network that loads and drives nothing');
    }
  }
  const weights = new Float32Array(last.weights);
  const biases = new Float32Array(last.outputs);
  for (let j = 0; j < last.outputs; j++) {
    const g = Math.fround(gain[j]), o = Math.fround(offset[j]);
    const row = j * last.inputs;
    for (let i = 0; i < last.inputs; i++) weights[row + i] = Math.fround(weights[row + i] * g);
    biases[j] = Math.fround(Math.fround(last.biases[j] * g) + o);
  }
  return {
    mean: params.mean, std: params.std,
    layers: [...params.layers.slice(0, -1),
             { inputs: last.inputs, outputs: last.outputs, weights, biases }],
  };
}

/**
 * The canonical bytes back out, in duckkit's own layout: mean, std, then each
 * layer's weights and biases, outermost first, little-endian binary32.
 *
 * IT EXISTS TO BE COMPARED, NOT TO BE SHIPPED. The fold above claims to be the
 * same arithmetic as the Swift writer's, and the only way to check a claim like
 * that is to put both results side by side as bytes — which means this side has
 * to be able to produce bytes. Nothing in the bench writes a policy file.
 */
export function canonicalBytes(params) {
  const widths = params.layers.map(l => [l.inputs, l.outputs]);
  const alpha = widths.length === WIDTHS.length && widths.every(([a, b], i) => a === WIDTHS[i][0] && b === WIDTHS[i][1]);
  // duckkit's `canonicalIdentityBytes`: v1 (no header) for the alpha shape, so
  // the nine recorded official fingerprints still match; v2 for anything else.
  const header = alpha ? 0 : 8 + 8 * widths.length;
  const floats = floatsFor(widths);
  const out = new Uint8Array(header + floats * 4);
  if (!alpha) {
    const view = new DataView(out.buffer);
    out.set([0x44, 0x50, 0x76, 0x32]); // "DPv2"
    view.setUint32(4, widths.length, true);
    widths.forEach(([a, b], l) => { view.setUint32(8 + 8 * l, a, true); view.setUint32(12 + 8 * l, b, true); });
  }
  const all = new Float32Array(out.buffer, header, floats);
  let at = 0;
  const put = a => { all.set(a, at); at += a.length; };
  put(params.mean); put(params.std);
  for (const layer of params.layers) { put(layer.weights); put(layer.biases); }
  if (at !== floats) throw new Error(`wrote ${at} floats, not ${floats}`);
  return out;
}

/** A folded policy, ready to run — the same session shape as `makeForwardSession`. */
export function makeFoldedSession(bytes, gain, offset, name) {
  const params = foldParameters(loadParameters(bytes), gain, offset);
  return { name, run: observation => forward(params, observation) };
}
