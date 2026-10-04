import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { setImmediate } from "node:timers/promises";
import { test } from "node:test";
import vm from "node:vm";

const page = readFileSync(new URL("../src/articraft/viewer.html", import.meta.url), "utf8");
// Run the page's motion functions with only rendering and HTTP replaced.
const motion = page.slice(page.indexOf("    async function poseOnServer()"), page.indexOf("    function fit()"));

function viewer(solver = "server", respond = () => ({ bodies: { body: [1, 2, 3] }, dofs: { "follower.rotZ": 0.2 } })) {
  const requests = [];
  const joint = name => ({ spec: { name, type: "revolute", motion_limits: { lower: -0.5, upper: 0.5 } }, value: 0, input: {}, current: {} });
  const context = vm.createContext({
    state: {
      version: { id: "0000", model: { solver, can_pose: true } },
      joints: new Map(["driver", "follower"].map(name => [name, joint(name)])),
      parts: new Map([["body", { node: { matrix: [0, 0, 0] } }]]),
    },
    preview: { minimumCycle: 2.6, angularSpeed: Math.PI / 4 },
    THREE: {
      MathUtils: { euclideanModulo: (n, m) => ((n % m) + m) % m, lerp: (a, b, t) => a + (b - a) * t },
      Matrix4: class { set(...values) { this.values = values; return this; } },
    },
    applyMatrix: (node, matrix) => { node.matrix = matrix.values; },
    format: (_spec, value) => String(value),
    treePoses: 0,
    renderTree() {},
    document: { querySelector: () => ({ setAttribute() {} }) },
    performance,
    async fetch(url, request) {
      const values = JSON.parse(request.body);
      requests.push({ url, values });
      const result = await respond(values);
      return { ok: true, json: async () => result };
    },
  });
  vm.runInContext(`function poseTree(){ treePoses++; }\n${motion}`, context);
  return { context, requests, run: code => vm.runInContext(code, context) };
}

test("closed-loop preview drives one joint and applies solved body and follower poses", async () => {
  const { context, requests, run } = viewer();
  run("animateMotion(1)");
  await setImmediate();
  assert.equal(requests.length, 1);
  assert.deepEqual(Object.keys(requests[0].values), ["driver"]);
  assert.notEqual(requests[0].values.driver, 0);
  assert.deepEqual(Array.from(context.state.parts.get("body").node.matrix), [1, 2, 3]);
  assert.equal(context.state.joints.get("follower").value, 0.2);
  assert.equal(context.state.joints.get("follower").input.value, 0.2);
  assert.equal(context.treePoses, 0);
});

test("tree preview still moves every joint without an HTTP request", () => {
  const { context, requests, run } = viewer("tree");
  run("animateMotion(1)");
  assert.equal(context.treePoses, 1);
  assert.equal(requests.length, 0);
  for (const joint of context.state.joints.values()) assert.notEqual(joint.value, 0);
});

test("a slider sends only its chosen driver", async () => {
  const { requests, run } = viewer();
  run('move("follower", 0.1)');
  await setImmediate();
  assert.deepEqual(requests[0].values, { follower: 0.1 });
});

test("animation queues the latest pose while one solve is in flight", async () => {
  let finish;
  let calls = 0;
  const { context, requests, run } = viewer("server", () => {
    if (++calls === 1) return new Promise(resolve => { finish = resolve; });
    return { bodies: {}, dofs: {} };
  });
  run("animateMotion(0.5); animateMotion(1)");
  assert.equal(requests.length, 1);
  const latest = context.state.joints.get("driver").value;
  finish({ bodies: {}, dofs: {} });
  await setImmediate();
  assert.equal(requests.length, 2);
  assert.equal(requests[1].values.driver, latest);
  assert.equal(context.state.solving, false);
});

test("reset clears the driver and requests the rest pose", async () => {
  const { context, requests, run } = viewer("server", values => ({
    bodies: { body: Object.keys(values).length ? [1] : [0] }, dofs: { "follower.rotZ": 0 },
  }));
  run("animateMotion(1)");
  await setImmediate();
  run("reset()");
  await setImmediate();
  assert.deepEqual(requests[1].values, {});
  assert.deepEqual(Array.from(context.state.parts.get("body").node.matrix), [0]);
  assert.equal(context.state.joints.get("driver").value, 0);
});

test("a solve from an old version cannot move a newly loaded model", async () => {
  let finish;
  const { context, run } = viewer("server", () => new Promise(resolve => { finish = resolve; }));
  run("animateMotion(1)");
  context.state.version = { id: "0001", model: { solver: "tree", can_pose: true } };
  finish({ bodies: { body: [99] }, dofs: {} });
  await setImmediate();
  assert.deepEqual(context.state.parts.get("body").node.matrix, [0, 0, 0]);
  assert.equal(context.state.solving, false);
});
