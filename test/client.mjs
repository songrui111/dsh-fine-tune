/**
 * Client-half test: the browser bundle loads under a fake module loader with a
 * stub React, registers exactly two seats, and its pure helpers behave.
 *
 * This is the only test that catches the failure mode "the host works but the
 * panel is a blank rectangle": the bundle is a plain script, so `node --check`
 * cannot validate it, and a typo in the factory shows up only at load time.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, '..', 'client', 'client.js'), 'utf8');

// --- a stub React good enough for a createElement tree -----------------------
const React = {
  createElement(type, props, ...children) {
    return { type, props: props || {}, children: children.filter((child) => child !== undefined && child !== null) };
  },
  Fragment: Symbol('Fragment'),
  useState(initial) {
    return [typeof initial === 'function' ? initial() : initial, () => {}];
  },
  useEffect() {},
  useRef(value) {
    return { current: value };
  },
  useCallback(fn) {
    return fn;
  },
};

// --- a fake DSH module loader ------------------------------------------------
const registered = [];
const injected = [];
const ctx = {
  slots: {
    inject(name, fn) {
      injected.push(name);
      fn();
    },
    register(slot, component) {
      registered.push({ slot, component });
      return () => {};
    },
  },
};

const sandbox = {
  window: {
    __ModuleLoader__: {
      load(definition) {
        sandbox.__loaded = definition;
      },
    },
    setTimeout: () => 0,
    setInterval: () => 0,
    clearInterval: () => {},
  },
  console,
  fetch: () => ({ then: () => {}, catch: () => {} }),
  EventSource: function EventSource() {},
  JSON,
  Math,
  Date,
  String,
  Number,
  Object,
  Array,
};
sandbox.window.console = console;

vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: 'client.js' });

const loaded = sandbox.__loaded;
assert.ok(loaded !== undefined, 'the bundle must call __ModuleLoader__.load');
assert.equal(loaded.id, 'dsh-fine-tune');

const module = loaded.factory((name) => {
  if (name === 'react') return React;
  throw new Error(`unexpected require: ${name}`);
});

// --- the registered seats ----------------------------------------------------
// The two halves are built by separate loaders, so compare plain values rather
// than cross-realm arrays (a vm-created array is not deepEqual to one from here).
assert.equal(module.inject.length, 1);
assert.equal(module.inject[0], 'slots');
assert.equal(typeof module.apply, 'function');
assert.doesNotThrow(() => module.apply(ctx), 'apply must not throw against a healthy host');
assert.deepEqual(injected.sort(), ['main', 'sidebar.panellist']);
assert.equal(registered.length, 2, 'exactly two seats: the panel and its rail icon');
const main = registered.find((entry) => entry.slot.name === 'main');
const icon = registered.find((entry) => entry.slot.name === 'sidebar.panellist');
assert.equal(main.slot.key, icon.slot.id, 'the main seat and the rail entry must share an id');
assert.equal(typeof main.component, 'function');

// --- hooks only while rendering ------------------------------------------------
// A helper that calls React hooks is only safe if the host renders it as a
// component. One that is invoked directly (`Dropdown(...)`) runs those hooks inside
// the parent, conditionally, and React then throws "Rendered more hooks than during
// the previous render" ? which blanked the entire panel in a real session. This
// stub refuses hooks outside a component render so that mistake fails here first.
{
  let depth = 0;
  const strict = {
    createElement(type, props, ...children) {
      if (typeof type === 'function') {
        depth += 1;
        try { type(props || {}); } finally { depth -= 1; }
      }
      return { type, props: props || {}, children };
    },
    Fragment: React.Fragment,
    useState(initial) {
      if (depth === 0) throw new Error('hook used outside a component render');
      return [typeof initial === 'function' ? initial() : initial, () => {}];
    },
    useEffect() { if (depth === 0) throw new Error('hook used outside a component render'); },
    useRef(value) { if (depth === 0) throw new Error('hook used outside a component render'); return { current: value }; },
    useCallback(fn) { return fn; },
    useMemo(fn) { return fn(); },
  };
  const strictSeats = [];
  const strictModule = loaded.factory((name) => {
    if (name === 'react') return strict;
    throw new Error(`unexpected require: ${name}`);
  });
  strictModule.apply({
    slots: {
      inject: (name, fn) => fn(),
      register: (slot, component) => { strictSeats.push({ slot, component }); return () => {}; },
    },
  });
  const strictMain = strictSeats.find((entry) => entry.slot.name === 'main');
  depth += 1;
  try {
    assert.doesNotThrow(() => strictMain.component({}), 'the panel must render without breaking hook order');
  } finally {
    depth -= 1;
  }
}

assert.equal(typeof icon.component, 'function');

// A host without `slots` must degrade to "no panel", not throw.
const quiet = console.error;
console.error = () => {};
try {
  assert.doesNotThrow(() => module.apply({}));
  assert.doesNotThrow(() => module.apply(undefined));
} finally {
  console.error = quiet;
}

// --- pure helpers ------------------------------------------------------------
const internals = module._internals;
assert.ok(internals !== undefined);

const points = internals.lossPoints([
  { kind: 'metric', values: { loss: 1.5, step: 1 } },
  { kind: 'log', message: 'x' },
  { kind: 'metric', values: { loss: 1.2, step: 2 } },
  { kind: 'metric', values: { step: 3 } },
]);
assert.equal(JSON.stringify(points), JSON.stringify([{ step: 1, loss: 1.5 }, { step: 2, loss: 1.2 }]));

const record = { artifacts: ['C:/x/adapter.gguf'] };
const greenEval = { kind: 'evaluation', verdict: { status: 'green', reasons: [] } };
assert.equal(internals.publishState({ allowPublish: false }, { record, events: [greenEval] }).ok, false);
assert.equal(internals.publishState({ allowPublish: true }, null).ok, false);
assert.equal(internals.publishState({ allowPublish: true }, { record, events: [] }).ok, false);
assert.equal(internals.publishState({ allowPublish: true }, { record, events: [{ kind: 'evaluation', verdict: { status: 'blocked', reasons: ['x'] } }] }).ok, false);
assert.equal(internals.publishState({ allowPublish: true }, { record: { artifacts: [] }, events: [greenEval] }).ok, false);
const open = internals.publishState({ allowPublish: true }, { record, events: [greenEval] });
assert.equal(open.ok, true);
assert.match(open.reason === null ? '' : open.reason, /^$/);

// The API base must point at the host's action route, which is the only writer.
assert.equal(internals.API, '/fine-tune/api');

// --- the training plan and its validation -----------------------------------
const plan = JSON.parse(JSON.stringify(internals.defaultPlan()));
assert.equal(plan.mode, 'qlora');
assert.equal(internals.validatePlan(plan), null, 'the default plan must be valid');
assert.match(internals.validatePlan({ ...plan, mode: 'dora' }), /qlora \/ lora/);
assert.match(internals.validatePlan({ ...plan, loraR: 12 }), /4\/8\/16\/32\/64/);
assert.match(internals.validatePlan({ ...plan, maxSeqLen: 99999 }), /128–8192/);
assert.match(internals.validatePlan({ ...plan, learningRate: 1 }), /1e-6–1e-2/);
assert.match(internals.validatePlan({ ...plan, modelId: 'x; rm -rf /' }), /不安全/);
// Both modes are offered, and switching to LoRA keeps the plan valid.
assert.equal(typeof internals.MODE_INFO.qlora, 'object');
assert.equal(typeof internals.MODE_INFO.lora, 'object');
assert.equal(internals.validatePlan({ ...plan, mode: 'lora', quantType: 'nf4' }), null);

// --- the evaluation comparison flow ------------------------------------------
const pair = {
  cases: [
    { index: 0, prompt: '看看 README', expected: '{"tool":"read"}', flip: false, answers: { A: 'A0', B: 'B0' } },
    { index: 1, prompt: '跑测试', expected: '{"tool":"bash"}', flip: true, answers: { A: 'A1', B: 'B1' } },
  ],
};
const prompt = internals.buildJudgePrompt(pair, { knownTools: ['read', 'bash'] });
assert.match(prompt, /#0/);
assert.match(prompt, /#1/);
assert.match(prompt, /read, bash/, 'the judge is told which tools existed');
assert.match(prompt, /"index": 编号/, 'the judge is told the exact reply format');

// A line that claims to be a verdict but does not parse (a truncated reply) is
// reported; prose that happens to contain braces is not.
const withGarbage = internals.parseJudgeReply('{"index": 2, "winner": ');
assert.equal(withGarbage.unknown.length, 1, 'a broken verdict is counted');
const notAVerdict = internals.parseJudgeReply('see {some note} and {}');
assert.equal(notAVerdict.unknown.length, 0, 'a stray brace is not a malformed verdict');

const parsed = internals.parseJudgeReply('```json\n{"index": 0, "winner": "A", "reason": "对"}\n{"index": 1, "winner": "b", "reason": "更好"}\n```');
assert.equal(parsed.verdicts.length, 2, 'two well-formed verdicts');
assert.equal(parsed.verdicts[1].winner, 'B', 'lower-case winners are accepted');
assert.equal(parsed.unknown.length, 0);

const tally = internals.tallyVerdicts(pair.cases, parsed.verdicts);
assert.equal(tally.judged, 2);
assert.equal(tally.aWins, 1);
assert.equal(tally.bWins, 1);
// A judge that prefers "A" every time must not translate to "the new model won":
// case 0 has A = candidate, case 1 has A = base.
const allA = internals.tallyVerdicts(pair.cases, [{ index: 0, winner: 'A' }, { index: 1, winner: 'A' }]);
assert.equal(allA.aWins, 2);

console.log('client.js: bundle loads, registers two seats, plan + judge helpers behave');
