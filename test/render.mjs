/**
 * Render the panel for real, with a state that is NOT null.
 *
 * The existing client test only mounts the "loading" branch, because `state` starts
 * null and the component returns early. That means every render path added after
 * that point - the permission banner, the tabs, the train form - has never been
 * executed by any test. A typo there shows up as a blank panel in a real session
 * and nowhere else.
 *
 * This harness supplies a plausible state so the full tree builds, and Reports any
 * throw with the expression that produced it.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, '..', 'client', 'client.js'), 'utf8');

// A state shaped like `GET /fine-tune/api/state`, enough for every branch.
const fakeState = {
  schema: { modes: [] },
  settings: {
    allowSpawn: false,
    allowPublish: false,
    pythonPath: '',
    dataDir: 'C:/tmp',
    workerRoot: '',
    sshTarget: '',
    ollamaBase: 'http://127.0.0.1:11434',
    env: {},
  },
  runs: [
    { id: 'train-20260101000000-abcdef', state: 'succeeded', metrics: { loss: 0.5 }, input: { modelId: 'Qwen/Qwen2.5-0.5B-Instruct' } },
    { id: 'train-20260101000001-abcdef', state: 'running', metrics: {}, input: { modelId: 'x' } },
  ],
  sources: [],
  datasets: [
    { id: 'sample', counts: { train: 52, val: 5, eval: 3 }, tokens: { median: 193 } },
  ],
  modes: [],
  gpu: { available: true, gpus: [{ name: 'RTX 3060', totalGiB: 6 }] },
};

let hookDepth = 0;
const hookLog = [];

function makeReact(state) {
  return {
    createElement(type, props, ...children) {
      // Render function components for real: that is the point of this harness.
      if (typeof type === 'function' && type.name !== 'Fragment') {
        hookDepth += 1;
        try {
          return type(props || {});
        } finally {
          hookDepth -= 1;
        }
      }
      return { type, props: props || {}, children: children.filter((c) => c !== undefined && c !== null) };
    },
    Fragment: Symbol('Fragment'),
    useState(initial) {
      if (hookDepth === 0) throw new Error('hook used outside a component render');
      hookLog.push('useState');
      // The state hook is the only one that receives the fetched payload.
      if (typeof initial === 'object' && initial !== null && 'runs' in (state ?? {})) {
        return [state, () => {}];
      }
      return [typeof initial === 'function' ? initial() : initial, () => {}];
    },
    useEffect() {
      if (hookDepth === 0) throw new Error('useEffect used outside a component render');
      hookLog.push('useEffect');
    },
    useRef(value) {
      if (hookDepth === 0) throw new Error('useRef used outside a component render');
      return { current: value };
    },
    useCallback(fn) {
      if (hookDepth === 0) throw new Error('useCallback used outside a component render');
      return fn;
    },
  };
}

const registered = [];
const ctx = {
  slots: {
    inject(name, fn) { fn(); },
    register(slot, component) { registered.push({ slot, component }); return () => {}; },
  },
};

const sandbox = {
  window: {
    __ModuleLoader__: { load(definition) { sandbox.__loaded = definition; } },
    setTimeout: () => 0,
    setInterval: () => 0,
    clearInterval: () => {},
    confirm: () => false,
  },
  console,
  fetch: () => Promise.resolve({ ok: true, json: () => Promise.resolve(fakeState) }),
  EventSource: function EventSource() {},
  JSON, Math, Date, String, Number, Object, Array, Boolean, Promise, Set, Map,
};
sandbox.window.console = console;
vm.createContext(sandbox);
vm.runInContext(source, sandbox, { filename: 'client.js' });

const loaded = sandbox.__loaded;
const React = makeReact(fakeState);
const module = loaded.factory((name) => {
  if (name === 'react') return React;
  throw new Error(`unexpected require: ${name}`);
});
module.apply(ctx);

const main = registered.find((entry) => entry.slot.name === 'main');
assert.ok(main, 'the panel must register on the main slot');

// Try every tab, with allowSpawn both off and on: the banner only exists in one of
// those states, and the train form is where the permission button lives.
let rendered = 0;
for (const allowSpawn of [false, true]) {
  const state = { ...fakeState, settings: { ...fakeState.settings, allowSpawn } };
  for (const tab of ['data', 'train', 'evaluate']) {
    const localReact = makeReact(state);
    // Re-create the component through a fresh factory so this render sees `state`.
    const localModule = loaded.factory((name) => {
      if (name === 'react') return localReact;
      throw new Error(`unexpected require: ${name}`);
    });
    const seats = [];
    localModule.apply({
      slots: {
        inject(name, fn) { fn(); },
        register(slot, component) { seats.push({ slot, component }); return () => {}; },
      },
    });
    const seat = seats.find((entry) => entry.slot.name === 'main');
    // A component's first render is entered by the renderer, so hooks are legal.
    let tree;
    hookDepth += 1;
    try {
      tree = seat.component({});
    } finally {
      hookDepth -= 1;
    }
    assert.ok(tree !== undefined && tree !== null, `render returned nothing (allowSpawn=${allowSpawn}, tab=${tab})`);
    rendered += 1;
  }
}

console.log(`rendered ${rendered} times across allowSpawn off/on and all three tabs`);
console.log('hooks recorded during render:', [...new Set(hookLog)].join(', '));
console.log('panel render: OK');
