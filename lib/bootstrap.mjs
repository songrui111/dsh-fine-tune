/**
 * Bootstrap the Python environment a training run needs.
 *
 * ## Why this is not a `postinstall`
 *
 * Installing torch is a multi-gigabyte download whose *version* depends on the
 * user's graphics card, and that is not something to do silently behind an
 * `npm install` that the user believes finished. So it is an explicit action,
 * offered by the panel, with progress. A user who never trains never pays for it.
 *
 * ## What is checked, and in what order
 *
 *   1. Windows. The CUDA wheels below are the ones verified on this project; Linux
 *      would need a second matrix nobody has run, and claiming it works untested
 *      is worse than refusing it.
 *   2. An NVIDIA GPU via `nvidia-smi`, plus the driver's CUDA version. The torch
 *      build must not be newer than the driver supports.
 *   3. Free VRAM, which decides which models the result can train at all.
 *   4. A Python 3.10+ interpreter to build the venv from.
 *
 * Only after all four does anything get downloaded.
 *
 * ## Why the venv lives outside the plugin
 *
 * `$DSH_HOME/fine-tune/venv`, not `<plugin>/python/.venv`. It is 5–8 GB, and a
 * plugin update or reinstall replaces the plugin directory — putting the venv
 * there would make every update re-download torch. This location survives.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spawn } from 'node:child_process';

/** Exact versions that were validated together on this project. */
export const PINS = {
  torch: '2.5.1',
  transformers: '4.46.3',
  peft: '0.13.2',
  datasets: '3.1.0',
  accelerate: '1.1.1',
  bitsandbytes: '0.44.1',
  safetensors: '0.4.5',
};

/**
 * CUDA wheel index per driver generation, **preferred first**.
 *
 * Two rules that pull in opposite directions, and the order here is the
 * resolution:
 *
 *   - a wheel newer than the driver supports fails with "CUDA driver version is
 *     insufficient", which reads like a broken install rather than a wrong choice
 *   - but newer is not better here. `cu121` is the build this project was
 *     validated on (2.5.1+cu121, RTX 3060, the whole training and eval loop), so it
 *     is what a typical driver gets. `cu124` is only handed out when the driver is
 *     too new for the validated build ? never merely because it is available.
 *
 * `minDriverCuda` is the minimum driver-level CUDA that can load the wheel.
 */
export const WHEEL_INDEX = [
  { minDriverCuda: 12.1, maxDriverCuda: 12.3, cuda: '121', index: 'https://download.pytorch.org/whl/cu121', validated: true },
  { minDriverCuda: 11.8, maxDriverCuda: 12.0, cuda: '118', index: 'https://download.pytorch.org/whl/cu118' },
  { minDriverCuda: 12.4, maxDriverCuda: 99.9, cuda: '124', index: 'https://download.pytorch.org/whl/cu124' },
];

/**
 * The wheel a driver at `driverCuda` should use, or null when nothing fits.
 *
 * CUDA is forward compatible within a major version: a driver reporting 13.1 runs
 * cu121 and cu124 alike, so a new driver is not a reason to skip the validated
 * build. The bands above exist to catch the reverse ? a driver too old for a wheel.
 */
export function pickCuda(version) {
  const match = /^(\d+)\.(\d+)/.exec(String(version ?? '').trim());
  if (match === null) return null;
  const driverCuda = Number(match[1]) * 10 + Number(match[2]);
  // The validated build first when the driver can run it at all.
  const validated = WHEEL_INDEX.find((entry) => entry.validated === true);
  if (validated !== undefined && driverCuda >= Math.round(validated.minDriverCuda * 10)) return validated;
  // Otherwise the oldest wheel the driver still supports: closest to what was tested.
  const usable = WHEEL_INDEX.filter((entry) => driverCuda >= Math.round(entry.minDriverCuda * 10));
  if (usable.length === 0) return null;
  // Flagged so the panel can say "this build was not validated here" instead of
  // presenting every install as equally certain.
  return { ...usable.sort((a, b) => a.minDriverCuda - b.minDriverCuda)[0], unvalidated: true };
}

/**
 * Parse `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`.
 *
 * Returns the *largest* card when several are present: training targets one GPU,
 * and reporting the smallest would understate what is possible.
 */
export function parseGpuListing(text) {
  const gpus = [];
  for (const line of String(text ?? '').split('\n')) {
    const trimmed = line.trim();
    if (trimmed === '') continue;
    const parts = trimmed.split(',').map((part) => part.trim());
    if (parts.length < 2) continue;
    const name = parts[0];
    // The memory column carries a unit; stripping it is what broke the GPU probe
    // once already (`Number("6144 MiB")` is NaN and the panel said "no GPU").
    const memory = /(\d+(?:\.\d+)?)\s*(MiB|GiB|MB|GB)/i.exec(parts[1]);
    if (memory === null) continue;
    const amount = Number(memory[1]);
    const unit = memory[2].toLowerCase();
    const totalMiB = unit.startsWith('g') ? Math.round(amount * 1024) : Math.round(amount);
    gpus.push({ name, totalMiB, driverVersion: parts[2] ?? null });
  }
  if (gpus.length === 0) return null;
  return gpus.sort((a, b) => b.totalMiB - a.totalMiB)[0];
}

/**
 * Which training modes a VRAM budget can actually run.
 *
 * Thresholds are the measured peaks from this project, not estimates: QLoRA on a
 * 0.5B model at seq 1024 peaked at 2.86 GiB, and 1.5B at the same length needed
 * 6.59 GiB. A card that cannot fit the smallest useful configuration is told so
 * before it downloads anything.
 */
export function planForVram(totalMiB) {
  const gib = totalMiB / 1024;
  if (gib >= 20) {
    return { verdict: 'comfortable', summary: '可以训 7B 级别的模型（QLoRA）', maxModel: 'Qwen2.5-7B-Instruct' };
  }
  if (gib >= 11) {
    return { verdict: 'good', summary: '可以训 3B–4B 级别的模型（QLoRA）', maxModel: 'Qwen2.5-3B-Instruct' };
  }
  if (gib >= 7) {
    return { verdict: 'workable', summary: '可以训 1.5B 级别的模型（QLoRA，seq ≤ 1024）', maxModel: 'Qwen2.5-1.5B-Instruct' };
  }
  if (gib >= 5) {
    return { verdict: 'minimal', summary: '只能训 0.5B 模型（QLoRA，seq ≤ 1024）—— 已验证可用，效果有限', maxModel: 'Qwen2.5-0.5B-Instruct' };
  }
  return { verdict: 'insufficient', summary: `显存 ${gib.toFixed(1)} GiB 不足：最小的可用配置需要约 3 GiB 峰值余量`, maxModel: null };
}

/** Paths the bootstrap writes to, all under the plugin's data directory. */
export function venvPaths(root) {
  return {
    dir: join(root, 'venv'),
    python: join(root, 'venv', 'Scripts', 'python.exe'),
    ready: join(root, 'venv', '.ready.json'),
  };
}

/** Has a previous bootstrap completed, and against which pins? */
export function readReady(root) {
  const { ready } = venvPaths(root);
  if (!existsSync(ready)) return null;
  try {
    const state = JSON.parse(readFileSync(ready, 'utf8'));
    // A venv built against different pins is not "ready" for this build: the
    // versions were validated together, so a mismatch is reported, not trusted.
    const same = Object.entries(PINS).every(([name, version]) => state.pins?.[name] === version);
    return { ...state, current: same };
  } catch {
    return null;
  }
}

/** Record a successful bootstrap so the panel stops offering to install. */
export function writeReady(root, detail) {
  const { ready } = venvPaths(root);
  mkdirSync(join(root, 'venv'), { recursive: true });
  writeFileSync(ready, `${JSON.stringify({ pins: PINS, installedAt: new Date().toISOString(), ...detail }, null, 2)}\n`, 'utf8');
}

/**
 * Run one command, streaming every output line through `onLine`.
 *
 * A bootstrap that prints nothing for fifteen minutes looks hung, so both streams
 * are forwarded as they arrive rather than collected at the end.
 */
export function run(command, args, options = {}) {
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: { ...process.env, ...(options.env ?? {}), PIP_DISABLE_PIP_VERSION_CHECK: '1' },
      windowsHide: true,
    });
    let stdout = '';
    let stderr = '';
    const forward = (chunk, sink) => {
      const text = chunk.toString();
      sink === 'out' ? (stdout += text) : (stderr += text);
      if (typeof options.onLine === 'function') {
        for (const line of text.split('\n')) {
          const trimmed = line.trim();
          if (trimmed !== '') options.onLine(trimmed, sink);
        }
      }
    };
    child.stdout?.on('data', (chunk) => forward(chunk, 'out'));
    child.stderr?.on('data', (chunk) => forward(chunk, 'err'));
    child.on('error', (error) => resolve({ code: -1, stdout, stderr: `${stderr}${error.message}` }));
    child.on('close', (code) => resolve({ code: code ?? -1, stdout, stderr }));
  });
}

/** Argv that installs the torch build matching `wheel`, plus the rest of the stack. */
export function installCommands(wheel, python) {
  const pip = [python, '-m', 'pip', 'install', '--no-input', '--progress-bar', 'off'];
  return [
    {
      label: '升级 pip',
      args: [...pip.slice(1), '--upgrade', 'pip', 'setuptools', 'wheel'],
    },
    {
      // torch is installed first and from its own index: mixing CUDA wheels with
      // PyPI's CPU builds in one resolve is how a "GPU" install ends up on CPU.
      label: `安装 torch ${PINS.torch} (CUDA ${wheel.cuda})`,
      args: [...pip.slice(1), `torch==${PINS.torch}`, '--index-url', wheel.index],
    },
    {
      label: '安装训练依赖',
      args: [...pip.slice(1),
        `transformers==${PINS.transformers}`,
        `peft==${PINS.peft}`,
        `datasets==${PINS.datasets}`,
        `accelerate==${PINS.accelerate}`,
        `safetensors==${PINS.safetensors}`,
        'hf_transfer'],
    },
    {
      label: `安装 bitsandbytes ${PINS.bitsandbytes}（QLoRA 需要）`,
      args: [...pip.slice(1), `bitsandbytes==${PINS.bitsandbytes}`],
    },
  ];
}

/** The one-liner that proves the install actually works, not just imports. */
export const VERIFY_SCRIPT = [
  'import torch',
  'print("torch", torch.__version__)',
  'print("cuda_built", torch.version.cuda)',
  'assert torch.cuda.is_available(), "CUDA is not available to torch"',
  'a = torch.randn(512, 512, device="cuda", dtype=torch.float16)',
  'b = torch.randn(512, 512, device="cuda", dtype=torch.float16)',
  '(a @ b).sum().item()',
  'print("matmul", "ok")',
  'from transformers import AutoTokenizer',
  'from peft import LoraConfig',
  'import bitsandbytes, datasets, accelerate',
  'print("stack", "ok")',
].join('\n');
