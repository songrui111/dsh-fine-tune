/**
 * The install flow: detect, create the venv, install, verify.
 *
 * Each step reports through `onEvent` so the panel can show progress instead of a
 * spinner. Nothing is downloaded until every precondition passes — a user on a
 * machine without a supported GPU must not wait fifteen minutes to be told.
 *
 * The steps are exported separately from the runner so each can be tested without
 * touching the network or spawning pip.
 */

import { existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  PINS,
  VERIFY_SCRIPT,
  installCommands,
  parseGpuListing,
  pickCuda,
  planForVram,
  readReady,
  run,
  venvPaths,
  writeReady,
} from './bootstrap.mjs';

/** Find a Python to build the venv from. Windows only: see lib/bootstrap.mjs. */
export async function findSystemPython() {
  const candidates = [
    'py', // the Windows launcher, if present
    'python',
    'python3',
  ];
  for (const candidate of candidates) {
    const args = candidate === 'py' ? ['-3', '-c', 'import sys; print(sys.version_info[:2])'] : ['-c', 'import sys; print(sys.version_info[:2])'];
    const result = await run(candidate, args);
    if (result.code !== 0) continue;
    const match = /\((\d+),\s*(\d+)\)/.exec(result.stdout);
    if (match === null) continue;
    const major = Number(match[1]);
    const minor = Number(match[2]);
    // 3.10 is the floor: the pinned torch has no wheels below it.
    if (major !== 3 || minor < 10) continue;
    return { command: candidate, major, minor, version: `${major}.${minor}` };
  }
  return null;
}

/** Detect the GPU, or explain precisely why nothing can proceed. */
export async function detectGpu() {
  const result = await run('nvidia-smi', [
    '--query-gpu=name,memory.total,driver_version',
    '--format=csv,noheader',
  ]);
  if (result.code !== 0) {
    return {
      ok: false,
      reason: 'nvidia-smi 不可用：这台机器没有 NVIDIA 显卡，或驱动没有安装。',
    };
  }
  const gpu = parseGpuListing(result.stdout);
  if (gpu === null) {
    return { ok: false, reason: `nvidia-smi 有输出但读不出显卡：${result.stdout.trim().slice(0, 200)}` };
  }
  const cuda = await run('nvidia-smi', []);
  // The driver's CUDA version is a header line ("CUDA Version: 12.1"); the query
  // form above does not carry it.
  const match = /CUDA Version:\s*([\d.]+)/.exec(cuda.stdout);
  const driverCuda = match === null ? null : match[1];
  const wheel = pickCuda(driverCuda);
  if (wheel === null) {
    return {
      ok: false,
      gpu,
      reason: `驱动报告的 CUDA 版本是 ${driverCuda ?? '未知'}，没有匹配的 torch 轮子。请升级 NVIDIA 驱动到支持 CUDA 11.8 以上。`,
    };
  }
  return { ok: true, gpu, driverCuda, wheel, plan: planForVram(gpu.totalMiB) };
}

/**
 * Run the whole install.
 *
 * @param {object} options
 * @param {string} options.root  Plugin data directory (`$DSH_HOME/fine-tune`).
 * @param {(event: {phase: string, message: string, level?: string}) => void} [options.onEvent]
 * @param {boolean} [options.force] Rebuild even when a ready venv exists.
 */
export async function install({ root, onEvent = () => {}, force = false }) {
  const emit = (phase, message, level = 'info') => onEvent({ phase, message, level });
  const paths = venvPaths(root);

  if (process.platform !== 'win32') {
    return {
      ok: false,
      reason: `当前只支持 Windows（检测到 ${process.platform}）。Linux 的安装矩阵还没有被验证过，与其装一半失败，不如直接说明。`,
    };
  }

  if (!force) {
    const ready = readReady(root);
    if (ready !== null && ready.current === true) {
      return { ok: true, already: true, pythonPath: paths.python, detail: ready };
    }
    if (ready !== null && ready.current === false) {
      emit('detect', '已有的环境是用不同版本装的，需要重建', 'warn');
    }
  }

  emit('detect', '检查显卡…');
  const gpu = await detectGpu();
  if (gpu.ok !== true) {
    return { ok: false, reason: gpu.reason, gpu: gpu.gpu ?? null };
  }
  emit('detect', `显卡：${gpu.gpu.name}，${(gpu.gpu.totalMiB / 1024).toFixed(1)} GiB 显存`, 'ok');
  emit('detect', `驱动 CUDA ${gpu.driverCuda} → 使用 torch 的 CUDA ${gpu.wheel.cuda} 轮子`,
    gpu.wheel.unvalidated === true ? 'warn' : 'ok');
  if (gpu.wheel.unvalidated === true) {
    // Honest about the test matrix: cu121 is what the whole training and eval loop
    // was measured on. Other wheels install the same packages but were not run here.
    emit('detect', `注意：CUDA ${gpu.wheel.cuda} 不是本项目验证过的版本（验证过的是 cu121）。装好后如果训练出错，先怀疑这个差异。`, 'warn');
  }
  emit('detect', gpu.plan.summary, gpu.plan.verdict === 'insufficient' ? 'warn' : 'ok');
  if (gpu.plan.verdict === 'insufficient') {
    return {
      ok: false,
      reason: `${gpu.plan.summary}。装完也训不动，所以没有下载任何东西。`,
      gpu: gpu.gpu,
    };
  }

  emit('detect', '查找 Python 3.10 以上…');
  const python = await findSystemPython();
  if (python === null) {
    return {
      ok: false,
      reason: '找不到 Python 3.10 或更新的版本。请先安装 Python（python.org，勾选 Add to PATH），再重试。',
      gpu: gpu.gpu,
    };
  }
  emit('detect', `使用 ${python.command}（Python ${python.version}）`, 'ok');

  if (force && existsSync(paths.dir)) {
    emit('venv', '删除旧环境…');
    rmSync(paths.dir, { recursive: true, force: true });
  }
  mkdirSync(root, { recursive: true });
  if (!existsSync(paths.python)) {
    emit('venv', `创建虚拟环境：${paths.dir}`);
    const created = await run(python.command, python.command === 'py' ? ['-3', '-m', 'venv', paths.dir] : ['-m', 'venv', paths.dir]);
    if (created.code !== 0) {
      return { ok: false, reason: `创建虚拟环境失败：${created.stderr.slice(-400)}` };
    }
  } else {
    emit('venv', '复用已有虚拟环境', 'ok');
  }

  // pip talks to the network; every line is forwarded so a slow download does not
  // look like a hang. A failure keeps the last lines, which is where pip puts the
  // reason.
  const commands = installCommands(gpu.wheel, paths.python);
  for (const [index, command] of commands.entries()) {
    emit('install', `[${index + 1}/${commands.length}] ${command.label}`);
    const result = await run(paths.python, command.args, {
      onLine: (line) => {
        // Only lines that carry progress or an error: pip's full output is noise
        // in a panel, and forwarding it would drown the useful lines.
        if (/Downloading|Collecting|Installing collected|Successfully installed|error|ERROR|warning/i.test(line)) {
          emit('install', line.slice(0, 200));
        }
      },
    });
    if (result.code !== 0) {
      return {
        ok: false,
        reason: `${command.label} 失败。最后几行：\n${result.stderr.split('\n').slice(-8).join('\n')}`,
      };
    }
    emit('install', `${command.label} 完成`, 'ok');
  }

  emit('verify', '验证 GPU 真的能用（不只是 import 成功）…');
  const verify = await run(paths.python, ['-c', VERIFY_SCRIPT], {
    onLine: (line) => emit('verify', line.slice(0, 200)),
  });
  if (verify.code !== 0) {
    return {
      ok: false,
      reason: `依赖装好了但 GPU 验证失败：\n${(verify.stderr || verify.stdout).split('\n').slice(-8).join('\n')}`,
      pythonPath: paths.python,
    };
  }

  const detail = {
    gpu: gpu.gpu,
    driverCuda: gpu.driverCuda,
    wheelCuda: gpu.wheel.cuda,
    plan: gpu.plan,
    pythonVersion: python.version,
  };
  writeReady(root, detail);
  emit('done', '环境就绪', 'ok');
  return { ok: true, pythonPath: paths.python, detail };
}
