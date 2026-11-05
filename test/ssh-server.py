"""A throwaway SSH server for testing the plugin's ssh/sftp transport end to end.

Run:  <venv>/python.exe test/ssh-server.py <port>

It accepts one key (printed below on startup), serves a shell command channel
and an SFTP subsystem rooted at a temp directory, and exits when killed. This is
test scaffolding only — nothing in the plugin imports it, and it is not part of
the shipped package.

Why bother: the transport layer's interesting failures are protocol-shaped
(a channel that closes early, an sftp batch line the server rejects, a path with a
space), and none of them can be reproduced with a mock. A real server proves the
code speaks the protocol; it says nothing about a *specific* remote machine,
which is why the real target still has to be probed once it exists.
"""

import asyncio
import base64
import io
import os
import re
import shlex
import sys
import tempfile

import asyncssh
from asyncssh import sftp as sftp_mod

# Where this checkout lives. Declared first because the key directory below is
# derived from it.
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Keys live inside the checkout, not in %TEMP% or the user profile:
#
#   - a key in TEMP inherits an ACL whose owner is BUILTIN\Administrators, and
#     OpenSSH refuses it with "UNPROTECTED PRIVATE KEY FILE" because the *user*
#     cannot tighten permissions on a file they do not own
#   - the profile root and Documents are not writable by the process running the
#     tests (Administrators-owned, and the harness sandbox blocks them)
#   - the checkout is writable and the directory is gitignored
#
# The key must be owned by the user or ssh will not use it.
_KEY_DIR = os.path.join(_PLUGIN_ROOT, '.test-keys')
os.makedirs(_KEY_DIR, exist_ok=True)
HOST_KEY_PATH = os.path.join(_KEY_DIR, 'host-key')
CLIENT_KEY_PATH = os.path.join(_KEY_DIR, 'client-key')

# Where the real worker runs. The end-to-end test needs `fnt_worker` importable by
# a Python that has the ML stack; the plugin's own venv is the only such one here.
WORKER_PYTHON = os.path.join(_PLUGIN_ROOT, 'python', '.venv', 'Scripts', 'python.exe')
WORKER_CWD = os.path.join(_PLUGIN_ROOT, 'python')


def _tighten(path):
    """Give the key to the current user alone, or OpenSSH refuses to read it.

    Windows: `%TEMP%` and a git checkout both inherit ACLs with entries ssh will not
    accept ? an Administrators-owned owner, or a stale `UNKNOWN\\UNKNOWN` SID left by
    an ACL rewrite. `icacls /inheritance:r` plus one grant is the fix, and doing it
    here means the test suite works on a fresh checkout without manual setup.
    Non-Windows: chmod 600 is the equivalent.
    """
    try:
        if os.name == 'nt':
            import subprocess

            user = os.environ.get('USERNAME', '')
            subprocess.run(['icacls', path, '/inheritance:r'], check=False, capture_output=True)
            if user:
                subprocess.run(['icacls', path, '/grant:r', f'{user}:R'], check=False, capture_output=True)
        else:
            os.chmod(path, 0o600)
    except Exception:
        # A permission fix that cannot run is reported by ssh itself, which is a
        # clearer error than anything printed here.
        pass


def ensure_keys():
    """Generate the host key and the client keypair once, reused across runs."""
    if not os.path.exists(HOST_KEY_PATH):
        key = asyncssh.generate_private_key('ssh-ed25519')
        key.write_private_key(HOST_KEY_PATH)
    if not os.path.exists(CLIENT_KEY_PATH):
        key = asyncssh.generate_private_key('ssh-ed25519')
        key.write_private_key(CLIENT_KEY_PATH)
        pub = key.export_public_key().decode().strip()
        with open(CLIENT_KEY_PATH + '.pub', 'w', encoding='utf-8') as handle:
            handle.write(pub + '\n')
    _tighten(HOST_KEY_PATH)
    _tighten(CLIENT_KEY_PATH)
    return asyncssh.read_private_key(HOST_KEY_PATH), asyncssh.read_private_key(CLIENT_KEY_PATH)


def make_sftp_factory(root):
    """An SFTP subsystem rooted at `root`, using asyncssh's own chroot.

    Written by hand at first (override `map_path`), which failed in a way worth
    recording: `-mkdir "/e2e/data"` reported success while nothing was created,
    because a hand-rolled map left asyncssh's other path handling (the SFTPServer
    `_chroot` bookkeeping used by `open`/`mkdir`) pointing at the unmapped name.
    `chroot=` is the documented way to get this right, and on Windows asyncssh
    normalises the backslashes itself.
    """

    def factory(chan):
        return RootedSFTPServer(chan, chroot=root)

    return factory


class RootedSFTPServer(sftp_mod.SFTPServer):
    """SFTPServer that gets its chroot from the constructor, nothing overridden."""

    def __init__(self, chan, chroot=None, **kwargs):
        super().__init__(chan, chroot=chroot, **kwargs)


class TestSSHServer(asyncssh.SSHServer):
    def begin_auth(self, username):
        return True

    def public_key_auth_supported(self):
        return True

    def validate_public_key(self, username, key):
        return key.export_public_key().decode().strip() == self._authorized

    def server_version(self):
        return 'dsh-test-sshd', '1.0'


async def handle_client(process):
    """A POSIX-ish command channel, emulated so a Windows host can serve it.

    The plugin's remote commands are POSIX (`ls -1`, `python3 -V`, `mkdir -p`), and
    this test box is Windows, so the handful of commands the plugin actually sends
    are reimplemented here instead of delegated to cmd.exe. That keeps the test
    about the *transport* rather than about the host OS — a real remote is Linux
    and these all exist there for free.

    The one exception is `python3 -m fnt_worker ...`, which is delegated to the
    real venv interpreter: the point of the end-to-end test is that the *worker*
    runs behind a real SSH channel, so faking it would defeat the test.
    """
    command = process.command
    if command is None:
        process.exit(1)
        return

    command = command.strip()
    if command == 'exit 0':
        process.exit(0)
        return
    if command.startswith('exit '):
        part = command.split()[1]
        process.exit(int(part) if part.isdigit() else 1)
        return
    if command.startswith('sleep '):
        await asyncio.sleep(float(command.split()[1]))
        process.exit(0)
        return

    # `python3 -m fnt_worker <verb> --config <path>`: the real thing.
    worker = re.search(r'python3\s+-m\s+fnt_worker\s+(\w+)\s+--config\s+(\S+)', command)
    if worker:
        verb, config_path = worker.group(1), worker.group(2).strip('"\'')
        # Model cache: inherited from the environment so the test can point the
        # worker at whatever cache this machine actually has. A real remote needs
        # its own cache — see the note in DESIGN.md §10, which is why the run
        # prints the worker's error verbatim instead of hiding it.
        worker_env = {
            **os.environ,
            'PYTHONUNBUFFERED': '1',
            'PYTHONIOENCODING': 'utf-8',
        }
        if 'HF_HOME' not in worker_env and 'HUGGINGFACE_HUB_CACHE' not in worker_env:
            worker_env['HF_HOME'] = os.environ.get('DSH_TEST_HF_HOME', '')
        proc = await asyncio.create_subprocess_exec(
            WORKER_PYTHON, '-m', 'fnt_worker', verb, '--config', config_path,
            cwd=WORKER_CWD,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={key: value for key, value in worker_env.items() if value != ''},
        )

        async def pump(stream, sink):
            while True:
                chunk = await stream.read(8192)
                if not chunk:
                    break
                sink.write(chunk.decode('utf-8', 'replace'))

        await asyncio.gather(pump(proc.stdout, process.stdout), pump(proc.stderr, process.stderr))
        process.exit(await proc.wait() or 0)
        return

    # `ls -1 <dir>` — what downloadArtifacts uses to list the adapter directory.
    # The client path is chroot-relative, but the command channel has no chroot,
    # so map it through the same root the sftp side uses.
    listing = re.match(r'^ls\s+-1\s+"?([^"]+)"?\s*2>/dev/null$', command)
    if listing:
        client_path = os.path.normpath(listing.group(1))
        root_real = os.path.realpath(process.server_root)
        # Two shapes arrive here:
        #   - a chroot-relative path ("/stage/run1/data") from the plugin's own
        #     downloadArtifacts
        #   - the absolute host path a Windows worker reports in its artifact event
        # A path already under the root is used as-is; anything else is joined to it.
        candidate = os.path.realpath(client_path)
        if not (candidate == root_real or candidate.startswith(root_real + os.sep)):
            relative = client_path.lstrip('/\\').replace('/', os.sep).replace('\\', os.sep)
            candidate = os.path.join(root_real, relative)
        try:
            names = sorted(os.listdir(candidate))
        except OSError:
            process.exit(1)
            return
        process.stdout.write('\n'.join(names) + ('\n' if names else ''))
        process.exit(0)
        return

    # `python3 ...` possibly chained with `&&`. The plugin's probe sends exactly
    # `python3 -V && python3 -c "..."`, so chains are handled rather than assumed
    # away — splitting on && before dispatch is what a real shell does.
    if command.startswith('python3'):
        segments = [segment.strip() for segment in re.split(r'\s+&&\s+', command)]
        for segment in segments:
            if not segment.startswith('python3'):
                # Anything else in the chain: echo is the only other thing tests send.
                if segment.startswith('echo '):
                    process.stdout.write(segment[5:].strip('"\'') + '\n')
                continue
            if '-V' in segment or '--version' in segment:
                process.stdout.write(f"Python {sys.version.split()[0]}\n")
                continue
            snippet = re.search(r'-c\s+"([^"]*)"', segment) or re.search(r"-c\s+'([^']*)'", segment)
            if snippet:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, '-c', snippet.group(1),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                out, err = await proc.communicate()
                process.stdout.write(out.decode('utf-8', 'replace'))
                process.stderr.write(err.decode('utf-8', 'replace'))
                if proc.returncode:
                    process.exit(proc.returncode)
        process.exit(0)
        return

    # `mkdir -p <dir>` — the SSH backend's remote directory step.
    mk = re.match(r'^mkdir\s+-p\s+(.+)$', command)
    if mk:
        for raw in shlex.split(mk.group(1)):
            os.makedirs(raw, exist_ok=True)
        process.exit(0)
        return

    # `echo ...` and `&&` chains between the commands we support.
    for part in [piece.strip() for piece in command.split('&&')]:
        if part.startswith('echo '):
            process.stdout.write(part[5:].strip('"') + '\n')
    process.exit(0)


async def main(port):
    host_key, client_key = ensure_keys()
    root = tempfile.mkdtemp(prefix='dsh-ssh-root-')
    os.makedirs(root, exist_ok=True)

    def process_factory(process):
        # The chroot is an sftp concept; the command channel needs it passed in so
        # a `ls -1 /x` and an sftp `ls /x` name the same place.
        process.server_root = root
        return handle_client(process)

    server = TestSSHServer()
    server._authorized = client_key.export_public_key().decode().strip()
    await asyncssh.create_server(
        lambda: server,
        '127.0.0.1',
        port,
        server_host_keys=[host_key],
        sftp_factory=make_sftp_factory(root),
        process_factory=process_factory,
    )
    print(f'READY port={port}', flush=True)
    print(f'ROOT {root}', flush=True)
    print(f'CLIENT_KEY {CLIENT_KEY_PATH}', flush=True)
    await asyncio.Event().wait()


if __name__ == '__main__':
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2222))
