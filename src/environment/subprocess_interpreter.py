# Copyright 2025-2026 Strands RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions copyright 2023-2026 CAMEL-AI.org, Apache-2.0.
# Adapted from camel/interpreters/subprocess_interpreter.py via THUDM/slime.

from __future__ import annotations

import ast
import logging
import os
import signal
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class InterpreterError(Exception):
    r"""Exception raised for errors that can be solved by regenerating code."""


class BaseInterpreter(ABC):
    r"""An abstract base class for code interpreters."""

    @abstractmethod
    def run(self, code: str, code_type: str) -> str:
        r"""Executes the given code based on its type."""

    @abstractmethod
    def supported_code_types(self) -> list[str]:
        r"""Provides supported code types by the interpreter."""

    @abstractmethod
    def update_action_space(self, action_space: dict[str, Any]) -> None:
        r"""Updates action space for *python* interpreter."""

    @abstractmethod
    def execute_command(self, command: str) -> str | tuple[str, str]:
        r"""Executes a command in the interpreter."""


class SubprocessInterpreter(BaseInterpreter):
    """Execute code in a subprocess with optional namespace isolation."""

    _CODE_EXECUTE_CMD_MAPPING: ClassVar[dict[str, dict[str, str]]] = {
        "python": {"posix": "python {file_name}", "nt": "python {file_name}"},
        "bash": {"posix": "bash {file_name}", "nt": "bash {file_name}"},
        "r": {"posix": "Rscript {file_name}", "nt": "Rscript {file_name}"},
    }

    _CODE_EXTENSION_MAPPING: ClassVar[dict[str, str]] = {
        "python": "py",
        "bash": "sh",
        "r": "R",
    }

    _CODE_TYPE_MAPPING: ClassVar[dict[str, str]] = {
        "python": "python",
        "py3": "python",
        "python3": "python",
        "py": "python",
        "shell": "bash",
        "bash": "bash",
        "sh": "bash",
        "r": "r",
        "R": "r",
    }

    _RLIMIT_BOOTSTRAP = "import resource as _r; _r.setrlimit(_r.RLIMIT_AS, ({b}, {b})); del _r\n"

    _UNSHARE = ["unshare", "--map-root-user", "--net", "--mount", "--pid", "--fork", "--kill-child",
                "--propagation", "private", "--"]

    _SANDBOX_BOOTSTRAP = r"""
import ctypes, os, platform, runpy, shutil, sys
libc = ctypes.CDLL(None, use_errno=True)
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC, MS_REMOUNT, MS_BIND, MS_REC = 1, 2, 4, 8, 32, 4096, 16384
MS_RELATIME, MS_STRICTATIME, MS_NOATIME, MS_NODIRATIME = 1 << 21, 1 << 24, 1024, 2048
def die(what):
    raise SystemExit("sandbox: %s failed: %s" % (what, os.strerror(ctypes.get_errno())))
def mnt(src, tgt, fstype, flags, data=None):
    if libc.mount(src.encode() if src else None, tgt.encode(), fstype.encode() if fstype else None,
                  ctypes.c_ulong(flags), data.encode() if data else None) != 0:
        die("mount %s -> %s" % (src, tgt))
class MountAttr(ctypes.Structure):
    _fields_ = [("attr_set", ctypes.c_uint64), ("attr_clr", ctypes.c_uint64),
                ("propagation", ctypes.c_uint64), ("userns_fd", ctypes.c_uint64)]
NR = {"x86_64": (442, 155), "aarch64": (442, 41)}[platform.machine()]  # mount_setattr, pivot_root
def make_ro(tree):
    # One recursive mount_setattr: RDONLY|NOSUID|NODEV added on every mount of the tree.
    # Only ADDS flags, so it is allowed on locked mounts inherited into the userns.
    attr = MountAttr(0x1 | 0x2 | 0x4, 0, 0, 0)
    if libc.syscall(NR[0], -100, tree.encode(), 0x8000, ctypes.byref(attr), ctypes.sizeof(attr)) == 0:
        return
    # Fallback (kernel < 5.12): per-mount remount, replaying each mount's own flags.
    FLAG = {"nosuid": MS_NOSUID, "nodev": MS_NODEV, "noexec": MS_NOEXEC, "relatime": MS_RELATIME,
            "strictatime": MS_STRICTATIME, "noatime": MS_NOATIME, "nodiratime": MS_NODIRATIME}
    for line in open("/proc/self/mountinfo"):
        f = line.split()
        tgt = f[4].replace("\\040", " ")
        if tgt == tree or tgt.startswith(tree + "/"):
            fl = MS_REMOUNT | MS_BIND | MS_RDONLY | MS_NOSUID | MS_NODEV
            for o in f[5].split(","):
                fl |= FLAG.get(o, 0)
            libc.mount(None, tgt.encode(), None, ctypes.c_ulong(fl), None)
run_dir, script = sys.argv[1], sys.argv[2]
binds = sys.argv[3:]
newroot = os.path.join(run_dir, ".newroot")
os.makedirs(newroot, exist_ok=True)
mnt("none", newroot, "tmpfs", MS_NOSUID | MS_NODEV, "mode=755")
for p in ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"] + binds:
    if not os.path.exists(p) or p == "/" or p.startswith(("/proc", "/dev", "/sys", "/tmp", "/run")):
        continue
    d = newroot + p
    if os.path.isdir(p):
        os.makedirs(d, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(d), exist_ok=True); open(d, "a").close()
    mnt(p, d, None, MS_BIND | MS_REC)   # recursive: brings locked submounts (nvidia files, /etc/hosts) along
    make_ro(d)
for sub in ("tmp", "dev/shm", "proc", "work", ".old"):
    os.makedirs(os.path.join(newroot, sub), exist_ok=True)
mnt("none", newroot + "/tmp", "tmpfs", MS_NOSUID | MS_NODEV, "mode=1777,size=1g")
mnt("none", newroot + "/work", "tmpfs", MS_NOSUID | MS_NODEV, "mode=755,size=1g")
mnt("none", newroot + "/dev/shm", "tmpfs", MS_NOSUID | MS_NODEV, "mode=1777,size=1g")
for dv in ("null", "zero", "random", "urandom"):
    open(newroot + "/dev/" + dv, "a").close()
    mnt("/dev/" + dv, newroot + "/dev/" + dv, None, MS_BIND)
# Fresh procfs needs the new PID namespace (we have it) and is refused when the
# host /proc carries locked over-mounts (k8s maskedPaths); then there is simply no
# /proc. NEVER bind the host /proc: /proc/<pid>/root escapes the mount namespace.
libc.mount(b"proc", (newroot + "/proc").encode(), b"proc", ctypes.c_ulong(MS_NOSUID | MS_NODEV | MS_NOEXEC), None)
target = "/work/" + os.path.basename(script)
shutil.copyfile(script, newroot + target)
os.chdir(newroot)
if libc.syscall(NR[1], b".", b".old") != 0:
    die("pivot_root")
os.chdir("/")
if libc.umount2(b"/.old", 2) != 0:   # MNT_DETACH
    die("detaching old root")
os.rmdir("/.old")
os.chdir("/work")
os.environ["HOME"] = "/work"; os.environ["TMPDIR"] = "/tmp"
sys.argv = [target]
sys.path[0] = "/work"
try:
    runpy.run_path(target, run_name="__main__")
except SystemExit:
    raise
except BaseException:
    # Show the model only the frames of ITS script, not this bootstrap or runpy.
    import traceback
    et, ev, tb = sys.exc_info()
    t = tb
    while t is not None and t.tb_frame.f_code.co_filename != target:
        t = t.tb_next
    traceback.print_exception(et, ev, t)  # t is None for SyntaxError: file/line/caret still printed
    sys.exit(1)
"""

    @staticmethod
    def _python_ro_binds() -> list[str]:
        """Return Python installation directories required as read-only mounts."""
        import site
        cands = [sys.base_prefix, sys.prefix, sys.exec_prefix, sys.base_exec_prefix]
        try:
            cands.append(site.getusersitepackages())
        except Exception:
            pass
        out: list[str] = []
        for c in cands:
            c = os.path.realpath(c)
            if not os.path.isdir(c) or c == "/":
                continue
            if any(c == b or c.startswith(b + "/") for b in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc")):
                continue
            if c not in out:
                out.append(c)
        return out

    def _sandbox_cmd(self, cmd: list[str], run_dir: str) -> list[str]:
        """Wrap an interpreter command in the namespace sandbox."""
        script = cmd[-1]
        return self._UNSHARE + [sys.executable, "-c", self._SANDBOX_BOOTSTRAP, run_dir, script, *self._python_ro_binds()]

    _SANDBOX_PROBE = r"""
import os, socket, sys
bad = []
try:
    socket.getaddrinfo('example.com', 443); bad.append('DNS_OK')
except Exception: pass
try:
    socket.create_connection(('1.1.1.1', 53), timeout=3).close(); bad.append('EGRESS_OK')
except Exception: pass
try:
    open('/usr/.sandbox_write_probe', 'w'); bad.append('USR_WRITABLE')
except Exception: pass
for m in ['/usr', '/lib', '/lib64', '/bin', '/sbin', '/etc'] + %(allowed)r:
    if os.path.exists(m) and not (os.statvfs(m).f_flag & os.ST_RDONLY): bad.append('RW_MOUNT:' + m)
allowed = [a.rstrip('/') for a in %(allowed)r]
def _allowed(path):
    return any(path == a or path.startswith(a + '/') for a in allowed)
for p in %(hidden)r:
    if os.path.exists(p) and not _allowed(p): bad.append('VISIBLE:' + p)
for root in ('/mnt', '/shared', '/home', '/root', '/srv', '/data', '/opt'):
    for dp, dn, fn in os.walk(root):
        if _allowed(dp): dn[:] = []; continue   # the read-only Python prefix binds are expected
        dn[:] = [d for d in dn if not _allowed(os.path.join(dp, d))]
        if fn: bad.append('FILES_UNDER:' + dp); break
try:
    open('/work/.probe', 'w').write('x')
except Exception as e: bad.append('WORK_NOT_WRITABLE:' + type(e).__name__)
if os.path.exists('/proc/1/root'):
    try:
        if not os.path.samestat(os.stat('/proc/1/root'), os.stat('/')): bad.append('HOST_PROC_VISIBLE')
    except Exception as e: bad.append('PROC_CHECK:' + type(e).__name__)
print('SANDBOX_FAIL ' + ' '.join(bad) if bad else 'SANDBOX_OK proc=' + ('mounted' if os.path.exists('/proc/self') else 'none') + ' root=' + ','.join(sorted(os.listdir('/'))))
"""

    @classmethod
    def assert_sandboxed(cls, hidden_paths: list[str] | None = None, timeout: float = 30.0) -> str:
        """Verify network and filesystem isolation on this host; fail closed."""
        import shutil as _sh
        hidden = list(hidden_paths or [])
        run_dir = tempfile.mkdtemp(prefix="sandbox_probe_")
        try:
            probe = Path(run_dir) / "probe.py"
            probe.write_text(cls._SANDBOX_PROBE % {"hidden": hidden, "allowed": cls._python_ro_binds()}, encoding="utf-8")
            self_like = cls.__new__(cls)
            cmd = self_like._sandbox_cmd([sys.executable, str(probe)], run_dir)
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=run_dir)
            except FileNotFoundError as e:
                raise InterpreterError(f"python sandbox unavailable: {e}; refusing to run the python tool unsandboxed") from e
            if out.returncode != 0 or "SANDBOX_OK" not in out.stdout:
                raise InterpreterError(
                    "python sandbox self-test FAILED (rc=%d). stdout=%r stderr=%r. If this says 'Operation not "
                    "permitted', unprivileged user namespaces are disabled on this host; refusing to run the python "
                    "tool unsandboxed" % (out.returncode, out.stdout.strip()[:400], out.stderr.strip()[:400])
                )
            return out.stdout.strip()
        finally:
            _sh.rmtree(run_dir, ignore_errors=True)

    assert_network_blocked = assert_sandboxed

    def __init__(
        self,
        require_confirm: bool = True,
        print_stdout: bool = False,
        print_stderr: bool = True,
        execution_timeout: int = 60,
        memory_limit_mb: int = 4096,
        network_isolation: bool = True,
    ) -> None:
        self.require_confirm = require_confirm
        self.print_stdout = print_stdout
        self.print_stderr = print_stderr
        self.execution_timeout = execution_timeout
        self.memory_limit_mb = memory_limit_mb

        self.network_isolation = network_isolation

    def _child_env(self, home: str) -> dict[str, str]:
        """Minimal env for the child: no parent secrets, single-threaded BLAS."""
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": home,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }

        try:
            import site
            usersite = site.getusersitepackages()
            if self.network_isolation and os.path.isdir(usersite):
                env["PYTHONPATH"] = usersite
        except Exception:
            pass
        return env

    def run_file(
        self,
        file: Path,
        code_type: str = "python",
    ) -> str:
        """Executes a code file in a subprocess and captures its output."""
        if not file.is_file():
            return f"{file} is not a file."
        code_type = self._check_code_type(code_type)
        temp_file = None
        if self._CODE_TYPE_MAPPING[code_type] == "python":
            with open(file, encoding="utf-8") as f:
                source = f.read()

            try:
                tree = ast.parse(source)
                if not tree.body:
                    raise InterpreterError(
                        "Python code must contain at least one executable statement; comments and whitespace alone are not executable."
                    )

                last_node = tree.body[-1]
                if isinstance(last_node, ast.Expr):
                    if not (
                        isinstance(last_node.value, ast.Call)
                        and isinstance(last_node.value.func, ast.Name)
                        and last_node.value.func.id == "print"
                    ):
                        tree.body[-1] = ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id="print", ctx=ast.Load()),
                                args=[
                                    ast.Call(
                                        func=ast.Name(id="repr", ctx=ast.Load()),
                                        args=[last_node.value],
                                        keywords=[],
                                    )
                                ],
                                keywords=[],
                            )
                        )

                ast.fix_missing_locations(tree)
                modified_source = ast.unparse(tree)

                if self.memory_limit_mb > 0:
                    cap = self.memory_limit_mb * 1024 * 1024
                    modified_source = self._RLIMIT_BOOTSTRAP.format(b=cap) + modified_source

                temp_file = self._create_temp_file(modified_source, "py")
                cmd = [sys.executable, str(temp_file)]
            except (SyntaxError, TypeError, ValueError) as e:
                logger.warning("Failed to parse Python code with AST: %s", e)
                cmd = [sys.executable, str(file)]
        else:
            platform_type = "posix" if os.name != "nt" else "nt"
            cmd_template = self._CODE_EXECUTE_CMD_MAPPING[code_type][platform_type]
            base_cmd = cmd_template.split()[0]

            if not self._is_command_available(base_cmd):
                raise InterpreterError(
                    f"Command '{base_cmd}' not found. Please ensure it is installed and available in your PATH."
                )

            cmd = [base_cmd, str(file)]

        run_dir = str((temp_file or file).parent)
        env = self._child_env(home=run_dir)
        if self.network_isolation:
            cmd = self._sandbox_cmd(cmd, run_dir)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=run_dir,
                start_new_session=True,
                shell=False,
            )

            stdout, stderr = proc.communicate(timeout=self.execution_timeout)
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            stdout, stderr = proc.communicate()
            return_code = proc.returncode
            timeout_msg = f"Process timed out after {self.execution_timeout} seconds and was terminated."
            stderr = f"{stderr}\n{timeout_msg}"

        if temp_file is not None:
            try:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except PermissionError:
                        logger.warning("Could not delete temp file %s (may be locked)", temp_file)
            except Exception as e:
                logger.warning("Failed to cleanup temporary file: %s", e)

        if self.print_stdout and stdout:
            print("======stdout======")
            print(stdout)
            print("==================")
        if self.print_stderr and stderr:
            print("======stderr======")
            print(stderr)
            print("==================")

        exec_result = ""
        if stdout:
            exec_result += stdout
        if stderr:
            exec_result += f"(stderr: {stderr})"
        if return_code != 0:
            error_msg = f"(Execution failed with return code {return_code})"
            if not stderr:
                exec_result += error_msg
            elif error_msg not in stderr:
                exec_result += error_msg
        return exec_result

    def run(
        self,
        code: str,
        code_type: str,
    ) -> str:
        """Run code from a temporary file and remove it afterward."""
        code_type = self._check_code_type(code_type)

        if self.require_confirm:
            self._confirm_execution(
                message=(f"The following {code_type} code will run on your computer: {code}"),
                prompt="Running code? [y/N]:",
                declined_message=(
                    "Execution halted: User opted not to run the code. "
                    "This choice stops the current operation and any "
                    "further code execution."
                ),
            )

        temp_file_path = None
        temp_dir = None
        try:
            temp_file_path = self._create_temp_file(code=code, extension=self._CODE_EXTENSION_MAPPING[code_type])
            temp_dir = temp_file_path.parent
            return self.run_file(temp_file_path, code_type)
        finally:
            try:
                if temp_file_path and temp_file_path.exists():
                    try:
                        temp_file_path.unlink()
                    except PermissionError:
                        logger.warning("Could not delete temp file %s", temp_file_path)

                if temp_dir and temp_dir.exists():
                    try:
                        import shutil

                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception as e:
                        logger.warning("Could not delete temp directory: %s", e)
            except Exception as e:
                logger.warning("Error during cleanup: %s", e)

    def _create_temp_file(self, code: str, extension: str) -> Path:
        """Creates a temporary file with the given code and extension."""
        try:
            temp_dir = tempfile.mkdtemp()

            file_path = Path(temp_dir) / f"temp_code.{extension}"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)

            return file_path
        except Exception as e:
            if "temp_dir" in locals():
                try:
                    import shutil

                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass
            logger.error("Failed to create temporary file: %s", e)
            raise

    def _check_code_type(self, code_type: str) -> str:
        if code_type not in self._CODE_TYPE_MAPPING:
            raise InterpreterError(
                f"Unsupported code type {code_type}. Currently "
                f"`{self.__class__.__name__}` only supports "
                f"{', '.join(self._CODE_EXTENSION_MAPPING.keys())}."
            )
        return self._CODE_TYPE_MAPPING[code_type]

    def _confirm_execution(self, message: str, prompt: str, declined_message: str) -> None:
        r"""Prompt the user before executing local subprocess work."""
        logger.info(message)
        while True:
            choice = input(prompt).lower().strip()
            if choice in ["y", "yes", "ye"]:
                return
            if choice in ["no", "n", ""]:
                raise InterpreterError(declined_message)
            print("Please enter 'y' or 'n'.")

    def supported_code_types(self) -> list[str]:
        r"""Provides supported code types by the interpreter."""
        return list(self._CODE_EXTENSION_MAPPING.keys())

    def update_action_space(self, action_space: dict[str, Any]) -> None:
        r"""Updates action space for *python* interpreter."""
        raise RuntimeError("SubprocessInterpreter doesn't support `action_space`.")

    def _is_command_available(self, command: str) -> bool:
        """Check if a command is available in the system PATH."""
        locator = "where" if os.name == "nt" else "which"
        try:
            with open(os.devnull, "w") as devnull:
                subprocess.check_call(
                    [locator, command],
                    stdout=devnull,
                    stderr=devnull,
                    shell=False,
                )
            return True
        except subprocess.CalledProcessError:
            return False

    def execute_command(self, command: str) -> tuple[str, str]:
        """Reject shell commands; use `run(code, "python")` instead."""
        raise InterpreterError("Shell execution is disabled in the forecast interpreter; use `run(code, 'python')`.")
