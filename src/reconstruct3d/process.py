from pathlib import Path
import shlex
import subprocess
from typing import Sequence


class CommandError(RuntimeError):
    pass


def run_logged_command(argv: Sequence[str], cwd: Path, log_path: Path, timeout: float | None = None) -> None:
    command = [str(value) for value in argv]
    if not command:
        raise ValueError("A reconstruction command cannot be empty")
    cwd = cwd.expanduser().resolve()
    log_path = log_path.expanduser().resolve()
    if not cwd.is_dir():
        raise CommandError(f"Command working directory does not exist: {cwd}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {shlex.join(command)}\n")
        log.write(f"WORKDIR: {cwd}\n\n")
        log.flush()
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                shell=False,
                text=True,
            )
        except FileNotFoundError as exc:
            log.write(f"\nEXECUTION ERROR: executable not found: {command[0]}\n")
            raise CommandError(
                f"Required reconstruction program was not found: {command[0]}. "
                f"See {log_path} for details."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            log.write(f"\nEXECUTION ERROR: timed out after {timeout} seconds\n")
            raise CommandError(
                f"Command timed out after {timeout} seconds: {command[0]}. See {log_path}."
            ) from exc
    if result.returncode != 0:
        tail = _tail(log_path)
        raise CommandError(
            f"Command failed with exit code {result.returncode}: {command[0]}. "
            f"See {log_path}.\n{tail}"
        )


def _tail(path: Path, line_count: int = 35) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])
