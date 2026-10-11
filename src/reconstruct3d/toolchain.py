from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess

from .config import QualityConfig


OPENMVS_PROGRAMS = (
    "InterfaceCOLMAP",
    "DensifyPointCloud",
    "ReconstructMesh",
    "RefineMesh",
    "TextureMesh",
    "TransformScene",
)


@dataclass(frozen=True)
class Toolchain:
    colmap: str | None
    openmvs: dict[str, str | None]
    ffmpeg: str | None
    ffprobe: str | None
    blender: str | None

    def missing_for(self, quality: QualityConfig, stop_after: str | None = None) -> list[str]:
        missing = []
        if stop_after != "keyframes" and not self.colmap:
            missing.append("COLMAP command line program (colmap)")
        if stop_after in {None, "dense"}:
            names = ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh", "TransformScene"]
            if quality.refine_mesh:
                names.append("RefineMesh")
            missing.extend(f"OpenMVS {name}" for name in names if not self.openmvs.get(name))
        return missing

    def to_dict(self) -> dict:
        return {
            "colmap": self.colmap,
            "openmvs": self.openmvs,
            "ffmpeg": self.ffmpeg,
            "ffprobe": self.ffprobe,
            "blender": self.blender,
        }


def discover_toolchain() -> Toolchain:
    colmap = _configured_program("COLMAP_BIN", "colmap")
    configured_openmvs = os.environ.get("OPENMVS_BIN")
    openmvs = {}
    for name in OPENMVS_PROGRAMS:
        if configured_openmvs:
            candidate = Path(configured_openmvs).expanduser()
            if candidate.is_dir():
                program = candidate / name
            elif candidate.is_file() and candidate.name == name:
                program = candidate
            else:
                program = None
            path = str(program.resolve()) if program and program.is_file() and os.access(program, os.X_OK) else None
        else:
            path = shutil.which(name)
        openmvs[name] = path
    return Toolchain(
        colmap=colmap,
        openmvs=openmvs,
        ffmpeg=shutil.which("ffmpeg"),
        ffprobe=shutil.which("ffprobe"),
        blender=_configured_program("BLENDER_BIN", "blender"),
    )


def _configured_program(environment_name: str, default_name: str) -> str | None:
    configured = os.environ.get(environment_name)
    if not configured:
        return shutil.which(default_name)
    path = Path(configured).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return str(path.resolve())
    return shutil.which(configured)


def validate_startup(
    toolchain: Toolchain,
    quality: QualityConfig,
    stop_after: str | None = None,
) -> list[dict]:
    missing = toolchain.missing_for(quality, stop_after)
    if missing:
        details = "\n".join(f"  - {name}" for name in missing)
        raise RuntimeError(
            "The requested reconstruction stages need native tools that are not installed:\n"
            f"{details}\nSee docs/software-setup.md for the CPU-only setup instructions."
        )
    commands = []
    if stop_after != "keyframes":
        commands.append((toolchain.colmap, [toolchain.colmap, "-h"], "COLMAP"))
    if stop_after in {None, "dense"}:
        names = ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh", "TransformScene"]
        if quality.refine_mesh:
            names.append("RefineMesh")
        commands.extend(
            (toolchain.openmvs[name], [toolchain.openmvs[name], "-h"], f"OpenMVS {name}")
            for name in names
        )

    checks = []
    for executable, command, label in commands:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"{label} was found but did not start: {exc}") from exc
        output = result.stdout + "\n" + result.stderr
        openmvs_help_exit = (
            label.startswith("OpenMVS ")
            and result.returncode == 1
            and "OpenMVS" in output
            and ("Available options" in output or "Generic options" in output)
        )
        if result.returncode and not openmvs_help_exit:
            detail = (result.stderr or result.stdout).strip().splitlines()
            suffix = f" Output: {detail[-1][:300]}" if detail else ""
            raise RuntimeError(
                f"{label} failed its startup check (exit code {result.returncode}).{suffix}"
            )
        banner = next(
            (line.strip() for line in output.splitlines() if line.strip()),
            None,
        )
        checks.append({
            "program": label,
            "path": executable,
            "started": True,
            "banner": banner,
            "returncode": result.returncode,
            "expected_usage_exit": openmvs_help_exit,
        })
    return checks
