import importlib.util
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import cv2

from .toolchain import discover_toolchain


def collect_environment_report(storage_path: Path | None = None) -> dict:
    storage_path = Path(storage_path or Path.cwd()).resolve()
    cpu = _command_output(["lscpu"], timeout=4)
    cpu_info = _parse_lscpu(cpu)
    memory = _read_memory()
    graphics = _graphics_report()
    cv_build = cv2.getBuildInformation()
    ffmpeg_in_opencv = bool(re.search(r"^\s*FFMPEG:\s+YES", cv_build, re.MULTILINE))
    storage = shutil.disk_usage(storage_path)

    docker_path = shutil.which("docker")
    docker_report = {"client": docker_path, "daemon_access": False}
    if docker_path:
        docker_result = _run([docker_path, "info", "--format", "{{.ServerVersion}}"], timeout=5)
        docker_report["daemon_access"] = docker_result["returncode"] == 0
        if docker_result["returncode"] == 0:
            docker_report["server_version"] = docker_result["stdout"].strip()
        else:
            docker_report["detail"] = (docker_result["stderr"] or "daemon unavailable").strip()[:300]

    tools = discover_toolchain()
    return {
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "cpu_model": cpu_info.get("model"),
        "cpu_count": os.cpu_count(),
        "physical_cores": cpu_info.get("cores_per_socket"),
        "threads_per_core": cpu_info.get("threads_per_core"),
        "ram_total_mb": memory.get("MemTotal"),
        "ram_available_mb": memory.get("MemAvailable"),
        "storage_path": str(storage_path),
        "storage_free_gb": round(storage.free / (1024 ** 3), 2),
        "gpu": graphics,
        "opencv": {"version": cv2.__version__, "ffmpeg_decoder": ffmpeg_in_opencv},
        "executables": {
            "ffmpeg": tools.ffmpeg,
            "ffprobe": tools.ffprobe,
            "colmap": tools.colmap,
            "openmvs": tools.openmvs,
            "blender": tools.blender,
            "gcc": _version_line("gcc"),
            "clang": _version_line("clang"),
            "cmake": _version_line("cmake"),
            "ninja": _version_line("ninja"),
            "git": _version_line("git"),
        },
        "alternative_reconstruction_commands": {
            name: shutil.which(executable)
            for name, executable in {
                "openmvg": "openMVG_main_SfM",
                "opensfm": "opensfm",
                "meshroom": "meshroom",
                "meshroom_batch": "meshroom_batch",
                "alicevision_depth": "aliceVision_depthMapEstimation",
                "mve_dmrecon": "dmrecon",
                "mve_surface": "fssrecon",
            }.items()
        },
        "docker": docker_report,
        "python_packages": {
            name: importlib.util.find_spec(name) is not None
            for name in (
                "pycolmap", "open3d", "torch", "trimesh", "opensfm",
                "nerfstudio", "dust3r", "vggt", "meshroom",
            )
        },
    }


def _read_memory() -> dict[str, int]:
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            match = re.match(r"(MemTotal|MemAvailable):\s+(\d+)\s+kB", line)
            if match:
                values[match.group(1)] = round(int(match.group(2)) / 1024)
    except OSError:
        return values
    return values


def _parse_lscpu(output: str) -> dict:
    result = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = label.strip()
        if key == "Model name":
            result["model"] = value.strip()
        elif key == "Core(s) per socket":
            result["cores_per_socket"] = _int_or_none(value.strip())
        elif key == "Thread(s) per core":
            result["threads_per_core"] = _int_or_none(value.strip())
    return result


def _graphics_report() -> dict:
    pci = _command_output(["lspci", "-nn"], timeout=4)
    graphics_lines = [
        line.strip() for line in pci.splitlines()
        if re.search(r"VGA compatible controller|3D controller|Display controller", line, re.IGNORECASE)
    ]
    render_nodes = sorted(Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").is_dir() else []
    nvidia_smi = shutil.which("nvidia-smi")
    nvcc = shutil.which("nvcc")
    opencl = _command_output(["clinfo", "-l"], timeout=5)
    vulkan_result = _run(["vulkaninfo", "--summary"], timeout=5)
    vulkan_available = vulkan_result["returncode"] == 0
    return {
        "devices_seen_on_pci": graphics_lines,
        "render_nodes": [str(path) for path in render_nodes],
        "render_node_accessible": any(os.access(path, os.R_OK | os.W_OK) for path in render_nodes),
        "cuda": {"available": bool(nvidia_smi and nvcc), "nvidia_smi": nvidia_smi, "nvcc": nvcc},
        "opencl_devices": opencl.strip()[:1200] or None,
        "opencl_gpu_seen": bool(re.search(r"device\s+#\d+:.*gpu", opencl, re.IGNORECASE)),
        "vulkan": {"available": vulkan_available, "detail": (vulkan_result["stderr"] or "").strip()[:300]},
    }


def _version_line(executable: str) -> str | None:
    path = shutil.which(executable)
    if not path:
        return None
    output = _command_output([path, "--version"], timeout=4)
    return (output.splitlines()[0].strip() if output.splitlines() else path)


def _command_output(command: list[str], timeout: float) -> str:
    result = _run(command, timeout)
    return result["stdout"] if result["returncode"] == 0 else ""


def _run(command: list[str], timeout: float) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None
