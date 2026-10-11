import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import time
import unicodedata

from .config import quality_config
from .dense import DenseReport, run_dense
from .keyframes import KeyframeConfig, KeyframeReport, SelectedFrame, extract_keyframes
from .reconstruction import SparseQualityError, SparseReport, run_sparse
from .system_info import collect_environment_report
from .toolchain import discover_toolchain, validate_startup
from .video import probe_video


def safe_job_name(value: str) -> str:
    base = str(value).replace("\\", "/").rstrip("/").split("/")[-1]
    stem = Path(base).stem
    normalized = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    result = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-.").lower()
    return (result[:64].rstrip("-.") or "reconstruction")


def main(argv=None):
    parser = _make_parser()
    args = parser.parse_args(argv)
    if args.doctor:
        tools = discover_toolchain()
        try:
            startup = validate_startup(tools, quality_config("balanced"))
            startup_error = None
        except RuntimeError as exc:
            startup = []
            startup_error = str(exc)
        report = {
            "hardware": collect_environment_report(Path.cwd()),
            "toolchain": tools.to_dict(),
            "missing_balanced_pipeline_tools": tools.missing_for(quality_config("balanced")),
            "native_startup_checks": startup,
            "native_startup_error": startup_error,
        }
        print(json.dumps(report, indent=2))
        return 0
    if not args.video:
        parser.error("provide a video path or use --doctor")
    try:
        return _run_job(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reconstruct",
        description="Reconstruct a 3D scene from a prerecorded walkthrough video (CPU-capable).",
    )
    parser.add_argument("video", nargs="?", help="input video file")
    parser.add_argument("--doctor", action="store_true", help="report hardware and reconstruction tools")
    parser.add_argument("--output", type=Path, help="job directory (default: outputs/<video-name>)")
    parser.add_argument("--quality", choices=("preview", "balanced", "high"), default="balanced")
    parser.add_argument("--cpu-threads", type=_positive_int, help="override stage thread count")
    parser.add_argument("--camera-model", help="COLMAP model; default SIMPLE_RADIAL, estimated from this video")
    parser.add_argument("--sample-fps", type=_positive_float, help="candidate-frame sampling rate (default 4)")
    parser.add_argument("--min-motion-px", type=_positive_float, help="minimum visual motion between selected frames")
    parser.add_argument("--max-gap-seconds", type=_positive_float, help="maximum preferred gap between keyframes")
    parser.add_argument("--max-keyframes", type=_positive_int, help="cap selected keyframes for this job")
    parser.add_argument("--stop-after", choices=("keyframes", "sparse", "dense"), help="stop after a completed stage")
    parser.add_argument("--resume", action="store_true", help="continue a matching job without overwriting it")
    parser.add_argument("--vocab-tree", type=Path, help="COLMAP vocabulary tree for loop detection on larger videos")
    return parser


def _run_job(args) -> int:
    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"Input video does not exist or is not a file: {video}")
    quality = quality_config(args.quality, threads=args.cpu_threads)
    if args.camera_model:
        quality = replace(quality, camera_model=args.camera_model)
    keyframes = quality.keyframes
    keyframe_updates = {}
    if args.sample_fps is not None:
        keyframe_updates["sample_fps"] = args.sample_fps
    if args.min_motion_px is not None:
        keyframe_updates["min_motion_px"] = args.min_motion_px
    if args.max_gap_seconds is not None:
        keyframe_updates["max_gap_seconds"] = args.max_gap_seconds
    if args.max_keyframes is not None:
        keyframe_updates["max_keyframes"] = args.max_keyframes
    keyframes = replace(keyframes, **keyframe_updates)
    vocab_tree_path = None
    vocab_tree_signature = None
    if args.vocab_tree:
        vocab_tree_path = args.vocab_tree.expanduser().resolve()
        if not vocab_tree_path.is_file():
            raise ValueError(f"COLMAP vocabulary tree does not exist: {vocab_tree_path}")
        vocab_stat = vocab_tree_path.stat()
        vocab_tree_signature = {
            "path": str(vocab_tree_path),
            "size_bytes": vocab_stat.st_size,
            "mtime_ns": vocab_stat.st_mtime_ns,
        }
    config_snapshot = {
        "quality": asdict(quality),
        "keyframes": asdict(keyframes),
        "vocab_tree": vocab_tree_signature,
    }
    signature = _source_signature(video)
    job_dir = (args.output or Path("outputs") / safe_job_name(video.name)).expanduser().resolve()
    job_path = job_dir / "job.json"

    if job_dir.exists():
        if not args.resume:
            raise ValueError(
                f"Job directory already exists: {job_dir}. Use --resume for the same video/settings, "
                "or choose a new --output directory."
            )
        if not job_path.is_file():
            raise ValueError(f"Cannot resume because {job_path} is missing.")
        state = json.loads(job_path.read_text(encoding="utf-8"))
        if state.get("source") != signature or state.get("config") != config_snapshot:
            raise ValueError(
                "Cannot resume: the video or processing settings changed. Use a new --output directory."
            )
        if state.get("status") == "complete":
            print(f"Job already complete: {job_dir}")
            return 0
    else:
        if args.resume:
            raise ValueError(f"Cannot resume a job directory that does not exist: {job_dir}")
        job_dir.mkdir(parents=True)
        state = {
            "job_dir": str(job_dir),
            "created_utc": _utc_now(),
            "source": signature,
            "config": config_snapshot,
            "status": "queued",
            "stages": {},
        }
        _write_json(job_path, state)

    state["status"] = "running"
    state.pop("error", None)
    _write_json(job_path, state)
    try:
        video_metadata = probe_video(video)
        toolchain = discover_toolchain()
        metadata_path = job_dir / "metadata.json"
        if not metadata_path.exists():
            metadata = {
                "input_video": str(video),
                "file_size_bytes": signature["size_bytes"],
                "video": video_metadata.to_dict(),
                "pipeline": "COLMAP CPU sparse reconstruction + OpenMVS CPU dense/mesh/texture",
                "camera_model": {
                    "selected": quality.camera_model,
                    "policy": "shared across frames; focal length and radial distortion estimated by COLMAP",
                    "focal_length_metadata_mm": video_metadata.focal_length_mm,
                },
                "quality": config_snapshot,
                "environment": collect_environment_report(job_dir),
                "toolchain": toolchain.to_dict(),
            }
            _write_json(metadata_path, metadata)

        keyframe_dir = job_dir / "keyframes"
        diagnostics_dir = job_dir / "diagnostics"
        keyframe_report_path = keyframe_dir / "report.json"
        keyframe_stage = state["stages"].get("keyframes", {})
        if keyframe_stage.get("status") == "complete" and keyframe_report_path.is_file():
            frame_count = _read_keyframe_count(keyframe_report_path)
            print(f"Keyframes: reusing {frame_count} frames")
        else:
            started = _start_stage(state, job_path, "keyframes")
            result = extract_keyframes(video, keyframe_dir, keyframes)
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(keyframe_dir / "keyframes.csv", diagnostics_dir / "keyframes.csv")
            shutil.copy2(keyframe_dir / "contact_sheet.jpg", diagnostics_dir / "keyframes-contact-sheet.jpg")
            _complete_stage(state, job_path, "keyframes", started, {
                "selected_frames": len(result.selected),
                "candidate_frames": result.candidate_count,
                "warnings": result.warnings,
            })
            print(f"Keyframes: selected {len(result.selected)} from {result.candidate_count} candidates")
            for warning in result.warnings:
                print(f"Keyframe warning: {warning}")
            frame_count = len(result.selected)
        if args.stop_after == "keyframes":
            state["status"] = "partial"
            state["stopped_after"] = "keyframes"
            _write_json(job_path, state)
            print(f"Stopped after keyframe diagnostics: {job_dir}")
            return 0

        state["preflight"] = {"status": "running", "started_utc": _utc_now()}
        _write_json(job_path, state)
        preflight_started = time.monotonic()
        try:
            tool_checks = validate_startup(toolchain, quality, args.stop_after)
        except Exception as exc:
            state["preflight"] = {
                "status": "failed",
                "failed_utc": _utc_now(),
                "error": str(exc),
            }
            _write_json(job_path, state)
            raise
        state["preflight"] = {
            "status": "complete",
            "completed_utc": _utc_now(),
            "runtime_seconds": round(time.monotonic() - preflight_started, 3),
            "tools": tool_checks,
        }
        _write_json(job_path, state)
        sparse_report = _existing_sparse_report(state, job_dir) if args.resume else None
        if sparse_report:
            print(
                f"Sparse: reusing {sparse_report.registered_images}/{sparse_report.selected_images} registered images, "
                f"{sparse_report.points} points"
            )
        else:
            started = _start_stage(state, job_path, "sparse")
            try:
                sparse_report = run_sparse(
                    toolchain.colmap,
                    keyframe_dir,
                    job_dir,
                    quality,
                    camera_model=quality.camera_model,
                    vocab_tree_path=vocab_tree_path,
                )
            except SparseQualityError as exc:
                _fail_stage(state, job_path, "sparse", started, exc)
                raise
            except Exception as exc:
                _fail_stage(state, job_path, "sparse", started, exc)
                raise
            _complete_stage(state, job_path, "sparse", started, {
                "selected_images": sparse_report.selected_images,
                "registered_images": sparse_report.registered_images,
                "registered_fraction": sparse_report.registered_fraction,
                "points": sparse_report.points,
                "warnings": sparse_report.warnings,
            })
            print(
                f"Sparse: registered {sparse_report.registered_images}/{sparse_report.selected_images} images, "
                f"{sparse_report.points} points"
            )
            for warning in sparse_report.warnings:
                print(f"Sparse warning: {warning}")
        if args.stop_after == "sparse":
            state["status"] = "partial"
            state["stopped_after"] = "sparse"
            _write_json(job_path, state)
            print(f"Stopped after sparse reconstruction: {job_dir}")
            return 0

        dense_report = _existing_dense_report(state, job_dir) if args.resume else None
        if dense_report:
            print(f"Dense: reusing existing textured model {dense_report.glb_path}")
        else:
            started = _start_stage(state, job_path, "dense")
            try:
                dense_report = run_dense(
                    toolchain.colmap,
                    toolchain.openmvs,
                    job_dir / "sparse" / "images",
                    sparse_report.model_path,
                    job_dir,
                    quality,
                    resume=args.resume,
                )
            except Exception as exc:
                _fail_stage(state, job_path, "dense", started, exc)
                raise
            _complete_stage(state, job_path, "dense", started, {
                "dense_points": dense_report.dense_points,
                "mesh_vertices": dense_report.mesh_vertices,
                "mesh_triangles": dense_report.mesh_triangles,
                "glb_path": str(dense_report.glb_path),
            })
            print(
                f"Dense: {dense_report.dense_points} points; mesh {dense_report.mesh_vertices} vertices / "
                f"{dense_report.mesh_triangles} triangles"
            )
        state["status"] = "complete"
        state.pop("stopped_after", None)
        state["completed_utc"] = _utc_now()
        _write_json(job_path, state)
        print(f"Exports: {job_dir / 'exports'}")
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        state["failed_utc"] = _utc_now()
        _write_json(job_path, state)
        raise


def _source_signature(video: Path) -> dict:
    stat = video.stat()
    return {"path": str(video), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_keyframe_count(path: Path) -> int:
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["selected_frames"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"The saved keyframe report is invalid: {path}") from exc


def _existing_sparse_report(state: dict, job_dir: Path) -> SparseReport | None:
    if state["stages"].get("sparse", {}).get("status") != "complete":
        return None
    path = job_dir / "sparse" / "report.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("ready_for_dense") or not Path(data["model_path"]).is_dir():
        return None
    return SparseReport(
        selected_images=data["selected_images"],
        registered_images=data["registered_images"],
        registered_fraction=data["registered_fraction"],
        points=data["points"],
        model_path=Path(data["model_path"]),
        text_model_path=Path(data["text_model_path"]),
        sparse_ply_path=Path(data["sparse_ply_path"]),
        ready_for_dense=True,
        warnings=data.get("warnings", []),
    )


def _existing_dense_report(state: dict, job_dir: Path) -> DenseReport | None:
    if state["stages"].get("dense", {}).get("status") != "complete":
        return None
    path = job_dir / "dense" / "report.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    glb = Path(data["glb_path"])
    if not glb.is_file():
        return None
    return DenseReport(
        dense_points=data["dense_points"],
        mesh_vertices=data["mesh_vertices"],
        mesh_triangles=data["mesh_triangles"],
        point_cloud_path=Path(data["point_cloud_path"]),
        mesh_path=Path(data["mesh_path"]),
        obj_path=Path(data["obj_path"]) if data.get("obj_path") else None,
        glb_path=glb,
        texture_files=[Path(value) for value in data.get("texture_files", [])],
    )


def _start_stage(state: dict, job_path: Path, stage: str) -> float:
    started = time.monotonic()
    state["stages"][stage] = {"status": "running", "started_utc": _utc_now()}
    _write_json(job_path, state)
    return started


def _complete_stage(state: dict, job_path: Path, stage: str, started: float, result: dict) -> None:
    state["stages"][stage] = {
        "status": "complete",
        "started_utc": state["stages"][stage]["started_utc"],
        "completed_utc": _utc_now(),
        "runtime_seconds": round(time.monotonic() - started, 3),
        **result,
    }
    _write_json(job_path, state)


def _fail_stage(state: dict, job_path: Path, stage: str, started: float, error: Exception) -> None:
    state["stages"][stage] = {
        "status": "failed",
        "started_utc": state["stages"][stage]["started_utc"],
        "failed_utc": _utc_now(),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "error": str(error),
    }
    _write_json(job_path, state)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed
