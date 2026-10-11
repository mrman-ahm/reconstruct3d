import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    frame_count: int | None
    codec: str | None = None
    rotation_degrees: int = 0
    camera_make: str | None = None
    camera_model: str | None = None
    focal_length_mm: float | None = None
    metadata_source: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


def metadata_from_ffprobe(path: Path, payload: dict) -> VideoMetadata:
    streams = payload.get("streams") or []
    stream = next((item for item in streams if item.get("codec_type", "video") == "video"), None)
    if stream is None:
        raise ValueError(f"ffprobe found no video stream in {path}")

    fmt = payload.get("format") or {}
    stream_tags = stream.get("tags") or {}
    format_tags = fmt.get("tags") or {}
    side_data = stream.get("side_data_list") or []
    rotation = _rotation_degrees(stream_tags, side_data)

    width = _to_int(stream.get("width"), 0)
    height = _to_int(stream.get("height"), 0)
    if rotation % 180:
        width, height = height, width

    fps = _parse_rate(stream.get("avg_frame_rate"))
    if fps <= 0:
        fps = _parse_rate(stream.get("r_frame_rate"))
    frame_count = _to_optional_int(stream.get("nb_frames"))
    duration = _first_positive(
        stream.get("duration"), fmt.get("duration"),
        (frame_count / fps) if frame_count and fps > 0 else None,
    )

    tags = {_normalize_tag(k): str(v).strip() for d in (stream_tags, format_tags) for k, v in d.items()}
    make = _first_tag(tags, "make", "manufacturer", "comapplequicktimemake")
    model = _first_tag(tags, "model", "comapplequicktimemodel", "cameramodelname")
    focal_text = _first_tag(tags, "focallength", "comapplequicktimecameralensmodel")
    focal_length = _parse_focal_length(focal_text)

    return VideoMetadata(
        path=str(path.expanduser().resolve()),
        duration_seconds=duration,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        codec=stream.get("codec_name"),
        rotation_degrees=rotation,
        camera_make=make,
        camera_model=model,
        focal_length_mm=focal_length,
        metadata_source="ffprobe",
    )


def probe_video(path: Path, ffprobe_path: str | None = None) -> VideoMetadata:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Input video does not exist or is not a file: {path}")

    executable = ffprobe_path or shutil.which("ffprobe")
    if executable and Path(executable).exists():
        result = subprocess.run(
            [
                executable, "-v", "error", "-show_streams", "-show_format",
                "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            detail = result.stderr.strip() or f"exit status {result.returncode}"
            raise ValueError(f"ffprobe could not inspect {path}: {detail}")
        try:
            return metadata_from_ffprobe(path, json.loads(result.stdout))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"ffprobe returned invalid metadata for {path}: {exc}") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {path}")
    try:
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise ValueError(f"Video contains no decodable frames: {path}")
        height, width = first_frame.shape[:2]
        frame_count_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = int(round(frame_count_value)) if frame_count_value > 0 else None
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps < 0.01:
            fps = 0.0
        duration = (frame_count / fps) if frame_count and fps > 0 else 0.0
        rotation = 0
        if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            angle = capture.get(cv2.CAP_PROP_ORIENTATION_META)
            if angle is not None and angle == angle:
                rotation = int(round(angle / 90.0) * 90) % 360
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ") or None
    finally:
        capture.release()

    return VideoMetadata(
        path=str(path),
        duration_seconds=duration,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        codec=codec,
        rotation_degrees=rotation,
        metadata_source="opencv",
    )


def _parse_rate(value: object) -> float:
    if not value or str(value).strip() in {"N/A", "0/0"}:
        return 0.0
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_optional_int(value: object) -> int | None:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _first_positive(*values: object) -> float:
    for value in values:
        try:
            parsed = float(value)
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            continue
    return 0.0


def _normalize_tag(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _first_tag(tags: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = tags.get(key)
        if value:
            return value
    return None


def _rotation_degrees(tags: dict, side_data: list[dict]) -> int:
    value = None
    for item in side_data:
        if "rotation" in item:
            value = item["rotation"]
            break
    if value is None:
        value = tags.get("rotate") or tags.get("ROTATE") or 0
    try:
        return int(round(float(value) / 90.0) * 90) % 360
    except (TypeError, ValueError):
        return 0


def _parse_focal_length(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mm\b", value, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
