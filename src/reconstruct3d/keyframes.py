import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import KeyframeConfig
from .video import probe_video


@dataclass(frozen=True)
class CandidateMetrics:
    blur_variance: float
    clipped_fraction: float
    good_matches: int
    inliers: int
    inlier_ratio: float
    median_motion_px: float


@dataclass(frozen=True)
class SelectedFrame:
    path: Path
    timestamp_seconds: float
    reason: str
    metrics: CandidateMetrics


@dataclass(frozen=True)
class KeyframeReport:
    selected: list[SelectedFrame]
    candidate_count: int
    warnings: list[str]


@dataclass(frozen=True)
class CandidateRecord:
    candidate_index: int
    timestamp_seconds: float
    selected: bool
    reason: str
    output_path: str
    metrics: CandidateMetrics


def classify_candidate(
    metrics: CandidateMetrics,
    gap_seconds: float,
    is_first: bool,
    config: KeyframeConfig,
    adjacent_metrics: CandidateMetrics | None = None,
) -> tuple[bool, str]:
    if metrics.blur_variance < config.min_blur_variance:
        return False, "blurred"
    if metrics.clipped_fraction > config.max_clipped_fraction:
        return False, "exposure_clipped"
    if is_first:
        return True, "first_usable"
    if gap_seconds < config.min_gap_seconds:
        return False, "too_soon"
    adjacent_overlap = adjacent_metrics is not None and (
        adjacent_metrics.good_matches >= config.min_matches
        and adjacent_metrics.inliers >= config.min_inliers
        and adjacent_metrics.inlier_ratio >= config.min_inlier_ratio
        and adjacent_metrics.median_motion_px <= config.max_motion_px
    )
    can_relink = gap_seconds >= config.max_gap_seconds and adjacent_overlap
    if metrics.median_motion_px > config.max_motion_px:
        if can_relink:
            return True, "max_gap_relinked"
        return False, "fast_motion"
    has_overlap = (
        metrics.good_matches >= config.min_matches
        and metrics.inliers >= config.min_inliers
        and metrics.inlier_ratio >= config.min_inlier_ratio
    )
    if not has_overlap:
        if can_relink:
            return True, "max_gap_relinked"
        if adjacent_metrics is None and gap_seconds >= config.max_gap_seconds and metrics.good_matches > 0:
            return True, "max_gap_low_overlap"
        return False, "low_overlap"
    if metrics.median_motion_px < config.min_motion_px:
        return False, "too_similar"
    return True, "visual_change"


def extract_keyframes(path: Path, output_dir: Path, config: KeyframeConfig) -> KeyframeReport:
    path = path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = probe_video(path)
    if metadata.fps <= 0:
        raise ValueError(
            f"Could not determine a usable frame rate for {path}. Install ffprobe or use a video with frame-rate metadata."
        )

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not decode video: {path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    detector = cv2.ORB_create(nfeatures=config.orb_features, fastThreshold=8)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    sample_every = max(1, int(round(metadata.fps / config.sample_fps)))
    selected: list[SelectedFrame] = []
    records: list[CandidateRecord] = []
    warnings: list[str] = []
    previous_keypoints = None
    previous_descriptors = None
    previous_candidate_keypoints = None
    previous_candidate_descriptors = None
    frame_index = -1
    candidate_index = 0
    decoded = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            decoded += 1
            if frame_index % sample_every:
                continue

            timestamp = _frame_timestamp(capture, frame_index, metadata.fps)
            gray = _scoring_image(frame, config.scoring_width)
            blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            clipped = np.logical_or(gray <= 5, gray >= 250)
            clipped_fraction = float(np.count_nonzero(clipped) / clipped.size)
            keypoints, descriptors = detector.detectAndCompute(gray, None)

            if previous_descriptors is None or descriptors is None:
                good_matches = inliers = 0
                inlier_ratio = median_motion = 0.0
            else:
                good_matches, inliers, inlier_ratio, median_motion = _match_views(
                    previous_keypoints,
                    previous_descriptors,
                    keypoints,
                    descriptors,
                    matcher,
                    config.ransac_threshold_px,
                )
            metrics = CandidateMetrics(
                blur_variance=blur_variance,
                clipped_fraction=clipped_fraction,
                good_matches=good_matches,
                inliers=inliers,
                inlier_ratio=inlier_ratio,
                median_motion_px=median_motion,
            )
            if previous_candidate_descriptors is None or descriptors is None:
                adjacent_metrics = CandidateMetrics(blur_variance, clipped_fraction, 0, 0, 0.0, 0.0)
            else:
                adjacent_matches, adjacent_inliers, adjacent_ratio, adjacent_motion = _match_views(
                    previous_candidate_keypoints,
                    previous_candidate_descriptors,
                    keypoints,
                    descriptors,
                    matcher,
                    config.ransac_threshold_px,
                )
                adjacent_metrics = CandidateMetrics(
                    blur_variance,
                    clipped_fraction,
                    adjacent_matches,
                    adjacent_inliers,
                    adjacent_ratio,
                    adjacent_motion,
                )
            gap = timestamp - selected[-1].timestamp_seconds if selected else 0.0
            keep, reason = classify_candidate(metrics, gap, not selected, config, adjacent_metrics)
            output_path = ""

            if keep and len(selected) < config.max_keyframes:
                filename = f"frame_{len(selected) + 1:04d}_{round(timestamp * 1_000_000):012d}us.jpg"
                frame_path = output_dir / filename
                written = cv2.imwrite(
                    str(frame_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
                )
                if not written:
                    raise OSError(f"Could not write selected keyframe: {frame_path}")
                selected.append(SelectedFrame(frame_path, timestamp, reason, metrics))
                output_path = filename
                previous_keypoints = keypoints
                previous_descriptors = descriptors
                if reason in {"max_gap_low_overlap", "max_gap_relinked"}:
                    warnings.append(
                        f"Frame at {timestamp:.2f}s was kept to bridge a long gap using adjacent-frame overlap."
                    )
            elif keep:
                reason = "keyframe_limit"

            records.append(
                CandidateRecord(candidate_index, timestamp, bool(output_path), reason, output_path, metrics)
            )
            if (
                descriptors is not None
                and blur_variance >= config.min_blur_variance
                and clipped_fraction <= config.max_clipped_fraction
            ):
                previous_candidate_keypoints = keypoints
                previous_candidate_descriptors = descriptors
            candidate_index += 1
    finally:
        capture.release()

    if not selected:
        raise ValueError(
            "No usable keyframes were selected. The video may be blurred, severely over/underexposed, or featureless."
        )
    if len(selected) < 6:
        warnings.append(
            f"Only {len(selected)} keyframes were selected; sparse reconstruction may not be reliable."
        )
    _write_diagnostics(output_dir, records, selected, decoded, metadata)
    return KeyframeReport(selected, candidate_index, warnings)


def _scoring_image(frame: np.ndarray, width: int) -> np.ndarray:
    height, source_width = frame.shape[:2]
    if source_width <= width:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    target_height = max(1, int(round(height * width / source_width)))
    resized = cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _frame_timestamp(capture: cv2.VideoCapture, frame_index: int, fps: float) -> float:
    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    if math.isfinite(timestamp) and timestamp > 0:
        return timestamp
    return frame_index / fps


def _match_views(
    previous_keypoints,
    previous_descriptors: np.ndarray,
    keypoints,
    descriptors: np.ndarray,
    matcher: cv2.BFMatcher,
    ransac_threshold: float,
) -> tuple[int, int, float, float]:
    if not previous_keypoints or not keypoints or descriptors is None:
        return 0, 0, 0.0, 0.0
    try:
        pairs = matcher.knnMatch(previous_descriptors, descriptors, k=2)
    except cv2.error:
        return 0, 0, 0.0, 0.0
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance]
    if len(good) < 3:
        return len(good), 0, 0.0, 0.0

    source = np.float32([previous_keypoints[m.queryIdx].pt for m in good])
    target = np.float32([keypoints[m.trainIdx].pt for m in good])
    displacement = np.linalg.norm(target - source, axis=1)
    inlier_mask = None
    if len(good) >= 8:
        try:
            _, inlier_mask = cv2.findFundamentalMat(
                source, target, cv2.FM_RANSAC, ransac_threshold, 0.99
            )
        except cv2.error:
            inlier_mask = None
    if inlier_mask is None:
        try:
            _, inlier_mask = cv2.estimateAffinePartial2D(
                source, target,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=500,
                confidence=0.99,
            )
        except cv2.error:
            inlier_mask = None

    if inlier_mask is None:
        count = 0
        motion = float(np.median(displacement)) if displacement.size else 0.0
    else:
        inlier_mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
        count = int(np.count_nonzero(inlier_mask))
        motion = float(np.median(displacement[inlier_mask])) if count else 0.0
    return len(good), count, count / len(good), motion


def _write_diagnostics(
    output_dir: Path,
    records: list[CandidateRecord],
    selected: list[SelectedFrame],
    decoded_frames: int,
    metadata,
) -> None:
    csv_path = output_dir / "keyframes.csv"
    fields = [
        "candidate_index", "timestamp_seconds", "selected", "reason", "output_path",
        "blur_variance", "clipped_fraction", "good_matches", "inliers",
        "inlier_ratio", "median_motion_px",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "candidate_index": record.candidate_index,
                "timestamp_seconds": f"{record.timestamp_seconds:.6f}",
                "selected": int(record.selected),
                "reason": record.reason,
                "output_path": record.output_path,
                **asdict(record.metrics),
            }
            writer.writerow(row)

    _write_contact_sheet(output_dir, selected)
    report = {
        "source_video": metadata.path,
        "decoded_frames": decoded_frames,
        "candidate_frames": len(records),
        "selected_frames": len(selected),
        "selection": [
            {
                "file": frame.path.name,
                "timestamp_seconds": frame.timestamp_seconds,
                "reason": frame.reason,
                "metrics": asdict(frame.metrics),
            }
            for frame in selected
        ],
        "warnings": [
            f"{sum(record.reason == reason for record in records)} candidate(s) {reason}."
            for reason in (
                "blurred", "exposure_clipped", "low_overlap", "fast_motion",
                "too_similar", "max_gap_low_overlap",
            )
            if any(record.reason == reason for record in records)
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _write_contact_sheet(output_dir: Path, selected: list[SelectedFrame]) -> None:
    columns, cell_width, cell_height, label_height = 4, 320, 220, 34
    rows = math.ceil(len(selected) / columns)
    sheet = np.full((rows * (cell_height + label_height), columns * cell_width, 3), 32, dtype=np.uint8)
    for index, item in enumerate(selected):
        image = cv2.imread(str(item.path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        thumb = cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        x = (index % columns) * cell_width
        y = (index // columns) * (cell_height + label_height)
        sheet[y:y + cell_height, x:x + cell_width] = thumb
        label = f"{item.timestamp_seconds:7.2f}s  {item.reason}"
        cv2.putText(sheet, label, (x + 7, y + cell_height + 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (240, 240, 240), 1, cv2.LINE_AA)
    path = output_dir / "contact_sheet.jpg"
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(f"Could not write keyframe contact sheet: {path}")
