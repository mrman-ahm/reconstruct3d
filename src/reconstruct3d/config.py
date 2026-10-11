from dataclasses import dataclass


@dataclass(frozen=True)
class KeyframeConfig:
    sample_fps: float = 4.0
    min_blur_variance: float = 15.0
    max_clipped_fraction: float = 0.65
    min_gap_seconds: float = 0.2
    max_gap_seconds: float = 0.9
    min_matches: int = 8
    min_inliers: int = 6
    min_inlier_ratio: float = 0.2
    min_motion_px: float = 8.0
    max_motion_px: float = 100.0
    max_keyframes: int = 160
    scoring_width: int = 640
    orb_features: int = 1200
    ransac_threshold_px: float = 2.5
    jpeg_quality: int = 94


@dataclass(frozen=True)
class QualityConfig:
    name: str
    max_image_size: int
    max_keyframes: int
    cpu_threads: int
    openmvs_resolution_level: int
    refine_mesh: bool
    camera_model: str = "SIMPLE_RADIAL"

    @property
    def keyframes(self) -> KeyframeConfig:
        return KeyframeConfig(max_keyframes=self.max_keyframes)


QUALITY_PRESETS = {
    "preview": QualityConfig("preview", 960, 60, 2, 1, False),
    "balanced": QualityConfig("balanced", 1440, 160, 3, 1, False),
    "high": QualityConfig("high", 1920, 280, 4, 1, True),
}


def quality_config(name: str, *, threads: int | None = None) -> QualityConfig:
    try:
        config = QUALITY_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(QUALITY_PRESETS)
        raise ValueError(f"Unknown quality '{name}'. Choose one of: {choices}.") from exc
    if threads is not None:
        if threads < 1:
            raise ValueError("threads must be at least 1")
        return QualityConfig(
            name=config.name,
            max_image_size=config.max_image_size,
            max_keyframes=config.max_keyframes,
            cpu_threads=threads,
            openmvs_resolution_level=config.openmvs_resolution_level,
            refine_mesh=config.refine_mesh,
            camera_model=config.camera_model,
        )
    return config
