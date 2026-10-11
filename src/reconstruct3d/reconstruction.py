from dataclasses import dataclass
import math
from pathlib import Path
import re
import shutil
import json
import csv

import numpy as np

from .config import QualityConfig
from .process import CommandError, run_logged_command


@dataclass(frozen=True)
class SparseAssessment:
    ready_for_dense: bool
    registered_images: int
    selected_images: int
    points: int
    registered_fraction: float
    warnings: list[str]


@dataclass(frozen=True)
class SparseReport:
    selected_images: int
    registered_images: int
    registered_fraction: float
    points: int
    model_path: Path
    text_model_path: Path
    sparse_ply_path: Path
    ready_for_dense: bool
    warnings: list[str]


def build_sparse_commands(
    colmap, images_dir, job_dir, image_count, quality, camera_model=None,
    exhaustive_limit=160, vocab_tree_path=None,
):
    database = Path(job_dir) / "sparse" / "database.db"
    models = Path(job_dir) / "sparse" / "models"
    camera_model = camera_model or quality.camera_model
    extraction = [
        colmap, "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "0",
        "--FeatureExtraction.num_threads", str(quality.cpu_threads),
        "--FeatureExtraction.max_image_size", str(quality.max_image_size),
        "--SiftExtraction.max_num_features", "8192",
    ]
    shared_matching = [
        "--database_path", str(database),
        "--FeatureMatching.use_gpu", "0",
        "--FeatureMatching.num_threads", str(quality.cpu_threads),
        "--SiftMatching.max_ratio", "0.8",
        "--FeatureMatching.max_num_matches", "16384",
    ]
    if image_count <= exhaustive_limit:
        matcher_name = "exhaustive matching"
        matching = [colmap, "exhaustive_matcher", *shared_matching]
    else:
        matcher_name = "sequential matching"
        matching = [
            colmap, "sequential_matcher", *shared_matching,
            "--SequentialMatching.overlap", "10",
            "--SequentialMatching.loop_detection", "1" if vocab_tree_path else "0",
        ]
        if vocab_tree_path:
            matching.extend(["--SequentialMatching.vocab_tree_path", str(Path(vocab_tree_path).resolve())])
    mapping = [
        colmap, "mapper",
        "--database_path", str(database),
        "--image_path", str(images_dir),
        "--output_path", str(models),
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_focal_length", "1",
    ]
    return [
        ("feature extraction", extraction),
        (matcher_name, matching),
        ("sparse mapping", mapping),
    ]


def assess_sparse(selected_images: int, registered_images: int, points: int) -> SparseAssessment:
    fraction = registered_images / selected_images if selected_images else 0.0
    warnings = []
    minimum_registered = max(6, math.ceil(selected_images * 0.45))
    if selected_images < 6 or registered_images < minimum_registered:
        warnings.append(
            f"Sparse image registration is too low ({registered_images}/{selected_images}); "
            f"at least {minimum_registered} images are required before dense reconstruction."
        )
    elif fraction < 0.7:
        warnings.append(
            f"Only {registered_images}/{selected_images} keyframes registered; camera coverage may be incomplete."
        )
    if points < 100:
        warnings.append(
            f"Sparse model has only {points} points; it is too weak to guide dense reconstruction."
        )
    elif points < 500:
        warnings.append(f"Sparse model is small ({points} points); dense geometry may be incomplete.")
    ready = selected_images >= 6 and registered_images >= minimum_registered and points >= 100
    return SparseAssessment(ready, registered_images, selected_images, points, fraction, warnings)


def read_sparse_counts(images_txt: Path, points_txt: Path) -> tuple[int, int]:
    registered = 0
    for line in Path(images_txt).read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) < 10 or tokens[0].startswith("#"):
            continue
        try:
            int(tokens[0])
            [float(value) for value in tokens[1:8]]
            int(tokens[8])
        except ValueError:
            continue
        registered += 1
    points = sum(
        1 for line in Path(points_txt).read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return registered, points


def build_dense_commands(colmap, openmvs, image_dir, sparse_model, job_dir, quality):
    root = Path(job_dir)
    dense_root = root / "dense"
    undistorted = dense_root / "colmap"
    mvs_root = dense_root / "mvs"
    scene = mvs_root / "scene.mvs"
    undistort = [
        colmap, "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_model),
        "--output_path", str(undistorted),
        "--output_type", "COLMAP",
        "--max_image_size", str(quality.max_image_size),
    ]
    import_colmap = [
        openmvs["InterfaceCOLMAP"],
        "-i", str(undistorted),
        "-o", str(scene),
        "--image-folder", str(undistorted / "images"),
    ]
    # The documented setup builds OpenMVS without CUDA.  Do not pass a device
    # selector: CUDA builds expose a version-dependent GPU flag, and -1 means
    # "best GPU" in current OpenMVS releases rather than CPU-only.
    common = ["--max-threads", str(quality.cpu_threads)]
    commands = [
        ("image undistortion", undistort),
        ("COLMAP import", import_colmap),
        (
            "dense reconstruction",
            [
                openmvs["DensifyPointCloud"], str(scene),
                "--resolution-level", str(quality.openmvs_resolution_level),
                "--max-resolution", str(quality.max_image_size), *common,
            ],
        ),
        (
            "mesh reconstruction",
            [openmvs["ReconstructMesh"], "scene_dense.mvs", *common],
        ),
    ]
    mesh_file = "scene_dense_mesh.ply"
    if quality.refine_mesh:
        commands.append(
            (
                "mesh refinement",
                [
                    openmvs["RefineMesh"], "scene_dense.mvs",
                    "--mesh-file", mesh_file,
                    "--output-file", "scene_dense_mesh_refine.ply",
                    *common,
                ],
            )
        )
        mesh_file = "scene_dense_mesh_refine.ply"
    mesh_stem = Path(mesh_file).stem
    textured_glb = f"{mesh_stem}_texture.glb"
    commands.append(
        (
            "texturing",
            [
                openmvs["TextureMesh"], "scene_dense.mvs",
                "--mesh-file", mesh_file,
                "--export-type", "glb",
                "--output-file", textured_glb,
                *common,
            ],
        )
    )
    return commands


def run_sparse(
    colmap: str,
    images_dir: Path,
    job_dir: Path,
    quality: QualityConfig,
    *,
    camera_model: str | None = None,
    exhaustive_limit: int = 160,
    vocab_tree_path: Path | None = None,
) -> SparseReport:
    images_dir = Path(images_dir).resolve()
    job_dir = Path(job_dir).resolve()
    images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file()
        and p.name.startswith("frame_")
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) < 6:
        raise ValueError(f"Sparse reconstruction needs at least 6 keyframes; found {len(images)}.")
    sparse_dir = job_dir / "sparse"
    logs_dir = job_dir / "logs" / "sparse"
    models_dir = sparse_dir / "models"
    sfm_images_dir = sparse_dir / "images"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    for stale_database in (sparse_dir / "database.db", sparse_dir / "database.db-shm", sparse_dir / "database.db-wal"):
        stale_database.unlink(missing_ok=True)
    if models_dir.exists():
        shutil.rmtree(models_dir)
    if sfm_images_dir.exists():
        shutil.rmtree(sfm_images_dir)
    sfm_images_dir.mkdir(parents=True)
    for image in images:
        shutil.copy2(image, sfm_images_dir / image.name)
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    for index, (name, argv) in enumerate(
        build_sparse_commands(
            colmap,
            sfm_images_dir,
            job_dir,
            len(images),
            quality,
            camera_model,
            exhaustive_limit=exhaustive_limit,
            vocab_tree_path=vocab_tree_path,
        )
    ):
        run_logged_command(argv, cwd=job_dir, log_path=logs_dir / f"{index:02d}-{_slug(name)}.log")

    candidates = sorted(
        p for p in models_dir.iterdir()
        if p.is_dir() and (p / "images.bin").is_file() and (p / "points3D.bin").is_file()
    )
    if not candidates:
        raise CommandError(
            f"COLMAP completed without a sparse model under {models_dir}. "
            f"Review {logs_dir} and the keyframe diagnostics."
        )

    candidate_reports = []
    for model in candidates:
        text_dir = sparse_dir / "text-candidates" / model.name
        text_dir.mkdir(parents=True, exist_ok=True)
        run_logged_command(
            [colmap, "model_converter", "--input_path", str(model), "--output_path", str(text_dir), "--output_type", "TXT"],
            cwd=job_dir,
            log_path=logs_dir / f"convert-{model.name}.log",
        )
        registered, points = read_sparse_counts(text_dir / "images.txt", text_dir / "points3D.txt")
        candidate_reports.append((registered, points, model, text_dir))
    candidate_reports.sort(key=lambda item: (item[0], item[1]), reverse=True)
    registered, points, model, text_dir = candidate_reports[0]
    final_text = sparse_dir / "text"
    if final_text.exists():
        shutil.rmtree(final_text)
    shutil.copytree(text_dir, final_text)

    sparse_ply = job_dir / "exports" / "sparse.ply"
    sparse_ply.parent.mkdir(parents=True, exist_ok=True)
    run_logged_command(
        [colmap, "model_converter", "--input_path", str(model), "--output_path", str(sparse_ply), "--output_type", "PLY"],
        cwd=job_dir,
        log_path=logs_dir / "export-sparse-ply.log",
    )
    trajectory = write_camera_trajectory(final_text / "images.txt", sparse_dir / "camera_trajectory.csv")
    assessment = assess_sparse(len(images), registered, points)
    report = SparseReport(
        selected_images=len(images),
        registered_images=registered,
        registered_fraction=assessment.registered_fraction,
        points=points,
        model_path=model,
        text_model_path=final_text,
        sparse_ply_path=sparse_ply,
        ready_for_dense=assessment.ready_for_dense,
        warnings=assessment.warnings,
    )
    payload = {
        "selected_images": report.selected_images,
        "registered_images": report.registered_images,
        "registered_fraction": report.registered_fraction,
        "points": report.points,
        "model_path": str(report.model_path),
        "text_model_path": str(report.text_model_path),
        "sparse_ply_path": str(report.sparse_ply_path),
        "trajectory_cameras": trajectory,
        "ready_for_dense": report.ready_for_dense,
        "warnings": report.warnings,
    }
    (sparse_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not report.ready_for_dense:
        raise SparseQualityError(report)
    return report


class SparseQualityError(RuntimeError):
    def __init__(self, report: SparseReport):
        self.report = report
        super().__init__("Sparse reconstruction is too weak for dense processing: " + " ".join(report.warnings))


def write_camera_trajectory(images_txt: Path, output_csv: Path) -> int:
    rows = []
    for line in images_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) < 10 or tokens[0].startswith("#"):
            continue
        try:
            image_id = int(tokens[0])
            qw, qx, qy, qz, tx, ty, tz = (float(value) for value in tokens[1:8])
            camera_id = int(tokens[8])
        except ValueError:
            continue
        name = " ".join(tokens[9:])
        rotation = _quaternion_rotation(qw, qx, qy, qz)
        center = -rotation.T @ np.array([tx, ty, tz], dtype=np.float64)
        rows.append((image_id, camera_id, name, qw, qx, qy, qz, *center.tolist(), tx, ty, tz))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "image_id", "camera_id", "image", "qw", "qx", "qy", "qz",
            "x", "y", "z", "t_x", "t_y", "t_z",
        ])
        writer.writerows(rows)
    return len(rows)


def _quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
