from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil

from .process import CommandError, run_logged_command
from .reconstruction import build_dense_commands
from .config import QualityConfig


@dataclass(frozen=True)
class DenseReport:
    dense_points: int
    mesh_vertices: int
    mesh_triangles: int
    point_cloud_path: Path
    mesh_path: Path
    obj_path: Path | None
    glb_path: Path
    texture_files: list[Path]


def read_ply_counts(path: Path) -> tuple[int, int]:
    path = Path(path)
    vertices = faces = 0
    saw_ply = False
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line == "ply":
                saw_ply = True
            elif line == "end_header":
                if not saw_ply:
                    raise ValueError(f"Not a PLY file: {path}")
                return vertices, faces
            else:
                match = re.fullmatch(r"element\s+(vertex|face)\s+(\d+)", line)
                if match:
                    count = int(match.group(2))
                    if match.group(1) == "vertex":
                        vertices = count
                    else:
                        faces = count
    raise ValueError(f"PLY header has no end_header marker: {path}")


def run_dense(colmap, openmvs, image_dir, sparse_model, job_dir, quality, *, resume=False):
    root = Path(job_dir)
    root = root.expanduser().resolve()
    image_dir = Path(image_dir).expanduser().resolve()
    sparse_model = Path(sparse_model).expanduser().resolve()
    dense_dir = root / "dense"
    mvs_dir = dense_dir / "mvs"
    logs_dir = root / "logs" / "dense"
    mvs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    commands = build_dense_commands(
        colmap=colmap,
        openmvs=openmvs,
        image_dir=image_dir,
        sparse_model=sparse_model,
        job_dir=root,
        quality=quality,
    )
    for index, (name, argv) in enumerate(commands):
        if resume and _stage_outputs_are_valid(name, mvs_dir, dense_dir, refined=quality.refine_mesh):
            continue
        cwd = root if name in {"image undistortion", "COLMAP import"} else mvs_dir
        log_path = logs_dir / f"{index:02d}-{_slug(name)}.log"
        run_logged_command(argv, cwd=cwd, log_path=log_path)
        _validate_stage_output(name, mvs_dir, dense_dir, refined=quality.refine_mesh)

    point_cloud = mvs_dir / "scene_dense.ply"
    mesh_stem = "scene_dense_mesh_refine" if quality.refine_mesh else "scene_dense_mesh"
    mesh_path = mvs_dir / f"{mesh_stem}.ply"
    textured_stem = f"{mesh_stem}_texture"
    glb_path = mvs_dir / f"{textured_stem}.glb"
    dense_points, _ = read_ply_counts(point_cloud)
    mesh_vertices, mesh_triangles = read_ply_counts(mesh_path)
    texture_files = sorted(mvs_dir.glob(f"{textured_stem}_*.png"))
    if not texture_files:
        raise CommandError(
            f"OpenMVS exported {glb_path.name} without its texture atlas PNG; "
            "inspect the texturing log before using the model."
        )

    mesh_dir = root / "mesh"
    exports_dir = root / "exports"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = sorted(mvs_dir.glob("scene_dense_mesh*"))
    for source in mesh_files:
        if source.suffix.lower() in {".mvs", ".ply"}:
            shutil.copy2(source, mesh_dir / source.name)
    shutil.copy2(mesh_path, exports_dir / "scene.ply")
    shutil.copy2(point_cloud, exports_dir / "scene_points.ply")
    shutil.copy2(glb_path, exports_dir / "scene.glb")
    exported_textures = []
    for texture in texture_files:
        target = exports_dir / texture.name
        shutil.copy2(texture, target)
        exported_textures.append(target)

    report = DenseReport(
        dense_points=dense_points,
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        point_cloud_path=exports_dir / "scene_points.ply",
        mesh_path=exports_dir / "scene.ply",
        obj_path=None,
        glb_path=exports_dir / "scene.glb",
        texture_files=exported_textures,
    )
    payload = {
        "dense_points": report.dense_points,
        "mesh_vertices": report.mesh_vertices,
        "mesh_triangles": report.mesh_triangles,
        "point_cloud_path": str(report.point_cloud_path),
        "mesh_path": str(report.mesh_path),
        "obj_path": None,
        "glb_path": str(report.glb_path),
        "texture_files": [str(path) for path in report.texture_files],
    }
    (dense_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report


def _validate_stage_output(stage: str, mvs_dir: Path, dense_dir: Path, refined: bool) -> None:
    stem = "scene_dense_mesh_refine" if refined else "scene_dense_mesh"
    texture_stem = f"{stem}_texture"
    required = {
        "image undistortion": [dense_dir / "colmap" / "images"],
        "COLMAP import": [mvs_dir / "scene.mvs"],
        "dense reconstruction": [mvs_dir / "scene_dense.mvs", mvs_dir / "scene_dense.ply"],
        "mesh reconstruction": [mvs_dir / "scene_dense_mesh.ply"],
        "mesh refinement": [mvs_dir / "scene_dense_mesh_refine.ply"],
        "texturing": [mvs_dir / f"{texture_stem}.glb"],
    }
    missing = [path for path in required.get(stage, []) if not path.exists()]
    empty = [path for path in required.get(stage, []) if path.is_file() and path.stat().st_size == 0]
    empty.extend(
        path for path in required.get(stage, [])
        if path.is_dir() and not any(path.iterdir())
    )
    missing.extend(empty)
    if missing:
        files = ", ".join(str(path) for path in missing)
        raise CommandError(
            f"{stage} finished without producing required artifact(s): {files}. "
            "Inspect the stage log and native tool version before continuing."
        )
    if stage == "texturing" and not any(mvs_dir.glob(f"{texture_stem}_*.png")):
        raise CommandError(
            f"{stage} finished without producing a texture atlas PNG beside "
            f"{texture_stem}.glb. Inspect the stage log before continuing."
        )


def _stage_outputs_are_valid(stage: str, mvs_dir: Path, dense_dir: Path, refined: bool) -> bool:
    try:
        _validate_stage_output(stage, mvs_dir, dense_dir, refined)
    except CommandError:
        return False
    return True


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
