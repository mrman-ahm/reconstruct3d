import tempfile
import unittest
from pathlib import Path

from reconstruct3d.config import quality_config
from reconstruct3d.reconstruction import (
    assess_sparse,
    build_dense_commands,
    build_sparse_commands,
    read_sparse_counts,
    run_sparse,
    write_camera_trajectory,
)


class SparseCommandTests(unittest.TestCase):
    def test_small_video_uses_cpu_exhaustive_matching_and_shared_intrinsics(self):
        quality = quality_config("balanced")
        steps = build_sparse_commands(
            colmap="colmap",
            images_dir=Path("job/keyframes"),
            job_dir=Path("job"),
            image_count=80,
            quality=quality,
        )
        by_name = dict(steps)
        self.assertEqual([name for name, _ in steps], ["feature extraction", "exhaustive matching", "sparse mapping"])
        extraction = by_name["feature extraction"]
        self.assertIn("--FeatureExtraction.use_gpu", extraction)
        self.assertIn("0", extraction)
        matcher = by_name["exhaustive matching"]
        self.assertIn("--FeatureMatching.max_num_matches", matcher)
        self.assertNotIn("--SiftMatching.max_num_matches", matcher)
        self.assertIn("--ImageReader.single_camera", extraction)
        self.assertIn("1", extraction)
        self.assertIn("SIMPLE_RADIAL", extraction)

    def test_sparse_mapping_output_directory_is_created_before_colmap_mapper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "keyframes"
            images.mkdir()
            for index in range(6):
                (images / f"frame_{index:04d}.jpg").touch()
            (images / "contact_sheet.jpg").touch()
            fake_colmap = root / "colmap-fixture"
            fake_colmap.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                "def option(name): return Path(args[args.index(name) + 1])\n"
                "if args[0] == 'feature_extractor':\n"
                "    image_dir = option('--image_path')\n"
                "    if any(not path.name.startswith('frame_') for path in image_dir.iterdir()):\n"
                "        raise SystemExit('non-frame diagnostic passed to COLMAP')\n"
                "elif args[0] == 'mapper':\n"
                "    output = option('--output_path')\n"
                "    if not output.is_dir(): raise SystemExit('output_path is not a directory')\n"
                "    model = output / '0'\n"
                "    model.mkdir()\n"
                "    (model / 'images.bin').touch()\n"
                "    (model / 'points3D.bin').touch()\n"
                "elif args[0] == 'model_converter':\n"
                "    output = option('--output_path')\n"
                "    if option('--output_type').name == 'TXT':\n"
                "        output.mkdir(parents=True, exist_ok=True)\n"
                "        rows = [f'{i} 1 0 0 0 0 0 0 1 frame_{i - 1:04d}.jpg' for i in range(1, 7)]\n"
                "        (output / 'images.txt').write_text('\\n\\n'.join(rows) + '\\n')\n"
                "        points = [f'{i} 0 0 0 255 255 255 0.2' for i in range(1, 101)]\n"
                "        (output / 'points3D.txt').write_text('\\n'.join(points) + '\\n')\n"
                "    else:\n"
                "        output.write_text('ply')\n",
                encoding="utf-8",
            )
            fake_colmap.chmod(0o755)

            report = run_sparse(str(fake_colmap), images, root / "job", quality_config("preview"))

            self.assertTrue((root / "job" / "sparse" / "models").is_dir())
            self.assertEqual(report.selected_images, 6)
            self.assertTrue(report.ready_for_dense)

    def test_large_video_uses_bounded_sequential_matching(self):
        steps = build_sparse_commands(
            colmap="colmap",
            images_dir=Path("images"),
            job_dir=Path("job"),
            image_count=220,
            quality=quality_config("high"),
        )
        name, matcher = steps[1]
        self.assertEqual(name, "sequential matching")
        self.assertIn("--SequentialMatching.overlap", matcher)
        self.assertIn("10", matcher)

    def test_large_video_can_enable_loop_closure_with_a_vocabulary_tree(self):
        steps = build_sparse_commands(
            colmap="colmap",
            images_dir=Path("images"),
            job_dir=Path("job"),
            image_count=220,
            quality=quality_config("balanced"),
            vocab_tree_path=Path("/data/colmap-vocab.bin"),
        )
        matcher = steps[1][1]
        self.assertIn("--SequentialMatching.loop_detection", matcher)
        self.assertIn("1", matcher)
        self.assertIn("--SequentialMatching.vocab_tree_path", matcher)

    def test_sparse_gate_stops_when_poses_or_points_are_inadequate(self):
        weak = assess_sparse(selected_images=20, registered_images=5, points=300)
        strong = assess_sparse(selected_images=20, registered_images=16, points=700)
        self.assertFalse(weak.ready_for_dense)
        self.assertTrue(any("registration" in warning.lower() for warning in weak.warnings))
        self.assertTrue(strong.ready_for_dense)

    def test_reads_sparse_image_and_point_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_text = root / "images.txt"
            point_text = root / "points3D.txt"
            image_text.write_text(
                "# image header\n1 1 0 0 0 0 0 0 1 frame_0001.jpg\n\n"
                "2 1 0 0 0 1 0 0 1 frame_0002.jpg\n\n"
            )
            point_text.write_text("# point header\n1 0 0 1 255 255 255 0.5\n2 1 0 1 0 0 0 0.7\n")

            registered, points = read_sparse_counts(image_text, point_text)

            self.assertEqual(registered, 2)
            self.assertEqual(points, 2)

    def test_camera_trajectory_converts_world_to_camera_translation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images.txt"
            output = root / "camera_trajectory.csv"
            images.write_text(
                "# image header\n1 1 0 0 0 1 2 3 1 frame_0001.jpg\n"
                "\n# points2D\n0.0 0.0 -1\n"
            )

            count = write_camera_trajectory(images, output)

            self.assertEqual(count, 1)
            fields = output.read_text().splitlines()[1].split(",")
            self.assertEqual(fields[2], "frame_0001.jpg")
            self.assertEqual(fields[7:10], ["-1.0", "-2.0", "-3.0"])
            self.assertEqual(fields[10:13], ["1.0", "2.0", "3.0"])


class DenseCommandTests(unittest.TestCase):
    def test_dense_stages_use_cpu_build_defaults_and_keep_native_openmvs_outputs(self):
        tools = {name: f"/tools/{name}" for name in (
            "InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh",
            "TextureMesh", "TransformScene",
        )}
        steps = build_dense_commands(
            colmap="colmap",
            openmvs=tools,
            image_dir=Path("job/keyframes"),
            sparse_model=Path("job/sparse/model"),
            job_dir=Path("job"),
            quality=quality_config("balanced"),
        )
        names = [name for name, _ in steps]
        self.assertEqual(names, [
            "image undistortion", "COLMAP import", "dense reconstruction",
            "mesh reconstruction", "texturing",
        ])
        mesh_file = dict(steps)["texturing"]
        self.assertEqual(mesh_file[1], "scene_dense.mvs")
        self.assertEqual(mesh_file[mesh_file.index("--mesh-file") + 1], "scene_dense_mesh.ply")
        self.assertEqual(mesh_file[mesh_file.index("--export-type") + 1], "glb")
        self.assertEqual(mesh_file[mesh_file.index("--output-file") + 1], "scene_dense_mesh_texture.glb")
        for _, command in steps:
            self.assertNotIn("--cuda-device", command)
            self.assertNotIn("--gpu-device", command)
        for name, command in steps[2:]:
            if name != "GLB export":
                self.assertIn("--max-threads", command, name)

    def test_high_quality_includes_cpu_mesh_refinement(self):
        tools = {name: f"/tools/{name}" for name in (
            "InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh",
            "TextureMesh", "TransformScene",
        )}
        steps = build_dense_commands(
            colmap="colmap",
            openmvs=tools,
            image_dir=Path("images"),
            sparse_model=Path("sparse"),
            job_dir=Path("job"),
            quality=quality_config("high"),
        )
        self.assertIn("mesh refinement", [name for name, _ in steps])
        mesh_file = dict(steps)["texturing"]
        self.assertEqual(mesh_file[mesh_file.index("--mesh-file") + 1], "scene_dense_mesh_refine.ply")


if __name__ == "__main__":
    unittest.main()
