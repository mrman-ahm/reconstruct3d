import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from reconstruct3d.config import quality_config
from reconstruct3d.dense import read_ply_counts, run_dense


PLY_MESH = """ply
format ascii 1.0
element vertex 4
property float x
element face 2
property list uchar int vertex_indices
end_header
"""


class PlyHeaderTests(unittest.TestCase):
    def test_reads_vertex_and_face_counts_from_ascii_or_binary_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mesh.ply"
            path.write_text(PLY_MESH + "0\n0\n0\n0\n0\n0\n")

            self.assertEqual(read_ply_counts(path), (4, 2))


class DenseRunTests(unittest.TestCase):
    def test_fake_executables_exercise_stage_order_and_real_output_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tools = self.make_fake_tools(root / "bin")
            job = root / "walk-job"
            keyframes = job / "keyframes"
            sparse = job / "sparse" / "model"
            keyframes.mkdir(parents=True)
            sparse.mkdir(parents=True)
            (keyframes / "frame_0001.jpg").write_bytes(b"image")

            report = run_dense(
                colmap=tools["colmap"],
                openmvs={key: value for key, value in tools.items() if key != "colmap"},
                image_dir=keyframes,
                sparse_model=sparse,
                job_dir=job,
                quality=quality_config("balanced"),
            )
            calls = (job / "dense" / "mvs" / "calls.txt").read_text()
            resumed_report = run_dense(
                colmap=tools["colmap"],
                openmvs={key: value for key, value in tools.items() if key != "colmap"},
                image_dir=keyframes,
                sparse_model=sparse,
                job_dir=job,
                quality=quality_config("balanced"),
                resume=True,
            )

            self.assertEqual(report.dense_points, 3)
            self.assertEqual(resumed_report.dense_points, 3)
            self.assertEqual((job / "dense" / "mvs" / "calls.txt").read_text(), calls)
            self.assertEqual((report.mesh_vertices, report.mesh_triangles), (4, 2))
            self.assertTrue(report.point_cloud_path.is_file())
            self.assertTrue(report.mesh_path.is_file())
            self.assertIsNone(report.obj_path)
            self.assertTrue(report.glb_path.is_file())
            self.assertEqual(len(report.texture_files), 1)
            self.assertTrue(report.texture_files[0].is_file())
            self.assertTrue((job / "dense" / "report.json").is_file())
            self.assertTrue((job / "logs" / "dense" / "04-texturing.log").is_file())

    @staticmethod
    def make_fake_tools(directory: Path) -> dict[str, str]:
        directory.mkdir(parents=True)
        script = directory / "fake_tool.py"
        script.write_text(textwrap.dedent("""\
            #!/usr/bin/env python3
            import pathlib, sys
            name = pathlib.Path(sys.argv[0]).name
            args = sys.argv[1:]
            with pathlib.Path('calls.txt').open('a') as calls: calls.write(name + '\\n')
            def option(flag): return pathlib.Path(args[args.index(flag) + 1])
            def ply(path, vertices, faces=0):
                path.write_text('ply\\nformat ascii 1.0\\nelement vertex %d\\nproperty float x\\nelement face %d\\nproperty list uchar int vertex_indices\\nend_header\\n' % (vertices, faces))
            if name == 'colmap' and args[0] == 'image_undistorter':
                images = option('--output_path') / 'images'
                images.mkdir(parents=True)
                (images / 'frame.jpg').write_bytes(b'image')
            elif name == 'InterfaceCOLMAP':
                option('-o').write_text('mvs scene')
            elif name == 'DensifyPointCloud':
                pathlib.Path('scene_dense.mvs').write_text('dense scene')
                ply(pathlib.Path('scene_dense.ply'), 3)
            elif name == 'ReconstructMesh':
                ply(pathlib.Path('scene_dense_mesh.ply'), 4, 2)
            elif name == 'TextureMesh':
                if (args[0] != 'scene_dense.mvs' or option('--mesh-file').name != 'scene_dense_mesh.ply'
                        or args[args.index('--export-type') + 1] != 'glb'):
                    raise SystemExit('TextureMesh must export the generated PLY directly as GLB')
                option('--output-file').write_bytes(b'fake glb')
                pathlib.Path('scene_dense_mesh_texture_0.png').write_bytes(b'fake atlas')
            elif name == 'TransformScene':
                pathlib.Path('scene_dense_mesh_texture.glb').write_bytes(b'fake glb')
            """), encoding="utf-8")
        script.chmod(0o755)
        names = ["colmap", "InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh", "TransformScene"]
        tools = {}
        for name in names:
            target = directory / name
            target.symlink_to(script)
            tools[name if name == "colmap" else name] = str(target)
        return tools


if __name__ == "__main__":
    unittest.main()
