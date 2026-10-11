import tempfile
import unittest
from pathlib import Path

from reconstruct3d.config import quality_config
from reconstruct3d.toolchain import Toolchain, validate_startup


class ToolchainTests(unittest.TestCase):
    def test_preflight_starts_cpu_reconstruction_programs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary = root / "fake-tool"
            binary.write_text("#!/usr/bin/env python3\nprint('help available')\n")
            binary.chmod(0o755)
            names = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh", "TransformScene")
            tools = Toolchain(
                colmap=str(binary),
                openmvs={name: str(binary) for name in names},
                ffmpeg=None,
                ffprobe=None,
                blender=None,
            )

            checks = validate_startup(tools, quality_config("balanced"))

            self.assertEqual(len(checks), 6)
            self.assertTrue(all(item["started"] for item in checks))
            self.assertTrue(all(item["banner"] == "help available" for item in checks))

    def test_preflight_lists_missing_cpu_tools(self):
        tools = Toolchain(
            colmap=None,
            openmvs={name: None for name in ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh", "TransformScene")},
            ffmpeg=None,
            ffprobe=None,
            blender=None,
        )
        with self.assertRaisesRegex(RuntimeError, "COLMAP") as raised:
            validate_startup(tools, quality_config("balanced"))
        self.assertIn("OpenMVS InterfaceCOLMAP", str(raised.exception))

    def test_preflight_accepts_openmvs_usage_exit_after_banner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary = root / "fake-tool.py"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "if pathlib.Path(sys.argv[0]).name == 'colmap':\n"
                " print('COLMAP 4.2.1 help')\n"
                "else:\n"
                " print('OpenMVS x64 v2.4.0\\nAvailable options')\n"
                " sys.exit(1)\n"
            )
            binary.chmod(0o755)
            colmap = root / "colmap"
            colmap.symlink_to(binary)
            names = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh", "TransformScene")
            programs = {}
            for name in names:
                path = root / name
                path.symlink_to(binary)
                programs[name] = str(path)
            tools = Toolchain(str(colmap), programs, None, None, None)

            checks = validate_startup(tools, quality_config("balanced"))

            openmvs_checks = [item for item in checks if item["program"].startswith("OpenMVS ")]
            self.assertEqual(len(openmvs_checks), 5)
            self.assertTrue(all(item["started"] for item in openmvs_checks))
            self.assertTrue(all(item["returncode"] == 1 for item in openmvs_checks))

    def test_preflight_rejects_unexpected_openmvs_startup_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary = root / "fake-tool.py"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "print('startup failed')\n"
                "sys.exit(1)\n"
            )
            binary.chmod(0o755)
            colmap = root / "colmap"
            colmap.symlink_to(binary)
            names = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh", "TransformScene")
            programs = {}
            for name in names:
                path = root / name
                path.symlink_to(binary)
                programs[name] = str(path)
            tools = Toolchain(str(colmap), programs, None, None, None)

            with self.assertRaisesRegex(RuntimeError, "failed its startup check"):
                validate_startup(tools, quality_config("balanced"))


if __name__ == "__main__":
    unittest.main()
