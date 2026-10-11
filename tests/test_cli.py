import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from reconstruct3d.cli import main, safe_job_name


class CliUtilityTests(unittest.TestCase):
    def test_job_name_is_a_safe_basename(self):
        self.assertEqual(safe_job_name("../../Room walkthrough.MP4"), "room-walkthrough")
        self.assertEqual(safe_job_name("..."), "reconstruction")

    def test_doctor_emits_machine_readable_environment_report(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["--doctor"])

        report = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertIn("hardware", report)
        self.assertIn("toolchain", report)
        self.assertIn("alternative_reconstruction_commands", report["hardware"])
        self.assertIn("cpu_count", report["hardware"])
        self.assertIsInstance(report["native_startup_error"], str)


class KeyframeCommandTests(unittest.TestCase):
    def test_keyframe_stop_and_resume_preserve_inspectable_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "room-smoke.avi"
            self.write_moving_texture_video(video)
            job = root / "room-job"
            args = [
                str(video), "--output", str(job), "--stop-after", "keyframes",
                "--sample-fps", "4", "--min-motion-px", "3", "--max-gap-seconds", "0.75",
            ]

            first_output = io.StringIO()
            with contextlib.redirect_stdout(first_output):
                first_result = main(args)
            first_report = json.loads((job / "keyframes" / "report.json").read_text())
            first_frame = sorted((job / "keyframes").glob("frame_*.jpg"))[0]
            first_mtime = first_frame.stat().st_mtime_ns

            resume_output = io.StringIO()
            with contextlib.redirect_stdout(resume_output):
                resume_result = main([*args, "--resume"])

            self.assertEqual(first_result, 0)
            self.assertEqual(resume_result, 0)
            self.assertGreater(first_report["selected_frames"], 2)
            self.assertTrue((job / "keyframes" / "contact_sheet.jpg").is_file())
            self.assertTrue((job / "diagnostics" / "keyframes.csv").is_file())
            self.assertEqual(first_frame.stat().st_mtime_ns, first_mtime)
            self.assertEqual(json.loads((job / "job.json").read_text())["stopped_after"], "keyframes")

    def test_default_run_stops_with_native_tool_diagnostic_after_saving_keyframes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            empty_path = root / "empty-bin"
            empty_path.mkdir()
            video = root / "room-smoke.avi"
            self.write_moving_texture_video(video)
            job = root / "room-job"
            stderr = io.StringIO()
            with mock.patch.dict(os.environ, {"PATH": str(empty_path)}):
                with contextlib.redirect_stderr(stderr):
                    result = main([str(video), "--output", str(job)])

            state = json.loads((job / "job.json").read_text())
            self.assertEqual(result, 2)
            self.assertIn("COLMAP command line program", stderr.getvalue())
            self.assertTrue((job / "keyframes" / "contact_sheet.jpg").is_file())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["preflight"]["status"], "failed")

    @staticmethod
    def write_moving_texture_video(path: Path) -> None:
        width, height, fps, count = 320, 240, 12.0, 36
        rng = np.random.default_rng(23)
        texture = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        if not writer.isOpened():
            raise unittest.SkipTest("OpenCV video writer is unavailable")
        for index in range(count):
            transform = np.float32([[1, 0, index * 2], [0, 1, 0]])
            frame = cv2.warpAffine(texture, transform, (width, height), borderMode=cv2.BORDER_WRAP)
            writer.write(frame)
        writer.release()


if __name__ == "__main__":
    unittest.main()
