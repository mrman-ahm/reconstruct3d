import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from reconstruct3d.video import VideoMetadata, metadata_from_ffprobe, probe_video


class ProbeMetadataTests(unittest.TestCase):
    def test_parses_fractional_frame_rate_and_camera_tags(self):
        metadata = metadata_from_ffprobe(
            Path("walk.mp4"),
            {
                "streams": [
                    {
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "30000/1001",
                        "nb_frames": "1800",
                        "duration": "60.06",
                        "tags": {"com.apple.quicktime.make": "ExampleCo", "model": "Phone X"},
                    }
                ],
                "format": {"duration": "60.06"},
            },
        )

        self.assertAlmostEqual(metadata.fps, 30000 / 1001)
        self.assertEqual(metadata.width, 1920)
        self.assertEqual(metadata.height, 1080)
        self.assertEqual(metadata.frame_count, 1800)
        self.assertEqual(metadata.codec, "h264")
        self.assertEqual(metadata.camera_make, "ExampleCo")
        self.assertEqual(metadata.camera_model, "Phone X")

    def test_applies_rotation_and_uses_format_duration_fallback(self):
        metadata = metadata_from_ffprobe(
            Path("portrait.mp4"),
            {
                "streams": [
                    {
                        "codec_name": "hevc",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "25/1",
                        "nb_frames": "N/A",
                        "side_data_list": [{"rotation": 90}],
                    }
                ],
                "format": {"duration": "12.5", "tags": {"model": "Phone Y"}},
            },
        )

        self.assertEqual(metadata.width, 1080)
        self.assertEqual(metadata.height, 1920)
        self.assertEqual(metadata.rotation_degrees, 90)
        self.assertAlmostEqual(metadata.duration_seconds, 12.5)
        self.assertIsNone(metadata.frame_count)
        self.assertEqual(metadata.camera_model, "Phone Y")

    def test_ignores_unrelated_private_tags(self):
        metadata = metadata_from_ffprobe(
            Path("walk.mp4"),
            {
                "streams": [{"width": 640, "height": 480, "avg_frame_rate": "10/1"}],
                "format": {"duration": "1", "tags": {"location": "+24.0+67.0/"}},
            },
        )

        serialized = json.dumps(metadata.to_dict())
        self.assertNotIn("location", serialized)
        self.assertNotIn("67.0", serialized)


class OpenCvProbeTests(unittest.TestCase):
    def test_reads_a_video_when_ffprobe_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "short.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (96, 64)
            )
            self.assertTrue(writer.isOpened(), "OpenCV must have a local test-video writer")
            for i in range(16):
                image = np.full((64, 96, 3), (i * 9) % 255, dtype=np.uint8)
                writer.write(image)
            writer.release()

            metadata = probe_video(path, ffprobe_path="/missing/ffprobe")

            self.assertIsInstance(metadata, VideoMetadata)
            self.assertEqual((metadata.width, metadata.height), (96, 64))
            self.assertEqual(metadata.frame_count, 16)
            self.assertAlmostEqual(metadata.fps, 8.0)
            self.assertAlmostEqual(metadata.duration_seconds, 2.0)
            self.assertEqual(metadata.metadata_source, "opencv")

    def test_rejects_missing_or_unreadable_video(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "video"):
                probe_video(Path(temp_dir) / "missing.mp4", ffprobe_path="/missing/ffprobe")


if __name__ == "__main__":
    unittest.main()
