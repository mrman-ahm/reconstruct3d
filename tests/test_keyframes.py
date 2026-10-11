import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from reconstruct3d.config import KeyframeConfig
from reconstruct3d.keyframes import CandidateMetrics, classify_candidate, extract_keyframes


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = KeyframeConfig(
            sample_fps=4.0,
            min_blur_variance=15.0,
            max_clipped_fraction=0.65,
            min_gap_seconds=0.2,
            max_gap_seconds=1.0,
            min_matches=8,
            min_inliers=6,
            min_inlier_ratio=0.2,
            min_motion_px=8.0,
            max_motion_px=100.0,
        )
        self.good = CandidateMetrics(
            blur_variance=80.0,
            clipped_fraction=0.02,
            good_matches=30,
            inliers=24,
            inlier_ratio=0.8,
            median_motion_px=18.0,
        )

    def test_accepts_first_sharp_candidate(self):
        keep, reason = classify_candidate(self.good, gap_seconds=0.0, is_first=True, config=self.config)
        self.assertTrue(keep)
        self.assertEqual(reason, "first_usable")

    def test_rejects_blurred_candidate_even_after_long_gap(self):
        blurred = CandidateMetrics(4.0, 0.02, 30, 24, 0.8, 18.0)
        keep, reason = classify_candidate(blurred, 3.0, False, self.config)
        self.assertFalse(keep)
        self.assertEqual(reason, "blurred")

    def test_skips_candidate_when_there_is_too_little_overlap(self):
        low_overlap = CandidateMetrics(80.0, 0.02, 7, 4, 0.57, 18.0)
        keep, reason = classify_candidate(low_overlap, 0.5, False, self.config)
        self.assertFalse(keep)
        self.assertEqual(reason, "low_overlap")

    def test_keeps_sharp_low_overlap_frame_after_maximum_gap_with_warning_reason(self):
        low_overlap = CandidateMetrics(80.0, 0.02, 7, 4, 0.57, 18.0)
        keep, reason = classify_candidate(low_overlap, 1.5, False, self.config)
        self.assertTrue(keep)
        self.assertEqual(reason, "max_gap_low_overlap")

    def test_skips_duplicate_view_and_excessively_fast_motion(self):
        duplicate = CandidateMetrics(80.0, 0.02, 30, 24, 0.8, 2.0)
        fast = CandidateMetrics(80.0, 0.02, 30, 24, 0.8, 130.0)
        self.assertEqual(classify_candidate(duplicate, 0.5, False, self.config), (False, "too_similar"))
        self.assertEqual(classify_candidate(fast, 0.5, False, self.config), (False, "fast_motion"))

    def test_relinks_after_a_long_gap_when_adjacent_views_still_overlap(self):
        fast_from_last_keyframe = CandidateMetrics(80.0, 0.02, 4, 3, 0.75, 180.0)

        keep, reason = classify_candidate(
            fast_from_last_keyframe,
            gap_seconds=1.2,
            is_first=False,
            config=self.config,
            adjacent_metrics=self.good,
        )

        self.assertTrue(keep)
        self.assertEqual(reason, "max_gap_relinked")


class KeyframeExtractionTests(unittest.TestCase):
    def make_moving_texture_video(self, path: Path) -> int:
        width, height, fps, count = 320, 240, 12.0, 36
        rng = np.random.default_rng(17)
        texture = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
        )
        if not writer.isOpened():
            self.skipTest("OpenCV video writer is unavailable")
        for i in range(count):
            dx = i * 2
            transform = np.float32([[1, 0, dx], [0, 1, 0]])
            frame = cv2.warpAffine(texture, transform, (width, height), borderMode=cv2.BORDER_WRAP)
            if 12 <= i < 16:
                frame = cv2.GaussianBlur(frame, (21, 21), 7)
            writer.write(frame)
        writer.release()
        return count

    def test_extracts_a_sparse_diagnostic_keyframe_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "translated-texture.avi"
            original_count = self.make_moving_texture_video(video)
            config = KeyframeConfig(
                sample_fps=4.0,
                min_blur_variance=8.0,
                max_clipped_fraction=0.9,
                min_gap_seconds=0.2,
                max_gap_seconds=0.75,
                min_matches=6,
                min_inliers=4,
                min_inlier_ratio=0.1,
                min_motion_px=3.0,
                max_motion_px=100.0,
                max_keyframes=20,
                scoring_width=320,
            )

            report = extract_keyframes(video, root / "keyframes", config)

            self.assertGreater(len(report.selected), 2)
            self.assertLess(len(report.selected), original_count)
            self.assertTrue((root / "keyframes" / "keyframes.csv").is_file())
            self.assertTrue((root / "keyframes" / "contact_sheet.jpg").is_file())
            self.assertTrue(all(item.path.is_file() for item in report.selected))
            self.assertGreater(report.candidate_count, len(report.selected))
            self.assertIn(",blurred,", (root / "keyframes" / "keyframes.csv").read_text())


if __name__ == "__main__":
    unittest.main()
