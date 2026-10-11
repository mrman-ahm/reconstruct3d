import sys
import tempfile
import unittest
from pathlib import Path

from reconstruct3d.process import CommandError, run_logged_command


class LoggedCommandTests(unittest.TestCase):
    def test_logs_argv_and_combined_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log = root / "stage.log"
            run_logged_command(
                [sys.executable, "-c", "print('command finished')"],
                cwd=root,
                log_path=log,
            )
            content = log.read_text()
            self.assertIn("command finished", content)
            self.assertIn("COMMAND:", content)

    def test_reports_nonzero_exit_and_log_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log = root / "failed.log"
            with self.assertRaisesRegex(CommandError, "exit code 7") as raised:
                run_logged_command(
                    [sys.executable, "-c", "import sys; print('backend failed'); sys.exit(7)"],
                    cwd=root,
                    log_path=log,
                )
            self.assertIn(str(log), str(raised.exception))
            self.assertIn("backend failed", str(raised.exception))

    def test_reports_missing_executable_without_shell_interpolation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(CommandError, "not found"):
                run_logged_command(
                    ["/missing/reconstruct-tool; touch /tmp/should-not-exist"],
                    cwd=Path(temp_dir),
                    log_path=Path(temp_dir) / "missing.log",
                )


if __name__ == "__main__":
    unittest.main()
