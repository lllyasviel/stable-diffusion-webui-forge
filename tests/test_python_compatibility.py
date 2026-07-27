import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules import launch_utils


def version(major, minor, micro=0):
    return SimpleNamespace(major=major, minor=minor, micro=micro)


class PythonCompatibilityTests(unittest.TestCase):
    def test_windows_supports_python_310_through_312(self):
        for minor in (10, 11, 12):
            with self.subTest(minor=minor):
                self.assertTrue(launch_utils.is_python_version_supported(version(3, minor), "Windows"))

    def test_other_platforms_keep_existing_versions_and_add_312(self):
        for minor in (7, 8, 9, 10, 11, 12):
            with self.subTest(minor=minor):
                self.assertTrue(launch_utils.is_python_version_supported(version(3, minor), "Linux"))

    def test_unsupported_versions_are_rejected(self):
        unsupported = (
            (version(2, 12), "Windows"),
            (version(3, 9), "Windows"),
            (version(3, 13), "Windows"),
            (version(3, 13), "Linux"),
        )

        for version_info, system in unsupported:
            with self.subTest(version=version_info, system=system):
                self.assertFalse(launch_utils.is_python_version_supported(version_info, system))

    def test_current_interpreter_is_supported(self):
        self.assertTrue(launch_utils.is_python_version_supported())


class RequirementsBootstrapTests(unittest.TestCase):
    @property
    def requirements_file(self):
        return Path(__file__).parents[1] / "requirements_versions.txt"

    def test_setuptools_pin_is_read_from_requirements(self):
        self.assertEqual(
            launch_utils.pinned_requirement_version(self.requirements_file, "setuptools"),
            "69.5.1",
        )

    def test_missing_requirement_returns_none(self):
        self.assertIsNone(
            launch_utils.pinned_requirement_version(self.requirements_file, "not-a-package"),
        )

    def test_missing_setuptools_is_bootstrapped(self):
        with mock.patch.object(launch_utils.args, "skip_install", False), \
                mock.patch.object(
                    launch_utils.importlib.metadata,
                    "version",
                    side_effect=launch_utils.importlib.metadata.PackageNotFoundError,
                ), \
                mock.patch.object(launch_utils, "run_pip") as run_pip:
            launch_utils.ensure_setuptools(self.requirements_file)

        run_pip.assert_called_once_with("install setuptools==69.5.1", "setuptools==69.5.1")

    def test_matching_setuptools_is_not_reinstalled(self):
        with mock.patch.object(launch_utils.args, "skip_install", False), \
                mock.patch.object(launch_utils.importlib.metadata, "version", return_value="69.5.1"), \
                mock.patch.object(launch_utils, "run_pip") as run_pip:
            launch_utils.ensure_setuptools(self.requirements_file)

        run_pip.assert_not_called()


if __name__ == "__main__":
    unittest.main()
