import unittest
from types import SimpleNamespace

from ryu_tts.engine_installer import build_engine_install_command


class EngineInstallCommandTests(unittest.TestCase):
    def test_pocket_uses_the_cpu_torch_index_on_linux(self) -> None:
        command = build_engine_install_command(
            SimpleNamespace(id="pocket", pip_packages=["pocket-tts"]),
            executable="/managed/python",
            platform="linux",
        )
        self.assertEqual(command[:5], [
            "/managed/python",
            "-m",
            "pip",
            "install",
            "pocket-tts",
        ])
        self.assertEqual(
            command[-2:],
            ["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
        )

    def test_pocket_uses_platform_default_cpu_wheels_elsewhere(self) -> None:
        for platform in ("darwin", "win32"):
            with self.subTest(platform=platform):
                command = build_engine_install_command(
                    SimpleNamespace(id="pocket", pip_packages=["pocket-tts"]),
                    executable="/managed/python",
                    platform=platform,
                )
                self.assertNotIn("--extra-index-url", command)

    def test_engine_package_names_come_from_the_registry(self) -> None:
        config = SimpleNamespace(id="kitten", pip_packages=["kittentts"])
        command = build_engine_install_command(config, executable="python", platform="linux")
        self.assertEqual(command[4:], config.pip_packages)
        self.assertNotIn("--extra-index-url", command)


if __name__ == "__main__":
    unittest.main()
