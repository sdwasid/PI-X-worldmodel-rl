import subprocess
import sys
from importlib import metadata
from pathlib import Path
from textwrap import dedent

import pytest


def test_all_top_level_modules_import_without_lerobot() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = dedent(
        """\
        import importlib
        import pkgutil
        import sys
        from importlib import metadata
        from pathlib import Path

        source_root = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(source_root))

        for loaded_name in tuple(sys.modules):
            if loaded_name == "lerobot" or loaded_name.startswith("lerobot."):
                del sys.modules[loaded_name]
        sys.modules["lerobot"] = None

        real_version = metadata.version

        def version_without_lerobot(distribution_name: str) -> str:
            if distribution_name.casefold() == "lerobot":
                raise metadata.PackageNotFoundError(distribution_name)
            return real_version(distribution_name)

        metadata.version = version_without_lerobot

        def raise_discovery_error(module_name: str) -> None:
            error = sys.exc_info()[1]
            if error is None:
                raise RuntimeError(f"Failed to discover package {module_name}")
            raise RuntimeError(
                f"Failed to discover package {module_name}: "
                f"{type(error).__name__}: {error}"
            ) from error

        current_module = "pi0fast_wm_rl"
        try:
            package = importlib.import_module(current_module)
            module_names = {package.__name__}
            for module_info in pkgutil.walk_packages(
                package.__path__,
                f"{package.__name__}.",
                onerror=raise_discovery_error,
            ):
                current_module = module_info.name
                module_names.add(current_module)

            for current_module in sorted(module_names):
                importlib.import_module(current_module)
        except BaseException as error:
            print(
                f"FAILED_IMPORT: {current_module}: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            raise

        print(f"IMPORTED_MODULE_COUNT={len(module_names)}")
        """
    )

    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(source_root)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, (
        "Import smoke subprocess failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "IMPORTED_MODULE_COUNT=" in result.stdout


def test_missing_lerobot_has_friendly_pi0fast_error(monkeypatch) -> None:
    from pi0fast_wm_rl.data import lerobot_adapter
    from pi0fast_wm_rl.policies.pi0fast_adapter import Pi0FastAdapter

    original = lerobot_adapter.metadata.version

    def missing(name: str) -> str:
        if name == "lerobot":
            raise metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(lerobot_adapter.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="LeRobot is not installed"):
        Pi0FastAdapter("models/pretrained/pi0fast-base", "pick up the cube")


def test_cli_doctor_runs_with_optional_dependencies_missing(capsys) -> None:
    from pi0fast_wm_rl.cli import main

    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert '"python"' in output
    assert '"lerobot"' in output
