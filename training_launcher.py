#!/usr/bin/env python3
"""
Load a training configuration file and launch `main.py` with the resolved arguments.

The configuration file can be YAML (preferred) or JSON and supports the following keys:
  env: mapping of environment variables to set before launching.
  main_args: mapping of CLI arguments (without the leading --) to values.
             Lists become repeated values after a single --key.
  flags: list of flag names (passed as `--flag` without a value).
  ensure_dirs: list of directories to create before launching.
  extra_args: list of raw strings appended to the command.

All string values support `{repo_dir}` and `{output_dir}` substitution.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as fh:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - user-facing requirement
                raise SystemExit(
                    "PyYAML is required to read YAML configs. "
                    "Install with `pip install pyyaml` or use JSON configs."
                ) from exc
            return yaml.safe_load(fh) or {}
        if suffix == ".json":
            return json.load(fh)
    raise SystemExit(f"Unsupported config extension: {path.suffix}")


def _format_value(value: Any, variables: Dict[str, str]) -> Any:
    if isinstance(value, str):
        formatted = value.format(**variables)
        expanded = os.path.expandvars(os.path.expanduser(formatted))
        return expanded
    if isinstance(value, list):
        return [_format_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _format_value(val, variables) for key, val in value.items()}
    return value


def _ensure_directories(paths: Iterable[str]) -> None:
    for entry in paths:
        if not entry:
            continue
        Path(entry).mkdir(parents=True, exist_ok=True)


def _build_command(
    repo_dir: Path,
    main_args: Dict[str, Any],
    flags: Iterable[str],
    extra_args: Iterable[str],
) -> List[str]:
    cmd: List[str] = [sys.executable, str(repo_dir / "main.py")]

    for flag in flags:
        flag = flag.strip()
        if not flag:
            continue
        if not flag.startswith("--"):
            cmd.append(f"--{flag}")
        else:
            cmd.append(flag)

    for key, value in main_args.items():
        if value is None:
            continue
        arg_name = f"--{key}"
        if isinstance(value, list):
            if not value:
                continue
            cmd.append(arg_name)
            cmd.extend(str(item) for item in value)
        else:
            cmd.extend([arg_name, str(value)])

    cmd.extend(str(arg) for arg in extra_args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Llada training from a config file.")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON training config.")
    parser.add_argument("--repo-dir", required=True, help="Repository root for resolving relative paths.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved command without executing.")
    args, remaining = parser.parse_known_args()

    repo_dir = Path(args.repo_dir).resolve()
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)

    variables: Dict[str, str] = {
        "repo_dir": str(repo_dir),
    }

    raw_main_args = config.get("main_args", {})
    if not isinstance(raw_main_args, dict):
        raise SystemExit("Config field `main_args` must be a mapping of argument names to values.")
    main_args = _format_value(raw_main_args, variables)

    output_dir = main_args.get("output_dir")
    if isinstance(output_dir, str):
        variables["output_dir"] = output_dir

    env_config = config.get("env", {})
    if not isinstance(env_config, dict):
        raise SystemExit("Config field `env` must be a mapping of environment variables to values.")
    env_formatted = _format_value(env_config, variables)

    ensure_dirs = config.get("ensure_dirs", [])
    if not isinstance(ensure_dirs, list):
        raise SystemExit("Config field `ensure_dirs` must be a list of directory paths.")
    ensure_formatted = _format_value(ensure_dirs, variables)

    flags = config.get("flags", [])
    if not isinstance(flags, list):
        raise SystemExit("Config field `flags` must be a list.")

    extra_args = config.get("extra_args", [])
    if not isinstance(extra_args, list):
        raise SystemExit("Config field `extra_args` must be a list.")
    extra_formatted = _format_value(extra_args, variables)

    cmd = _build_command(repo_dir, main_args, flags, extra_formatted)
    if remaining:
        cmd.extend(remaining)

    env = os.environ.copy()
    for key, value in env_formatted.items():
        env[key] = str(value)

    if output_dir:
        ensure_formatted = ensure_formatted + [output_dir]

    _ensure_directories(ensure_formatted)

    print("=========================================")
    print(f"Launching training with config: {config_path}")
    print("Command:")
    print("  " + " ".join(cmd))
    if env_formatted:
        print("Environment overrides:")
        for key, value in env_formatted.items():
            print(f"  {key}={value}")
    print("=========================================")

    if args.dry_run:
        return

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
