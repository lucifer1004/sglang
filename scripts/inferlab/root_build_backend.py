"""PEP 517 backend that builds the SGLang package in python/ from the repo root.

Each hook runs python/'s own build backend with python/ as the working
directory, and makes caller-supplied output directories absolute first so
they keep pointing where the frontend expects.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import tomllib
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "python"


def _backend():
    config = tomllib.loads((_PACKAGE_DIR / "pyproject.toml").read_text())
    name = config["build-system"]["build-backend"]
    module_name, _, attr = name.partition(":")
    backend = importlib.import_module(module_name)
    return getattr(backend, attr) if attr else backend


@contextlib.contextmanager
def _in_package_dir():
    previous = os.getcwd()
    os.chdir(_PACKAGE_DIR)
    try:
        yield _backend()
    finally:
        os.chdir(previous)


def _absolute(path):
    return None if path is None else os.path.abspath(path)


def get_requires_for_build_wheel(config_settings=None):
    with _in_package_dir() as backend:
        return backend.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    with _in_package_dir() as backend:
        return backend.get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    metadata_directory = _absolute(metadata_directory)
    with _in_package_dir() as backend:
        return backend.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    wheel_directory = _absolute(wheel_directory)
    metadata_directory = _absolute(metadata_directory)
    with _in_package_dir() as backend:
        return backend.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    sdist_directory = _absolute(sdist_directory)
    with _in_package_dir() as backend:
        return backend.build_sdist(sdist_directory, config_settings)
