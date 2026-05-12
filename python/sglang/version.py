try:
    import pathlib

    from setuptools_scm import get_version

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    version_tag_script = repo_root / "python" / "tools" / "get_version_tag.py"
    __version__ = get_version(
        root=str(repo_root),
        fallback_version="0.0.0.dev0",
        git_describe_command=["python3", str(version_tag_script)],
        version_scheme="post-release",
    )
    __version_tuple__ = tuple(__version__.split("."))
except Exception:
    try:
        from sglang._version import __version__, __version_tuple__
    except ImportError:
        __version__ = "0.0.0.dev0"
        __version_tuple__ = (0, 0, 0, "dev0")
