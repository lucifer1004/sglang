try:
    from sglang._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.5.6"
    __version_tuple__ = (0, 5, 6)
