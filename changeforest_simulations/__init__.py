import pkg_resources

from .load import load_iris, load_letters
from .simulate import simulate

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = ["load_iris", "load_letters", "simulate"]
