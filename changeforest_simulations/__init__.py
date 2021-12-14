import pkg_resources

from ._load import DATASETS, load
from .methods import estimate_changepoints
from .score import adjusted_rand_score
from .simulate import simulate

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = [
    "adjusted_rand_score",
    "load",
    "simulate",
    "DATASETS",
    "estimate_changepoints",
]
