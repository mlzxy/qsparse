from qsparse.convert import convert
from qsparse.fuse import fuse_bn
from qsparse.quantize import linear_quantize_callback, quantize
from qsparse.sparse import (
    prune,
    structured_prune_callback,
    unstructured_prune_callback,
    unstructured_uniform_prune_callback,
)
from qsparse.util import auto_name_prune_quantize_layers
from qsparse.util import get_option as get_qsparse_option
from qsparse.util import set_options as set_qsparse_options

__all__ = (
    "quantize",
    "prune",
    "convert",
    "linear_quantize_callback",
    "unstructured_prune_callback",
    "structured_prune_callback",
    "unstructured_uniform_prune_callback",
    "auto_name_prune_quantize_layers",
    "set_qsparse_options",
    "get_qsparse_option",
    "fuse_bn",
)

__version__ = "1.2.7"
