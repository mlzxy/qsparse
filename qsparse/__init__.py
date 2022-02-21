# fmt: off
from qsparse.convert import convert
from qsparse.fuse import fuse_bn
from qsparse.quantize import (DecimalOptimizer, ScalerOptimizer,
                              linear_quantize_callback, quantize,
                              scaler_quantize_callback)
from qsparse.sparse import (BanditPruningCallback, MagnitudePruningCallback,
                            UniformPruningCallback, prune)
from qsparse.util import auto_name_prune_quantize_layers
from qsparse.util import get_option as get_qsparse_option
from qsparse.util import set_options as set_qsparse_options

# fmt: on

__all__ = (
    "quantize",
    "prune",
    "convert",
    "linear_quantize_callback",
    "scaler_quantize_callback",
    "DecimalOptimizer",
    "ScalerOptimizer",
    "MagnitudePruningCallback",
    "UniformPruningCallback",
    "BanditPruningCallback",
    "auto_name_prune_quantize_layers",
    "set_qsparse_options",
    "get_qsparse_option",
    "fuse_bn",
)

__version__ = "1.2.11"
