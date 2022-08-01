# fmt: off
from qsparse.convert import convert
from qsparse.fuse import fuse_bn
from qsparse.quantize import quantize, DecimalQuantizer, ScalerQuantizer, AdaptiveQuantizer
from qsparse.sparse import MagnitudePruningCallback, UniformPruningCallback, prune, devise_layerwise_pruning_schedule
from qsparse.util import auto_name_prune_quantize_layers, calculate_mask_given_importance
from qsparse.util import get_option as get_qsparse_option
from qsparse.util import set_options as set_qsparse_options

# fmt: on

__version__ = "2.0.0"
