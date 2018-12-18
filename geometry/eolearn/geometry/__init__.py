"""
Subpackage containing EOTasks for geometrical transformations
"""

from .data_transform import VectorToRaster, RasterToVector
from .sampling import PointSamplingTask, PointSampler, PointRasterSampler
from .utilities import ErosionTask


__version__ = '0.4.1'
