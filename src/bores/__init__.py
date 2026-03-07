"""
**BORES**

3D 3-Phase Black-Oil Reservoir Modelling and Simulation Framework.
"""

from ._precision import *
from .analyses import *
from .boundary_conditions import *
from .capillary_pressures import *
from .config import *
from .constants import *
from .errors import *
from .factories import *
from .fractures import *
from .grids import *
from .models import *
from .relperm import *
from .serialization import *
from .simulate import *
from .solvers import *
from .states import *
from .stores import *
from .streams import *
from .tables import *
from .timing import *
from .types import *
from .utils import *
from .visualization import *
from .wells import *

use_32bit_precision()  


__version__ = "0.1.0"
