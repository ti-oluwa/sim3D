"""
**BORES**

3D 3-Phase Black-Oil Reservoir Modelling and Simulation Framework.
"""

from ._precision import *  # noqa
from .models import *  # noqa
from .simulate import *  # noqa
from .constants import *  # noqa
from .grids import *  # noqa
from .factories import *  # noqa
from .boundary_conditions import *  # noqa
from .wells import *  # noqa
from .types import *  # noqa
from .relperm import *  # noqa
from .capillary_pressures import *  # noqa
from .visualization import *  # noqa
from .grids import *  # noqa
from .states import *  # noqa
from .fractures import *  # noqa
from .utils import *  # noqa
from .timing import *  # noqa
from .config import *  # noqa
from .tables import *  # noqa
from .analyses import *  # noqa
from .stores import *  # noqa
from .streams import *  # noqa
from .serialization import *  # noqa
from .errors import *  # noqa
from .diffusivity import * # noqa

use_32bit_precision()  # noqa


__version__ = "0.0.1"
