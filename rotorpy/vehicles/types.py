"""Type aliases for the vehicle models: the name carries the unit and shape.

Aliases only, for readers and static checkers (Pylance); nothing is checked
at runtime.  Shapes are documented in the comments, not enforced.  Plain
assignments rather than the ``type`` statement, since the package supports
Python >= 3.8.
"""

import numpy as np
from numpy.typing import NDArray

Mass = float                              # kg
Position = NDArray[np.float64]            # (3,) m
InertiaTensor = NDArray[np.float64]       # (3, 3) kg*m^2, symmetric
