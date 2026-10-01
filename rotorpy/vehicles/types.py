"""Type aliases for the vehicle models: the name carries the unit and shape."""

import numpy as np
from numpy.typing import NDArray

Mass = float                              # kg
Position = NDArray[np.float64]            # (3,) m
InertiaTensor = NDArray[np.float64]       # (3, 3) kg*m^2, symmetric
Velocity = NDArray[np.float64]            # (3,) m/s
AngularVelocity = NDArray[np.float64]     # (3,) rad/s
Quaternion = NDArray[np.float64]          # (4,) [x, y, z, w], scalar last
PerRotor = NDArray[np.float64]            # (N,) one value per rotor
PerRotorVector = NDArray[np.float64]      # (N, 3) one vector per rotor
PerAssembly = NDArray[np.float64]         # (K,) one value per tilt hinge
PerSurface = NDArray[np.float64]          # (S,) one value per control surface, in the vehicle's surface order
