### Goal

Ultimate intention is to add a full model for arbitrary VTOL aircraft configuration. This includes:
- Support for asymmetric rotor definitions: Larger rotors in the main wing, smaller in the tail, for example. Each with it's own dynamics and forces.
- Allow arbitrary thrust direction of each rotor, not just body's z. So it is possible to model tilted propeller/rotors configurations
- Support aerodynamic surfaces, which unlocks true vtol as well as lifting surfaces for multirotors for efficiency. Note: the intention is NOT to delve deep into rotor-wing interactions, simply support any model with a clear interface that exposes the state of the vehicle and requires aerodynamic forces and moments. 
- Allow arbitrary tilting nacelle + rotor configurations, with support for moving CG as a fx() of tilt angle, also new control parameter. This will introduce complex dynamics and is the core change.

### Strategy

Go in stages:
- First 2 points, asymmetric rotors and freely defined rotor axis, in a first pass.
- Second pass may involve the core work, support for enhanced dynamics that support tiltrotors.
- Third, expanding the wrench interface to support lifting surfaces.

Last 2 can be added and tested in parallel, probably. 

### Interface Strategy

Make all this changes compatible with the rest of the code, meaning, mostly `step()` and all the involved physics is modified. Do not touch the rest of the modules, except for additions to the control code such that it supports aerodynamic surfaces and tilt control variables. 

### Physics modularity

Since the core dynamics modelling could be implemented in many ways, techniques and formulation, the plan is to have this compartimentalized enough that, the core physics can be swapped easily. Implementation paths (so far) are:
- "Hand derivation" extension, Newton-Euler formulation. 
- Using Kane's method for describing and solving the dynamics.
- Using a multibody solver (pydrake is in the aim, as of now)

### Verification

Connecting with the previous point, cross-validation between different method of describing the dynamics is a good way to verify it is properly implemented.

Besides this, the intention is to use some conservation checks to verify the model is properly implemented. Conservation working as a sort of "unit tests".