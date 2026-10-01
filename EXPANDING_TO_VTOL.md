### Goal

Ultimate intention is to add a full model for arbitrary VTOL aircraft configuration. This includes:
- Support for asymmetric rotor definitions: Larger rotors in the main wing, smaller in the tail, for example. Each with it's own dynamics and forces.
- Allow arbitrary thrust direction of each rotor, not just body's z. So it is possible to model tilted propeller/rotors configurations
- Support aerodynamic surfaces, which unlocks true vtol as well as lifting surfaces for multirotors for efficiency. Note: the intention is NOT to delve deep into rotor-wing interactions, simply support any model with a clear interface that exposes the state of the vehicle and requires aerodynamic forces and moments. 
- Allow arbitrary tilting nacelle + rotor configurations, with support for moving CG as a fx() of tilt angle, also new control parameter. This will introduce complex dynamics and is the core change.

### Strategy

Done in stages:
- First 2 points, asymmetric rotors and freely defined rotor axis, in a first pass. This is class `multirotor_extended.py`. It has the least amounts of changes to the model, and reuses most of the formulation from the original code.
- Second stage ended up being the complete model for a tiltrotor: Chosen strategy for dynamics is using a multibody solver (`Drake`) that makes it easier
and avoids mistakes while modelling angular momentum effects of heavy, tilting rotors and center of mass changes of non-triviial nacelle masses. This `vehicle` definition now supports any arbitrary model of rotor and aerodynamic forces (wrench) physics, that is compatible with modeling rotor wake interaction.
For that, an aerodynamic model for rotor-lifting surface interaction well understood was chosen ([`PoweredAeroModel`](rotorpy/vehicles/models/powered_aero/model.py), see [INTERACTION_AERODYNAMICS_MODEL.md](docs/INTERACTION_AERODYNAMICS_MODEL.md)), and an extendel ROM of the rotors ([`rotor.py`](rotorpy/vehicles/models/powered_aero/rotor.py)) was also expanded from the exiting default one, that allows to model forces from rotors at higher speeds and arbitrary inflow angles. 


### Interface Strategy

Make all this changes compatible with the rest of the code, meaning, mostly `step()` and all the involved physics is modified. Do not touch the rest of the modules, except for additions to the control code such that it supports aerodynamic surfaces and tilt control variables. 

### Physics modularity on dynamics.

The plan was to have this compartimentalized enough that, the core physics (dynamics "engine") could be swapped easily. Implementation paths (so far) are:
- "Hand derivation" extension, Newton-Euler formulation. 
- Using Kane's method for describing and solving the dynamics.
- Using a multibody solver (pydrake)

But after performance tests the multibody solver was deemed to be a beter choice to adopt. There is however an example of a "hand derivation" with prescribed tilt angle coming soon, mostly for learning purposes.

### Verification

Connecting with the previous point, cross-validation between different method of describing the dynamics is a good way to verify it is properly implemented.

Besides this, the intention is to use some conservation checks to verify the model is properly implemented. Conservation working as a sort of "unit tests".

### Separation of concerns

As [INTERACTION_AERODYNAMICS_MODEL.md](docs/INTERACTION_AERODYNAMICS_MODEL.md) further explaines, the separation must be clear and easy to swap between different techinques and support many vehicles (in the future). In essence the interfaces between modules must be clear:
- Aerodynamic reduced order model, what runs on the simulation at each step, 
- Dynamics: the engine that provides state and asks for forces, regardless of how they are sovled.
- Model fitting: the model must be expandable and representative enough to support both many diverse physical elements and agnostic to where the data is taken from (mid fidelity simulations, CFD, flight tests).  