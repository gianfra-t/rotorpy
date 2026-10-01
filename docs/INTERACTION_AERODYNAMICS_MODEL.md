
As explained in other parts of the docs, the dynamics "engine" (pydrake) is used in such a way that it delegates the rotor and aerodynamic forces
calculation to any defined function. 

The contract is simple: solver provides dynamic state (speeds, attitude, airflow v, etc), the function must provide rotor forces and moments (per rotor), and full aircraft forces and moments. Idea being, the aircraft forces can use the rotor wake to correct for aerodynamic interactions.

### Strategy taken

So the strategy taken is only one of many possible implementation of aerodynamic forces calculation. This is [`PoweredAeroModel`](../rotorpy/vehicles/models/powered_aero/model.py).

First, the typical aerodynamic wrench given derivatives and coefficients (clean airframe) can be supported easily ([`fixed_wing.py`](../rotorpy/vehicles/models/powered_aero/fixed_wing.py)). So it must be able to replicate any sort 
of case used to validate 6-DOF simulators. Mostly taken from Stevens, Lewis [1], this is the traditional derivation.

Second, the rotor model is expanded (see [`rotor.py`](../rotorpy/vehicles/models/powered_aero/rotor.py)), which essentially is a polynomial function of axial and edge flow components fitted from simulations or data [2, 3]. Original rotorpy model was enough for low speeds, but couldn't represent well forces at higher speeds and way off-angle (very required for vtol)

The third component is a correction added to the "clean airframe" forces and moments, as a function of the rotor's wake and lifting surface area touched by it [4, 5, 6] ([`interaction.py`](../rotorpy/vehicles/models/powered_aero/interaction.py)). It produces a "delta" force over the standard, clean airframe forces and uses known interaction coefficients [6, 7] with the possibility (and expectation) that the coefficients can be tuned using VPM or CFD simulations of rotor on wing interaction. 
Doesn't model rotor on rotor, or wing -rotor interactions. Althoguh wing - tail will somewhat be included in the global aircraft coefficients.

### References

1. Stevens, B. L.; Lewis, F. L.; Johnson, E. N. *Aircraft Control and Simulation*, 3rd ed. Wiley, 2015. ISBN 978-1-118-87098-3. Ch. 2, the F-16 model: coefficient build-up, control and rate derivatives.
2. Simmons, B. M. "System Identification for Propellers at High Incidence Angles." *Journal of Aircraft*, 2021 (AIAA 2021-1190). NASA NTRS 20210024634, https://ntrs.nasa.gov/citations/20210024634. Rotor loads as polynomials in the axial and in-plane advance ratios (Eqs. 29–30); identified terms (Tables 4–5).
3. Simmons, B. M.; Buning, P. G.; Murphy, P. C. "Full-Envelope Aero-Propulsive Model Identification for Lift+Cruise Aircraft Using Computational Experiments." AIAA Aviation 2021. NASA NTRS 20210017459. Dimensional thrust form in $n^2$, $nV$, $V^2$ (Eqs. 15–16).
4. Leishman, J. G. *Principles of Helicopter Aerodynamics*, 2nd ed. Cambridge University Press, 2006. ISBN 978-0-521-85860-1. Glauert forward-flight momentum, mean induced velocity $v_0$ (Ch. 2).
5. Chauhan, S. S.; Martins, J. R. R. A. "Tilt-Wing eVTOL Takeoff Trajectory Optimization." *Journal of Aircraft* 57(1):93–112, 2020. DOI 10.2514/1.C035476. Glauert induced velocity (Eq. 20).
6. Appleton, W.; Filippone, A.; Bojdo, N. "Interaction effects on the conversion corridor of tiltrotor aircraft." *The Aeronautical Journal* 125(1294), 2021. DOI 10.1017/aer.2021.33. Wake skew $\chi$ (Eq. 10); rigid cylindrical wake along the skewed centreline, immersion test and contracted radius $R_w$ (Eqs. 11–12), projected onto each lifting surface for the immersed area; wake speed $1.60\,v_0$ added to the freestream, the GTRS value (p. 10).
7. Harendra, P. B.; Joglekar, M. J.; Gaffey, T. M.; Marr, R. L. *V/STOL Tilt Rotor Study, Vol. V: A Mathematical Model for Real Time Flight Simulation of the Bell Model 301 Tilt Rotor Research Aircraft*. NASA CR-114614, 1973. Calibrating the wake constants from powered on/off data (§II.D.1–2).
