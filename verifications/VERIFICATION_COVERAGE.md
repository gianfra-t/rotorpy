### About 

Index for coverage code for verification of the new implementations. This includes: 

- Verification of the new implementations with respect to original `Multirotor` class.
- Verification against external and published sources (papers, reports, etc), where simulations reports on a comparable dynamic model are presented. Real flight test data is difficult to get in the first place, so for this case we can settle with good dynamic simulation cases.
- Cross-check verifications against different implementations of the extended dynamics.
- Conservation and analytical cases verifications.


### Inventory

- [`tests/test_vehicle_equivalence.py`](../tests/test_vehicle_equivalence.py): Consist on creating cases that cover all the "physical expressiveness" of the implementations compared. For example: define a complex, potentially non-symmetric configuration and a defined trajectory, in closed-loop. Simulate for some time. 
The state history should match for every compared class, if it does, the physical model can be considered equivalent. 

We're not checking the controller, or even physical sense of the model, simply that the implementations match. This is important, as we build on top of well-tested models like the existing `Multirotor` class, we should at least cover every case it can represent, that removes the need to go out and check against published data to verify the physical model for these cases.

- [`pei_case1.ipynb`](./pei_case1.ipynb): checks `DrakeMultirotor` against a paper's published results: Pei & Roithmayr, *Equations of Motion for a Generic Multibody Tilt-rotor Aircraft* (AIAA 2022-3511).
