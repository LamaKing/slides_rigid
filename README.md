
<img width="636" alt="Untitled" src="https://user-images.githubusercontent.com/19472018/212551025-70228a51-1591-4c3c-b2cd-9002f92dfb49.png">

# Superlubric interface detector

Compute interlocking potential between a periodic substrate and a finite-size adsorbate, in the rigid approximation.
The adsorbate is treated as a rigid body at a given orientation $\theta$ and center of mass (CM) position $(x_\mathrm{cm},y_\mathrm{cm})$

## Substrate
The substrate is defined as a periodic function resulting from either a monocromaitc superposition of plane waves or a potential well of a given shape repeated in space. 
The functions handling the substrate creation are in ```tool_create_substrate.py```.

For a plane wave superposition, the substrate is defined by a suitable set of wave vectors, where the number of vectors defines the symmetry and length of vectors defines the spacing [VanossiPNAS].

For a lattice of wells, the substrate is defined by the shape parameters of the well and the lattice vectors [CaoNatPhys, CaoPRE, CaoPRX]. This substrate can be decorated with a lattice.

The parameters are specified in a JSON file.
See Example/0-Substrate_types.ipynb for details.

## Cluster
The cluster is defined as a collection of points (optionally decorate with a basis) belonging to a given lattice.
For convenience, there are functions returning clusters in regular shapses, e.g. rectangles, hexagons, circles, etc.
The functions handling the substrate creation are in ```tool_create_substrate.py```.

See example/1-Cluster_creation.ipynb for details.

## Static Maps

See example/2-Cluster_on_substrate.ipynb for details on the following functions.

### Translations
To explore the energy landscape of an adsorbate over a substrate as a function of the CM at fixed orientation, see ```static_trasl_map.py```
#### Rotations
To explore the energy landscape of an adsorbate over a substrate as a function of the imposed rotation $\theta$, at fixed CM, see ```static_roto_map.py```
#### Roto-translations
To search for the global minimum of an adosrbate, one needs to combine rotations and translation, and locate the energy minimum in the $(x_\mathrm{cm}, y_\mathrm{cm}, \theta)$ space. See ```static_rototrasl_map.py``` for details.

## Dynamics Maps

To go beyond rigid maps, there are two essential tools: compute the minimum energy path between two minimum or perform a molecular dynamics calculation under given translational and rotational drives $(F_x, F_y, \tau)$.

### Barrier finding
The barrier between two points in the configurational space $(x_\mathrm{cm}, y_\mathrm{cm}) at fixed orientation can be estimated be the string algorithm [], similar to the NEB methods.
The ideal can be summarised like this: imagine the potential energy to be a hill landscape. Place a string between two points of the landscape and let it relax. The string would relax downhill until the gradient on the string vanishes, i.e. the string layes on the pass between the valleys and below the peaks.

See example/3-Barrier_from_stirng.ipynb

### Molecular dynamics 
The script ```MD_rigid_rototrasl.py``` solve the equation of motion for the center of mass and orientation of the cluster in the overdamped regime (no interial term)
#### Equations of motion
In the overdamped limit, the equation of motion are the following first order equations:
  
$$ \gamma_{t} d\mathbf{r}/dt = (\mathbf{F}_{ext} - grad U) $$

$$ \gamma_{r} d\theta/dt = (\tau_{ext} - dU/d\theta) $$
  
In this picture energy is not conserved (fully dissipated in the Langevin bath between successive timesteps) and the value of the dissipation constants, $\gamma_t$ and $\gamma_r$, effectively sets how quickly the time flows. 
Thus by lowering $\gamma$ one can "speed up" the simulations and match timescales similar to experiments.

## Units
The model can be ragarded as adimensional.

A coherent set of units useful to compare with experimental colloidal system is:
  - energy in zJ
  - length in $\mu\mathrm{m}$
  - mass in fKg

From which follows:
  - force in fN
  - torque in fN $\cdot \mu \mathrm{m}$
  - time in ms
  - translational damping constant $\gamma$ in fKg/ms
