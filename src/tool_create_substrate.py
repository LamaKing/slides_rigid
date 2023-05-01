import json
import numpy as np
from numpy import pi, sqrt

# --- calcolates simple matrix for mapping clusters colloids into primitive cell and viceversa.
# Square lattice matrix
def calc_matrices_square(R):
    """Metric matrices of square lattice of spacing R.

    Return 2x2 matrices to map to unit cell (u) and inverse (u_inv), back to real space."""
    area = R*R
    u     = np.array([[1,0], [0,1]])*R/area
    u_inv = np.array([[1,0], [0,1]])*R
    return u, u_inv
# Triangle lattice matrix
def calc_matrices_triangle(R):
    """Metric matrices of triangular lattice of spacing R.

    Return 2x2 matrices to map to unit cell (u) and inverse (u_inv), back to real space."""
    area = R*R*sqrt(3)/2.
    # NN along y
    #u     = np.array([[1,0], [-1./2, sqrt(3)/2]])*R/area
    #u_inv = np.array([[sqrt(3)/2,0], [1/2,1]])*R
    # NN along x (like tool_create_hex/circ)
    u =     np.array([[sqrt(3)/2.,0.5], [0,1]])*R/area
    u_inv = np.array([[1,-0.5],            [0.0, sqrt(3)/2.]])*R
    return u, u_inv
# Arbitraty lattice matrix from primitive vectors
def calc_matrices_bvect(b1, b2):
    """Metric matrices from primitive lattice vectors b1, b2.

    Return 2x2 matrices to map to unit cell (u) and inverse (u_inv), back to real space."""
    St = np.array([b1, b2])
    u = np.linalg.inv(St).T
    u_inv = St.T
    return u, u_inv
# ----------------------

# --- Tanh substrate
def particle_en_tanh(pos, pos_torque, basis, a, b, ww, epsilon, u, u_inv):
    """Calculate energy, forces and torque on each particle.

    Substrate energy is modelled as a lattice of tanh-shaped wells.

    Inputs are:
    - pos: position of particles in rigid cluster, as (N,2) array.
    - pos_torque: reference point (1,2 array) to compute the torque (usually CM).
    - basis: list of position of the substrate basis (N,2 array).

    Well specific inputs (usually passed as *en_input):
    - a, b: beginning and end of tanh well (W=-epsilon for r<a, W=0 for x>b).
    - ww: shape factor for the tanh potential.
    - epsilon: depth of the well.
    - u, u_inv: matrices to map to subsrtate unit cell (see calc_matrices_bvect)

    Returns (for each particle) energy (N array), force (N,2), torque (N)
    """
    en = np.zeros(shape=(pos.shape[0]))
    F = np.zeros(shape=(pos.shape[0],2))
    tau = np.zeros(shape=(pos.shape[0]))

    # For each crystal basis position
    for r in basis:
        # map to substrate cell
        posp = np.dot(u, (pos-r).T).T # Fast numpy dot with different convention on row/cols
        posp -= np.floor(posp + 0.5) # Map back to substrate
        pospp = np.dot(u_inv, posp.T).T # back to real space
        posR = np.linalg.norm(pospp, axis=1) # Radial

        # --> particles in the flat bottom (force and torque are zero)
        en[posR <= a] = -epsilon # energy inside flat bottom region
        # --> particles in the curved region (see X. Cao Phys. Rev. E 103, 1 (2021))
        inside = np.logical_and(posR<b, posR>a) # numpy mask vector
        Rin = posR[inside]
        rho = (Rin-a)/(b-a) # Reduce coordinate rho in [0,1]
        # Energy
        en[inside] += epsilon/2.*(np.tanh((rho-ww)/rho/(1-rho))-1.)
        # Force F = - grad(E)
        ff = (rho-ww)/rho/(1-rho)
        ass = (np.cosh(ff)*(1-rho)*rho)**2
        vecF = -epsilon/2*(rho*rho+ww-2*ww*rho)/ass
        vecF /= (b-a) # Go from rho to r again
        # Project to x and y
        F[inside,0] += vecF*pospp[inside,0]/Rin
        F[inside,1] += vecF*pospp[inside,1]/Rin
        # Torque tau = r vec F (with pos_torque application point)
        tau += np.cross(pos-r-pos_torque, F)
        # --> particles in the flat top have 0 energy (force and torque are zero). So do nothing.

    # Return energy, F and torque
    return en, F, tau

def calc_en_tanh(pos, pos_torque, basis, a, b, ww, epsilon, u, u_inv):
    """Calculate energy, forces and torque on CM.

    Substrate energy is modelled as a lattice of tanh-shaped wells.

    See corresponding particle function for details on parameters.

    Return total energy (scalar) force (2d vector) torque (scalar).
    """
    # This might be very slow. Test a bit.
    en, F, tau = particle_en_tanh(pos, pos_torque, basis, a, b, ww, epsilon, u, u_inv)
    return np.sum(en), np.sum(F, axis=0), np.sum(tau)
# ----------------------

# --- Gaussian substrate
# Non normalised
def gaussian(x, mu, sigma):
    return np.exp(-np.square(x - mu) / (2 * np.square(sigma)))
# Normalised by width (sigma)
#def gaussian(x, mu, sig):
#    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.))) / (2. * np.pi * np.power(sigma, 2.))

def particle_en_gaussian(pos, pos_torque, basis, a, b, sigma, epsilon, u, u_inv):
    """Calculate energy, forces and torque on each particle.

    Substrate energy is modelled as a lattice of gaussian-shaped wells.

    Inputs are:
    - pos: position of particles in rigid cluster, as (N,2) array.
    - pos_torque: reference point (1,2 array) to compute the torque (usually CM).
    - basis: list of position of the substrate basis (N,2 array).

    Well specific inputs (usually passed as *en_input):
    - a, b: beginning and end of tempered region (W=0 for x>b).
    - sigma: width of the gaussian.
    - epsilon: depth of the well.
    - u, u_inv: matrices to map to subsrtate unit cell (see calc_matrices_bvect)

    Returns (for each particle) energy (N array), force (N,2), torque (N)
    """
    en = np.zeros(shape=(pos.shape[0]))
    F = np.zeros(shape=(pos.shape[0],2))
    tau = np.zeros(shape=(pos.shape[0]))

    for r in basis:
        posp = np.dot(u, (pos-r).T).T # Fast numpy dot with different convention on row/cols
        posp -= np.floor(posp + 0.5) # Map back to substrate
        pospp = np.dot(u_inv, posp.T).T # back to real space
        posR = np.linalg.norm(pospp, axis=1) # Radial

        bulk, tail = posR<=a, np.logical_and(posR>a, posR<b) # Mask positions: bulk no damping, tail damped

        Rtail = posR[tail] # position in tail
        rho = (Rtail-a)/(b-a) # Reduce coordinate rho in [0,1]
        ftail = (1 - 10*rho**3 + 15*rho**4 - 6*rho**5)  # Damping f -> [1,0]
        dftail = (-30*rho**2 + 60*rho**3 - 30*rho**4)/(b-a) # Derivative of f

        # Energy
        en[bulk] += -epsilon*gaussian(posR[bulk], 0, sigma) # Bulk
        en[tail] += -epsilon*gaussian(Rtail, 0, sigma)*ftail # Tail
        # Forces bulk F = -dE/dr
        bulk = np.logical_and(posR<=a, posR != 0) # Exclude singular point in origin where F=0
        F[bulk, 0] = -epsilon*gaussian(posR[bulk],0,sigma) * (posR[bulk]/np.power(sigma, 2.))*pospp[bulk,0]/posR[bulk]
        F[bulk, 1] = -epsilon*gaussian(posR[bulk],0,sigma) * (posR[bulk]/np.power(sigma, 2.))*pospp[bulk,1]/posR[bulk]
        # Forces tail F = -d(E*f)/dr = - (E'*f + E*f')
        f1 = epsilon*gaussian(Rtail, 0, sigma)*dftail # E f'
        f2 = -ftail*epsilon*gaussian(Rtail,0,sigma) * (Rtail / np.power(sigma, 2.)) # E' f
        F[tail, 0] += (f1+f2) * pospp[tail,0]/posR[tail]
        F[tail, 1] += (f1+f2) * pospp[tail,1]/posR[tail]
        # Torque
        tau += np.cross(pos-r-pos_torque, F)

    return en, F, tau

def calc_en_gaussian(pos, pos_torque, basis, a, b, sigma, epsilon, u, u_inv):
    """Calculate energy, forces and torque on CM.

    See corresponding particle function for details on parameters.

    Return total energy (scalar) force (2d vector) torque (scalar).
    """
    en, F, tau = particle_en_gaussian(pos, pos_torque, basis, a, b, sigma, epsilon, u, u_inv)
    return np.sum(en), np.sum(F, axis=0), np.sum(tau)
# ----------------------

# --- Sinusoidal arbitrary substrate
# !! This is implemented a bit differently: we do not need information about the unit cell, and  it might not even exists, in the quasi-crystal case.
# Coefficients for reciprical space
# from Vanossi, Manini, Tosatti www.pnas.org/cgi/doi/10.1073/pnas.1213930109 PNAS∣October 9, 2012∣vol. 109∣no. 41∣16429–16433
# n, c_n, alpha_n = 2, 1, 0 # Lines
# n, c_n, alpha_n = 3, 4/3, 0 # Tri
# n, c_n, alpha_n = 4, np.sqrt(2), pi/4 # Square
# n, c_n, alpha_n = 5, 2, 0 # Qausi-cristal
# n, c_n, alpha_n = 6, 4/np.sqrt(3), -pi/6
def get_ks(R, n, c_n, alpha_n):
    """Compute wave vectors k of interfering plane waves"""
    return np.array([c_n*pi/R*np.array([np.cos(2*pi/n*l+alpha_n), np.sin(2*pi/n*l+alpha_n)])
                     for l in range(n)])

def particle_en_sin(pos, pos_torque, basis, ks, epsilon):
    """Calculate energy, forces and torque on each particle.

    Substrate energy is modelled as sum of plane waves defined by the reciprocal vecotrs ks.


    Inputs are:
    - pos: position of particles in rigid cluster, as (N,2) array.
    - pos_torque: reference point (1,2 array) to compute the torque (usually CM).
    - basis: list of position of the substrate basis (N,2 array).

    Well specific inputs (usually passed as *en_input):
    - ks: list of wavevector generating the potential.
    - epsilon: depth of the potential.

    Returns (for each particle) energy (N array), force (N,2), torque (N)
    """

    F = np.zeros(shape=(pos.shape[0],2))
    en = np.zeros(pos.shape[0])
    tau = np.zeros(pos.shape[0])
    n = len(ks) # Number of plane waves
    for r in basis:
        x, y = (pos-r)[:,0], (pos-r)[:,1]
        # energy
        exp_list = np.array([np.exp(1j*(x*k[0]+y*k[1]))
                             for k in ks])
        en += -epsilon/n**2*np.abs(np.sum(exp_list, axis=0))**2
        # force
        t1  = np.sum(np.array([np.cos(x*k[0]+y*k[1]) for k in ks]), axis=0)
        t2x = np.sum(np.array([k[0]*np.sin(x*k[0]+y*k[1]) for k in ks]), axis=0)
        t2y = np.sum(np.array([k[1]*np.sin(x*k[0]+y*k[1]) for k in ks]), axis=0)
        t3  = np.sum(np.array([np.sin(x*k[0]+y*k[1]) for k in ks]), axis=0)
        t4x = np.sum(np.array([k[0]*np.cos(x*k[0]+y*k[1]) for k in ks]), axis=0)
        t4y = np.sum(np.array([k[1]*np.cos(x*k[0]+y*k[1]) for k in ks]), axis=0)
        F[:,0] += -epsilon*2/n**2*(t1*t2x-t3*t4x)
        F[:,1] += -epsilon*2/n**2*(t1*t2y-t3*t4y)
        # torque
        tau += np.cross(pos-r-pos_torque, F)

    return en, F, tau

def calc_en_sin(pos, pos_torque, basis, ks, epsilon):
    """Calculate energy, forces and torque on CM.

    Substrate energy is modelled as sum of plane waves defined by the reciprocal vecotrs ks.

    See corresponding particle function for details on parameters.

    Return total energy (scalar) force (2d vector) torque (scalar).
    """
    en, F, tau = particle_en_sin(pos, pos_torque, basis, ks, epsilon)
    return np.sum(en), np.sum(F, axis=0), np.sum(tau)

# ----------------------

# --- Sinusoidal triangular substrate
# Shortcut for the one we use the most
def particle_en_sin_tri(pos, pos_torque, basis, R, epsilon, u, u_inv):
    """Calculate energy, forces and torque on each particle.

    !!! Coefficients are for triangular lattice only !!!
    Substrate energy is modelled as sum of three plane waves."""
    F = np.zeros(shape=(pos.shape[0],2))
    en = np.zeros(pos.shape[0])
    tau = np.zeros(pos.shape[0])

    for r in basis:
        # map to substrate cell
        posp = np.dot(u, (pos-r).T).T
        posp -= np.floor(posp + 0.5)
        pospp = np.dot(u_inv, posp.T).T
        x, y = pospp[:,0], pospp[:,1]

        # energy
        txy = np.cos((2*pi)/(sqrt(3)*R)*Y)*np.cos((2*pi)/R*X)
        ty = np.cos((4*pi)/(sqrt(3)*R)*Y)
        en = epsilon*-1/9*(3+4*txy+2*ty)
        # force
        fx = 2*pi/R
        fy = 2*pi/(R*sqrt(3))
        pre = -epsilon*8*pi/(9*R)
        F[:,0] = pre*np.cos(fy*Y)*np.sin(fx*X)
        F[:,1] = pre/sqrt(3)*np.sin(fy*Y)*(np.cos(fx*X)+2*np.cos(fy*Y))
        # torque
        tau = np.cross(pos-r-pos_torque, F)

    return en, F, tau

def calc_en_sin_tri(pos, pos_torque, basis, a, epsilon, u, u_inv):
    """Calculate energy, forces and torque on CM.

    !!! Coefficients are for triangular lattice only !!!
    Substrate energy is modelled as sum of three plane waves."""
    en, F, tau = particle_en_sin_tri(pos, pos_torque, basis, a, epsilon, u, u_inv)
    return np.sum(en), np.sum(F, axis=0), np.sum(tau)
# ----------------------

# --- Initialise the energy function from parameter file
def substrate_from_params(params):
    """Initialise a substrate from a parameter dictionary (usually from JSON file)

    Returns the functions computing the particle-wise and total energy, and the list of potential-specific paramters to pass to these functions.
    """


    # Read params
    epsilon, well_shape =  params['epsilon'], params['well_shape']
    basis = params['sub_basis']
    # define well shape
    if well_shape == 'gaussian':
        pen_func, en_func = particle_en_gaussian, calc_en_gaussian
        sigma, a, b = params['sigma'], params['a'], params['b']
        u, u_inv = calc_matrices_bvect(params['b1'], params['b2'])
        en_inputs = [basis, a, b, sigma, epsilon, u, u_inv]
    elif well_shape == 'tanh':
        pen_func, en_func = particle_en_tanh, calc_en_tanh
        wd, a, b = params['wd'], params['a'], params['b']
        u, u_inv = calc_matrices_bvect(params['b1'], params['b2'])
        en_inputs = [basis, a, b, wd, epsilon, u, u_inv]
    elif well_shape == 'sin':
        pen_func, en_func = particle_en_sin, calc_en_sin
        ks = params['ks']
        en_inputs = [basis, ks, epsilon]
    else:
        raise NotImplementedError("Well shape %s not implemented" % well_shape)

    return pen_func, en_func, en_inputs
