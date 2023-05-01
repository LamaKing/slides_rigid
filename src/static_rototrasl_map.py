#!/usr/bin/env python

import sys, os, json, logging, multiprocessing
import numpy as np
from time import time
from functools import partial
from tool_create_cluster import cluster_from_params, rotate
from tool_create_substrate import substrate_from_params, calc_matrices_bvect

def static_rototraslmap(pos, inputs, calc_en_f, name=None, log_propagate=True, debug=False):
    """Compute the energy of a rigid cluster as a function of CM position and orientation theta.

    Inputs contains the details of the system, along with the unit cell of the substrate (S) and the fractional range of nbin translation to explore along each lattice direction (S[0], S[1]), and range of orientations.

    Return a Nx7 array with theta, xcm, ycm, e_pot, forces[0], forces[1], torque"""
    N = pos.shape[0]
    if name == None:
        name = 'rototraslmap_N_%i' % N
        out_fname = '/dev/null' # if no name is given, do not write the results to file, just return them.
    else:
        out_fname = '%s.dat' % name

    #-------- SET UP LOGGER -------------
    # For this threads and children
    c_log = logging.getLogger(name)
    c_log.setLevel(logging.INFO)
    if debug: c_log.setLevel(logging.DEBUG)
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    log_format = logging.Formatter('[%(levelname)5s - %(funcName)10s] %(message)s')
    if not log_propagate:
        console = open('console-%s.log' % name, 'w')
        handler = logging.StreamHandler(console)
        handler.setFormatter(log_format)
        c_log.addHandler(handler)

    #-------- READ INPUTS -------------
    if type(inputs) == str: # Inputs passed as path to json file
        with open(inputs) as inj: inputs = json.load(inj)
    else:
        inputs = inputs.copy() # Copy it so multiple threads can access it
    c_log.debug("Input dict \n%s", "\n".join(["%10s: %10s" % (k, str(v)) for k, v in inputs.items()]))

    # Init vectors
    # roto
    theta0, theta1, Nsteps_roto = inputs['theta0'], inputs['theta1'], inputs['ntheta'] # [deg]
    dtheta = (theta1-theta0)/Nsteps_roto
    theta_range = np.linspace(theta0, theta1, Nsteps_roto)
    c_log.info("Running %i steps (th0=%.3g th1=%.3g dtheta=%.3g)" % (Nsteps_roto, theta0, theta1, dtheta))
    # trasl
    u_inv = np.array(inputs['S']).T
    nbin = inputs['nbin'] # Reduce coordinate fraction
    Nsteps_trasl = nbin**2
    da11, da12 = 0, 1 # Default map only one unit cell
    if 'da11' in inputs.keys(): da11 = inputs['da11']
    if 'da12' in inputs.keys(): da12 = inputs['da12']
    da21, da22 = 0, 1
    if 'da21' in inputs.keys(): da21 = inputs['da21']
    if 'da22' in inputs.keys(): da22 = inputs['da22']
    c_log.info("Running %i steps (nbin=%i)" % (Nsteps_trasl, nbin))
    c_log.info("Repeting along a1 from %.2g to %.2g" % (da11, da12))
    c_log.info("Repeting along a2 from %.2g to %.2g" % (da21, da22))
    # Combine
    Nsteps = Nsteps_roto*Nsteps_trasl
    c_log.info("Running total of %i steps" % (Nsteps))
    data = np.zeros((Nsteps, 7))
    en_params = inputs['en_params']

    #-------- OUTPUT SETUP -----------
    print_skip = 1 # Timesteps to skip between prints
    if 'print_skip' in inputs.keys(): print_skip = inputs['print_skip']
    printerr_skip = int(Nsteps/10)
    outstream = open(out_fname, 'w')
    # !! Labels and print_status data structures must be coherent !!
    num_space, indlab_space = 35, 2 # Width printed numerical values, Header index width
    lab_space = num_space-indlab_space-1 # Match width of printed number, including parenthesis
    header_labels = ['pos_cm[0]', 'pos_cm[1]', 'e_pot', 'forces[0]', 'forces[1]', 'torque']
    # Gnuplot-compatible (leading #) fix-width output file
    first = '#{i:0{ni}d}){s: <{n}}'.format(i=0, s='theta', ni=indlab_space, n=lab_space-1,c=' ')
    print(first+"".join(['{i:0{ni}d}){s: <{n}}'.format(i=il+1, s=lab, ni=indlab_space, n=lab_space,c=' ')
                        for il, lab in zip(range(len(header_labels)), header_labels)]), file=outstream)

    # Inner-scope shortcut for printing
    def print_status():
        data = [theta, pos_cm[0], pos_cm[1], e_pot, forces[0], forces[1], torque]
        print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space)
                       for val in data]), file=outstream)

    #-------- ENERGY ROTO-TRASL MAP ----------------
    t0 = time() # Start clock
    c_log.propagate = False
    it = 0
    pos0 = pos.copy()
    for theta in theta_range:
        pos = rotate(pos0, theta) # Rotate at starting angle
        for i1, dda1 in enumerate(np.linspace(da11, da12, nbin, endpoint=True)):
            for i2, dda2 in enumerate(np.linspace(da21, da22, nbin, endpoint=True)):
                pos_cm =  np.dot([dda1, dda2], u_inv) # Go from relative space to real space
                e_pot, forces, torque = calc_en_f(pos + pos_cm, pos_cm, *en_params)

                # Print progress
                if it % printerr_skip == 0:
                    c_log.info("it=%10.4g of %.4g (%2i%%) th=%9.3g x=%9.3g y=%9.3g en=%10.6g" % (it, Nsteps, 100.*it/Nsteps, theta, pos_cm[0], pos_cm[1], e_pot))
                # Assign step results
                data[it] = [theta, pos_cm[0], pos_cm[1], e_pot, forces[0], forces[1], torque]
                # Print step results
                if it % print_skip == 0: print_status()

                it += 1
    #-----------------------------
    outstream.close()

    c_log.propagate = True
    t_exec = time() - t0 # Stop clock
    c_log.info("Done in %is (%.2fmin or %.2fh). Speed %5.3f s/step" % (t_exec, t_exec/60, t_exec/3600, t_exec/Nsteps))

    return data

if __name__ == "__main__":
    t0 = time()
    debug = False

    # -------- SET UP LOGGER -------------
    c_log = logging.getLogger('driver') # Set name identifying the logger.
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    logging.basicConfig(format='[%(levelname)5s - %(name)15s] %(message)s')
    c_log.setLevel(logging.INFO)
    if debug: c_log.setLevel(logging.DEBUG)

    # -------- INPUTS --------
    with open(sys.argv[1]) as inj:
        inputs = json.load(inj)

    N0, N1, dN = int(sys.argv[2]), int(sys.argv[3]), 1
    if len(sys.argv) > 4: dN = int(sys.argv[4])
    c_log.info("From Nl %i to %i steps %i" % (N0, N1, dN))

    # -------- SUBSTRATE ENERGY --------
    _, calc_en_f, en_params = substrate_from_params(inputs)
    inputs['en_params'] = en_params
    if inputs['well_shape'] == 'sin': inputs['S'] = [[1,0], [0,1]] # Might not be define, just use cartesian
    else: _, inputs['S'] = calc_matrices_bvect(inputs['b1'], inputs['b2'])
    c_log.info("%s sub parms: " % inputs['well_shape'] + " ".join(["%s" % str(i) for i in en_params]))

    # ------ CLUSTER ----------
    Nl_range = range(N0, N1, dN) # Sizes to explore, in parallels
    pos_vector = []
    c_log.info("%s cluster parms: " % inputs['cluster_shape'] + " ".join(["%s" % str(i) for i in [inputs['a1'], inputs['a2'], inputs['cl_basis']]]))
    for Nl in Nl_range:
        inputs['N1'], inputs['N2'] = Nl, Nl
        pos = cluster_from_params(inputs)
        pos_vector.append(pos.copy()) # Just to be sure, pass a copy
        c_log.info("cluster size %i (Nl %i)" % (pos.shape[0], Nl))

    # ------------ MULTIPROCESS POOL ---------
    # Set up system for multiprocess
    ncpu, nworkers = os.cpu_count(), 1
    if len(sys.argv) > 5: nworkers = int(sys.argv[5])
    c_log.info("Running %i elements on %i processes (%i cores machine)" % (len(Nl_range), nworkers, ncpu))

    # Fix the all arguments a part from Nl, so that we can use pool.map
    wrap_func = partial(static_rototraslmap, inputs=inputs, calc_en_f=calc_en_f, log_propagate=False, debug=debug)

    # Launch all simulations with Pool of workers
    c_log.debug("Starting pool")
    pool = multiprocessing.Pool(processes=nworkers)
    results = pool.map(wrap_func, pos_vector)
    pool.close() # Pool doesn't accept new jobs
    pool.join() # Wait for all processes to finish
    c_log.debug("Results: %s" % str(results))

    # Print execution time
    t_exec = time()-t0
    c_log.info("Done in %is (%.2fmin, %.2fh)" % (t_exec, t_exec/60, t_exec/3600))
