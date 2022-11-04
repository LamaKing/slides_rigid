#!/usr/bin/env python

import sys, os, json, logging, multiprocessing
import numpy as np
from time import time
from functools import partial
from tool_create_cluster import cluster_from_params, rotate
from tool_create_substrate import substrate_from_params, calc_matrices_bvect
from string_method import PotentialPathAnalyt, Path

def static_barrier(pos, inputs, calc_en_f, name=None, log_propagate=True, debug=False):

    N = pos.shape[0]
    if name == None:
        name = 'sting_N_%i' % N
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
        with open(inputs) as inj:
            inputs = json.load(inj)
    else:
        inputs = inputs.copy() # Copy it so multiple threads can access it
    c_log.debug("Input dict \n%s", "\n".join(["%10s: %10s" % (k, str(v)) for k, v in inputs.items()]))

    en_params = inputs['en_params']
    Nsteps = 3000 # Relax for these many steps
    if 'Nsteps' in inputs.keys(): Nsteps = inputs['Nsteps']
    Npt = 100 # number of subdivisions of the path connecting a and b
    if 'Npt' in inputs.keys(): Npt = inputs['Npt']
    L = Path(inputs['p0'], inputs['p1'], pos, Npt, fix_ends=False)               # initalise the path
    V = PotentialPathAnalyt(L, calc_en_f, en_params)      # potential along the path
    c_log.info("Relax string of %i points in %i stesp" % (Npt, Nsteps))
    data = np.zeros((Npt, 6))

    #-------- OUTPUT SETUP -----------
    print_skip = 1 # Timesteps to skip between prints
    if 'print_skip' in inputs.keys(): print_skip = inputs['print_skip']
    printerr_skip = int(Nsteps/10)
    outstream = open(out_fname, 'w')
    # !! Labels and print_status data structures must be coherent !!
    num_space, indlab_space = 35, 2 # Width printed numerical values, Header index width
    lab_space = num_space-indlab_space-1 # Match width of printed number, including parenthesis
    header_labels = ['ly', 'e_pot', 'forces[0]', 'forces[1]', 'torque']
    # Gnuplot-compatible (leading #) fix-width output file
    first = '#{i:0{ni}d}){s: <{n}}'.format(i=0, s='lx', ni=indlab_space, n=lab_space-1,c=' ')
    print(first+"".join(['{i:0{ni}d}){s: <{n}}'.format(i=il+1, s=lab, ni=indlab_space, n=lab_space,c=' ')
                        for il, lab in zip(range(len(header_labels)), header_labels)]), file=outstream)

    # Inner-scope shortcut for printing
    def print_status():
        data = [lx, ly, e_pot, forces[0], forces[1], torque]
        print("".join(['{n:<{nn}.16g}'.format(n=val, nn=num_space)
                       for val in data]), file=outstream)

    #-------- STRING MINIMISATION ----------------
    t0 = time() # Start clock
    c_log.propagate = False
    infoskip = 20
    for i in range(Nsteps):
        if i % int(Nsteps/infoskip) == 0: c_log.info("On it %i (%.2f%%)" % (i, i/Nsteps*100))
        L.eulerArc(V, dt=1e-6)
        V.update(L)

    for it, (lx, ly) in enumerate(zip(L.x, L.y)):
        e_pot, forces, torque = calc_en_f(pos+[lx, ly], [lx, ly], *en_params)
        data[it] = [lx, ly, e_pot, forces[0], forces[1], torque]
        print_status()
    outstream.close()

    c_log.propagate = True
    t_exec = time() - t0
    c_log.info("Done in %is (%imin)" % (t_exec, t_exec/60))
    return data, L, V

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
    wrap_func = partial(static_barrier, inputs=inputs, calc_en_f=calc_en_f, log_propagate=False, debug=debug)

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
