#!/usr/bin/env python3

import os, shutil, json, sys
from os.path import join as pjoin
from time import time
import numpy as np
from MD_rigid_rototrasl import MD_rigid
from misc import handle_run

def F_loop(F0, F1, dF, inputs, thF=0, update_conf=True):
    """Loop over values of F form F0 to F1 in steps dF

    Flag update_conf concatenates position of cm and orientation theta between runs."""

    t0 = time() # Start the clock

    # Get force range from command line
    print("Start F0=%.4g end F1=%.4g step dF=%.4g (%i runs)" % (F0, F1, dF, 1+np.floor((F1-F0)/dF)))
    pwd =  os.environ['PWD']
    print('Working in ', pwd)
    outlast = open('last-F.dat', 'w')

    Tau = inputs['Tau']
    for F in np.arange(F0, F1, dF):
        print('--------- ON F=%15.8g -----------' % F)
        Fx, Fy = np.cos(thF)*F, np.sin(thF)*F
        cdir = handle_run(inputs, ['Fx', 'Fy'], [Fx, Fy], MD_rigid) # for json cannot be numpy
        # Extract last config of current run
        last_step = np.loadtxt(pjoin(cdir, 'out.dat'))[-1]
        print(('%25.15g '*3) % (Tau, Fx, Fy), ''.join(['%25.15g ' % f for f in last_step]), file=outlast)
        if update_conf:
            inputs['pos_cm'], inputs['theta'] = [float(last_step[[2]]), float(last_step[[3]])], float(last_step[6])
        print('-' * 80, '\n')
    outlast.close()
    t1=time()
    print('Done in %is (%.2fmin)' % (t1-t0, (t1-t0)/60))

if __name__ == "__main__":
    # -------- INPUTS --------
    with open(sys.argv[1]) as inj:
        inputs = json.load(inj)

    #F0, F1, dF = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    with open(sys.argv[2]) as inj:
        ranges = json.load(inj)

    F0, F1, dF = 0, 0, 1
    try:
        F0, F1, dF = ranges['F0'], ranges['F1'], ranges['dF']
        if 'thF' in ranges.keys(): thF = ranges['thF']
    except KeyError:
        print('Could not set force range')
        pass
    F_loop(F0, F1, dF, inputs)
