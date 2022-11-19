#!/usr/bin/env python3

import os, shutil, json, sys
from os.path import join as pjoin
from time import time
import numpy as np
from MD_rigid_rototrasl import MD_rigid
from misc import handle_run

def F_loop(F0, F1, dF, inputs, thF=0, update_conf=True):
    """ Loop over values of F form F0 to F1 in steps dF

    Update config concatenates pos cm and theta between runs"""

    t0 = time() # Start the clock

    # Get force range from command line
    print("Start F0=%.4g end F1=%.4g step dF=%.4g (%i runs)" % (F0, F1, dF, 1+np.floor((F1-F0)/dF)))
    pwd =  os.environ['PWD']
    print('Working in ', pwd)

    for F in np.arange(F0, F1, dF):
        print('--------- ON F=%15.8g -----------' % F)
        Fx, Fy = np.cos(thF)*F, np.sin(thF)*F
        cdir = handle_run(inputs, ['Fx', 'Fy'], [Fx, Fy], MD_rigid) # for json cannot be numpy
        if update_conf:
            # Extract last config of current run
            xcm, ycm, theta = np.loadtxt(pjoin(cdir, 'out.dat'))[-1, [2,3,6]]
            print('Last config x,y,th', xcm, ycm, theta)
            inputs['pos_cm'] = [xcm, ycm]
            inputs['theta'] = theta
        print('-' * 80, '\n')

    t1=time()
    print('Done in %is (%.2fmin)' % (t1-t0, (t1-t0)/60))

if __name__ == "__main__":
    # -------- INPUTS --------
    with open(sys.argv[1]) as inj:
        inputs = json.load(inj)

    F0, F1, dF = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    F_loop(F0, F1, dF, inputs)
