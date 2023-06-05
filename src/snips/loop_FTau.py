#!/usr/bin/env python3

import os, shutil, json, sys, argparse
from argparse import RawTextHelpFormatter
from os.path import join as pjoin
from time import time
import numpy as np
from MD_rigid_rototrasl import MD_rigid
from misc import handle_run

def FTau_loop(F0, F1, dF,
              Tau0, Tau1, dTau,
              inputs, thF=0):
    """Loop over two ranges (Tau0,Tau1,dTau) (F0,F1,dF) with F at angle thF (default=0)

    Update config concatenates pos cm and theta between runs in inner loop (forces)"""

    t0 = time() # Start the clock

    print("Start Tau0=%.4g end Tau1=%.4g step dTau=%.4g (%i runs)" % (Tau0, Tau1, dTau, 1+np.floor((Tau1-Tau0)/dTau)))
    print("Start F0=%.4g end F1=%.4g step dF=%.4g (%i runs)" % (F0, F1, dF, 1+np.floor((F1-F0)/dF)))
    pwd =  os.environ['PWD']
    print('Working in ', pwd)

    rcm0, theta0 = inputs['pos_cm'], inputs['theta']
    update_conf = False
    outlast = open('last-FTau.dat', 'w')
    for Tau in np.arange(Tau0, Tau1, dTau):
        for F in np.arange(F0, F1, dF):
            print('--------- ON Tau,F=%15.8g%15.8g -----------' % (Tau,F))
            Fx, Fy = np.cos(thF)*F, np.sin(thF)*F
            cdir = handle_run(inputs, ['Tau', 'Fx', 'Fy'], [float(Tau), float(Fx), float(Fy)], MD_rigid) # for json cannot be numpy
            # Extract last config of current run
            last_step = np.loadtxt(pjoin(cdir, 'out.dat'))[-1]
            print(('%25.15g '*3) % (Tau, Fx, Fy), ''.join(['%25.15g ' % f for f in last_step]), file=outlast)
            if update_conf:
                inputs['pos_cm'], inputs['theta'] = [float(last_step[[2]]), float(last_step[[3]])], float(last_step[6])
            print('-' * 80, '\n')
        # Reset for outer loop
        inputs['pos_cm'] = [float(rcm0[0]), float(rcm0[1])]
        inputs['theta'] = float(theta0)
    outlast.close()

    t1=time()
    print('Done in %is (%.2fmin)' % (t1-t0, (t1-t0)/60))

if __name__ == "__main__":
    #-------------------------------------------------------------------------------
    # Argument parser
    #-------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="""Snippet to run a loop over forces and torques for a given cluster""")

    # Optional args
    parser.add_argument('--input', '-i',
                        dest='inputf', type=str, default=False,
                        help='filename of JSON with system parameters')
    parser.add_argument('--ranges', '-r',
                        dest='rangesf', type=str, required=True,
                        help='filename of JSON with ranges of force and torque')
    parser.add_argument('--debug',
                        action='store_true', dest='debug',
                        help='show debug informations.')

    #-------------------------------------------------------------------------------
    # Initialize and check variables
    #-------------------------------------------------------------------------------
    args = parser.parse_args(sys.argv[1:])

    # -------- INPUTS --------
    with open(args.inputf) as inj:
        inputs = json.load(inj)

    with open(args.rangesf) as inj:
        ranges = json.load(inj)

    thF = 0
    F0, F1, dF = 0, 0, 1
    try:
        F0, F1, dF = ranges['F0'], ranges['F1'], ranges['dF']
        if 'thF' in ranges.keys(): thF = ranges['thF']
    except KeyError:
        print('Could not set force range')
        pass
    Tau0, Tau1, dTau = 0, 0, 1
    try:
        Tau0, Tau1, dTau = ranges['Tau0'], ranges['Tau1'], ranges['dTau']
    except KeyError:
        print('Could not set torque range')
        pass

    FTau_loop(F0, F1, dF, Tau0, Tau1, dTau, inputs)
