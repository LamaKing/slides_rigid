#!/usr/bin/env python3

import os, shutil, json, sys, argparse
from argparse import RawTextHelpFormatter
from os.path import join as pjoin
from time import time
import numpy as np
from MD_rigid_rototrasl import MD_rigid
from misc import handle_run

def Tau_loop(Tau0, Tau1, dTau, inputs, update_conf=True):
    """ Loop over values of Tau form Tau0 to Tau1 in steps dTau

    Update config concatenates pos cm and theta between runs"""

    t0 = time() # Start the clock

    # Get force range from command line
    print("Start Tau0=%.4g end Tau1=%.4g step dTau=%.4g (%i runs)" % (Tau0, Tau1, dTau, 1+np.floor((Tau1-Tau0)/dTau)))
    pwd =  os.environ['PWD']
    print('Working in ', pwd)

    Fx, Fy = inputs['Fx'], inputs['Fy']
    outlast = open('last-Tau.dat', 'w')
    for Tau in np.arange(Tau0, Tau1, dTau):
        print('--------- ON Tau=%15.8g -----------' % Tau)
        cdir = handle_run(inputs, 'Tau', float(Tau), MD_rigid) # for json cannot be numpy
        if update_conf:
            # Extract last config of current run
            last_step = np.loadtxt(pjoin(cdir, 'out.dat'))[-1]
            print(('%25.15g '*3) % (Tau, Fx, Fy), ''.join(['%25.15g ' % f for f in last_step]), file=outlast)
            if update_conf: # Set next run
                inputs['pos_cm'], inputs['theta'] = [float(last_step[[2]]), float(last_step[[3]])], float(last_step[6])
                print('-' * 80, '\n')
        print('-' * 80, '\n')
    outlast.close()

    t1=time()
    print('Done in %is (%.2fmin)' % (t1-t0, (t1-t0)/60))

if __name__ == "__main__":
        #-------------------------------------------------------------------------------
    # Argument parser
    #-------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="""Snippet to run a loop over torques for a given cluster""")

    # Optional args
    parser.add_argument('--input', '-i',
                        dest='inputf', type=str, default=False,
                        help='filename of JSON with system parameters')
    parser.add_argument('--ranges', '-r',
                        dest='rangesf', type=str, required=True,
                        help='filename of JSON with range of torques')
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

    Tau0, Tau1, dTau = 0, 0, 1
    try:
        Tau0, Tau1, dTau = ranges['Tau0'], ranges['Tau1'], ranges['dTau']
    except KeyError:
        print('Could not set torque range')
        pass

    Tau_loop(Tau0, Tau1, dTau, inputs)
