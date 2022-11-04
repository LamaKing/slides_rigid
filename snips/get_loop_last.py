#!/usr/bin/env python3

import os, shutil, json, sys
from os.path import join as pjoin
from time import time
import numpy as np

def get_loop_lasts(F0, F1, dF,
                   Tau0, Tau1, dTau,
                   thF=0, avg_len=100, method='average'):
    """Loop over values of Tau form Tau0 to Tau1 in steps dTau

    Update config concatenates pos cm and theta between runs"""
    # TODO: implement methods: average or max over last N frames (last frame should be obtained by len avg 1
    t0 = time() # Start the clock

    pwd =  os.environ['PWD']
    print('Working in ', pwd)

    if method=='average': c_method = np.average
    elif method=='max': c_method = np.max
    else: raise ValueError('Process method %s not implemented' % method)

    outlast = open('last.dat', 'w')

    Tau, F = None, None
    if Tau0 == Tau1: Taurange = [None]
    else: Taurange = np.arange(Tau0, Tau1, dTau)
    if F0 == F1: Frange = [None]
    else: Frange = np.arange(F0, F1, dF)

    for Tau in Taurange:
        for F in Frange:
            print('--------- ON Tau,F=-----------', (Tau,F), file=sys.stderr)
            if F!=None: Fx, Fy = np.cos(thF)*F, np.sin(thF)*F
            else: Fx, Fy = None, None
            c_key, c_val = ['Tau', 'Fx', 'Fy'], [Tau, Fx, Fy]
            cdir = '-'.join(['%s_%.4g' % (cc_key, cc_val)
                             for cc_key, cc_val in zip(c_key, c_val) if cc_val != None])
            # Extract last N output of current run: average or maximum
            last_step = np.loadtxt(pjoin(cdir, 'out.dat'))[-avg_len:]
            print(last_step.shape)
            last_step = c_method(last_step, axis=0)
            #print(last_step)
            loop_var = ' '.join(['%25.20g' % (cc_val)
                             for cc_val in c_val if cc_val != None])
            print(loop_var, ''.join(['%25.15g ' % f for f in last_step]), file=outlast)
    outlast.close()

    t1=time()
    print('Done in %is (%.2fmin)' % (t1-t0, (t1-t0)/60))

if __name__ == "__main__":
    # -------- RANGES --------
    with open(sys.argv[1]) as inj:
        ranges = json.load(inj)

    avg_len = 1 # How many frames to consider?
    if len(sys.argv)>2: avg_len = int(sys.argv[2])
    method = 'average' # What to extract, average or max?
    if len(sys.argv)>3: method = sys.argv[3]

    F0, F1, dF = 0, 0, 1
    try:
        F0, F1, dF = ranges['F0'], ranges['F1'], ranges['dF']
        if 'thF' in ranges.keys(): thF = ranges['thF']
    except KeyError: pass
    Tau0, Tau1, dTau = 0, 0, 1
    try:
        Tau0, Tau1, dTau = ranges['Tau0'], ranges['Tau1'], ranges['dTau']
    except KeyError: pass

    get_loop_lasts(F0, F1, dF, Tau0, Tau1, dTau, thF, method=method, avg_len=avg_len)
