#!/usr/bin/env python3

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tool_create_cluster import rotate
from tool_create_substrate import substrate_from_params

if __name__ == "__main__":
    
    #-------------------------------------------------------------------------------
    # Argument parser
    #-------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="""Convert the output of MD_rigid in xyz trajectory

    Output to stdout.
    """
    )
    # Optional args
    parser.add_argument('--pos', '-p',
                        dest='posfname', type=str, default='pos.npy',
                        help='numpy file containing the position of each particle in the rigid cluster')
    parser.add_argument('--out', '-o',
                        dest='outfname', type=str, default='out.dat',
                        help='output file of MD_rigid.')
    parser.add_argument('--input', '-i',
                        dest='inputfname', type=str, default='input.json',
                        help='JSON input file, with parameters for MD.')
    parser.add_argument('--Nskip', '-n',
                        dest='nskip', type=int, default='1',
                        help='number of frames to skip in rendering the trajectory')
    parser.add_argument('--debug',
                        action='store_true', dest='debug',
                        help='show debug informations.')

    #-------------------------------------------------------------------------------
    # Initialize and check variables
    #-------------------------------------------------------------------------------
    args = parser.parse_args(sys.argv[1:])

    posfname = args.posfname
    outfname = args.outfname
    inputfname = args.inputfname
    nskip = args.nskip 


    print('Load positions %s, trajectory %s and sys info %s' % (posfname, outfname, inputfname), file=sys.stderr)
    #pos0 = np.loadtxt(posfname) # position saved in plain text format
    pos0 = np.load(posfname)     # position saves in numpy machine format
    data = np.loadtxt(outfname)
    with open(inputfname) as inj:
        inputs = json.load(inj)


    pen_func, en_func, en_input = substrate_from_params(inputs)

    print("Sys info", file=sys.stderr)
    for k, v in inputs.items():
        print('%10s : %30s' %  (str(k), str(v)), file=sys.stderr)

    xv, yv, thv = data[:,2], data[:,3], data[:,6] # trajectory
    #x0, y0, th0 = data[0,1], data[0,2], data[0,6] # prev

    dummy_atom = 'X'
    xyzstring = "%4s %25.20g %25.20g %25.20g"
    #add_str = "" # no extra fields
    add_str = "%25.20g %25.20g %25.20g %25.20g" # extra fileds, e.g. force, substrate energy of each particle
    linestr = xyzstring + " " + add_str
    outstream = sys.stdout

    Np = pos0.shape[0]
    i = 0
    pos_cm = np.array([xv[0],yv[0]])
    d0 = pos_cm.copy()
    for x, y, th in zip(xv, yv, thv):

        pos_cm = np.array([x,y])

        if i%nskip != 0:
            i += 1
            continue
        # Update
        pos = rotate(pos0+d0, th) - d0

        tt = pos_cm - rotate(d0, th) + d0
        en, F, tau = pen_func(pos + pos_cm, pos_cm, *en_input)

        # Print
        print(Np, file=outstream)
        print("#time %20.15g" % data[i,0], file=outstream)
        for ir, r in enumerate(pos):
            r += tt
            #r += [0,y] # Do not add the direction of driving force, becomes difficult to follow
            print(linestr % (dummy_atom, r[0], r[1], 0, en[ir], F[ir, 0], F[ir, 1], tau[ir]), file=outstream)

        i += 1
