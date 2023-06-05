#!/usr/bin/env python3

import sys, os, json, logging, argparse
import numpy as np
from numpy import pi
from tool_create_substrate import get_ks

if __name__ == "__main__":

    #-------------------------------------------------------------------------------
    # Argument parser
    #-------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="""Snippet to create elementary sinusoidal potential

    Only supports basic, single wavevector symmetries defined in Vanossi, Manini, Tosatti PNAS (2011).
    To change the phase alpha_n (e.g. saddle point in triangular lattixe along x or along y), please use directly the python function get_ks.

    Read input json from file (and update in place) or from stdin and print on stdou.""")
    # Optional args
    parser.add_argument('--fname', '-f',
                        dest='fname', type=str, default=False,
                        help='filename of JSON input to be changed in place.')
    parser.add_argument('--n_symm', '-n',
                        dest='n', type=int, required=True,
                        help='symmetry order, between 2 and 5.')
    parser.add_argument('--spacing', '-r',
                        dest='R', type=float, required=True,
                        help='lattice spacing')
    parser.add_argument('--debug',
                        action='store_true', dest='debug',
                        help='show debug informations.')

    #-------------------------------------------------------------------------------
    # Initialize and check variables
    #-------------------------------------------------------------------------------
    args = parser.parse_args(sys.argv[1:])

    debug = args.debug
    #-------- SET UP LOGGER -------------
    c_log = logging.getLogger('driver') # Set name identifying the logger.
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    logging.basicConfig(format='[%(levelname)5s - %(name)15s] %(message)s')
    c_log.setLevel(logging.INFO)
    if args.debug: c_log.setLevel(logging.DEBUG)

    #-------- READ INPUTS -------------
    fname = args.fname
    if fname:
        c_log.info('Change file %s in place' % fname)
    else:
        c_log.info('Read from stdin print on stdout')

    R, n  = args.R, args.n

    #--------  READ IN
    if fname: inj = open(fname)
    else: inj = sys.stdin
    inputs = json.load(inj)
    if fname: inj.close()

    #-------- SCALE SUBSTRATE VECTOR

    # Define symmetry of substrate
    if n == 2:
        n, c_n, alpha_n = 2, 1, 0 # Lines
    elif n == 3:
        n, c_n, alpha_n = 3, 4/3, 0 # Tri
    elif n == 4:
        n, c_n, alpha_n = 4, np.sqrt(2), pi/4 # Square
    elif n == 5:
        n, c_n, alpha_n = 5, 2, 0 # Qausi-cristal
    elif n == 5:
        n, c_n, alpha_n = 6, 4/np.sqrt(3), -pi/6

    # Get the reciprocal vectors defined by the coefficients above
    ks = get_ks(R, n, c_n, alpha_n)
    # Change inputs dicitonary
    inputs['well_shape'] = 'sin'
    inputs['ks'] = [list(k) for k in ks]

    #-------- DUMP OUT
    outj = sys.stdout
    if fname:
        with open(fname, 'w') as outj:
            json.dump(inputs, outj, indent=4)
