#!/usr/bin/env python

import sys, os, json, logging, multiprocessing
import numpy as np
from numpy import pi
from tool_create_substrate import get_ks

debug = False
#-------- SET UP LOGGER -------------
c_log = logging.getLogger('driver') # Set name identifying the logger.
# Adopted format: level - current function name - message. Width is fixed as visual aid.
logging.basicConfig(format='[%(levelname)5s - %(name)15s] %(message)s')
c_log.setLevel(logging.INFO)
if debug: c_log.setLevel(logging.DEBUG)

#-------- READ INPUTS -------------
fname = False
if len(sys.argv) > 3:
    fname = sys.argv[3]
    c_log.info('Change file %s in place' % fname)
else:
    c_log.info('Read from stdin print on stdout')

R, n  = float(sys.argv[1]), int(sys.argv[2])

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
if fname: outj = open(fname, 'w')
json.dump(inputs, outj, indent=4)
if fname: outj.close()
