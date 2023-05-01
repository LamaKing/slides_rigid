#!/bin/bash
../MD_rigid_rototrasl.py params.json > out.dat
../../snips/rigid2xyz.py -i params.json > out.xyz
