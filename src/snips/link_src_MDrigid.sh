#!/bin/bash

# Shortcut to link the source of this package to the current director
# This should not be necessary if instelled via pip

src_fld=/PATH/TO/SOURCE/OF/rigid_cluster
src_fname=( tool_create_cluster.py tool_create_substrate.py misc.py ) # minimal
if [[ $1 == "all" ]]
then
        src_fname=( ${src_fname[@]} MD_rigid_rototrasl.py static_roto_map.py static_trasl_map.py static_rototrasl_map.py static_barrier_string.py string_method.py )
fi

force_opt=""
if [[ $2 == "f" ]]
then
    echo "Force option enabled"
    force_opt="-f"
fi


echo "link from $src_fld to $PWD ";
for i in ${src_fname[@]}
do
        echo "linking $i"
        ln $force_opt -s $src_fld/$i ;
done
