#!/bin/bash
# loop over radii and generate meshes accordingly

echo $PATH
PATH=$PATH:/bin:/usr/bin:/usr/local/bin:/home/simon/anaconda3/envs/fenics_old/bin:/home/simon/anaconda3/envs/fenics_old/lib/python3.8/*
export PATH

filename="channel_sphere.geo"
init_rad=".45"

# loop over radii
for (( radii=1; radii<10; radii++ ))
do

new_rad=$(bc -l <<< "scale=2; $((radii))/20")
echo $new_rad

# Take the search string
search="R = 0$init_rad"

# Take the replace string
replace="R = 0$new_rad"

# replace with new radius
if [[ $search != "" && $replace != "" ]]; then
sed -i "s/$search/$replace/" $filename
fi

# generate mesh file
gmsh channel_sphere.geo -format msh2 -2 -o channel_sphere_$((radii))_fine.msh

# generate xml files
dolfin-convert channel_sphere_$((radii))_fine.msh channel_sphere_$((radii))_fine.xml

# update initial radius to replace
init_rad=$new_rad
done