#!/bin/bash

base_name="channel_sphere"
fname="channel_sphere_testing"

gmsh $base_name.geo -format msh2 -2 -o $fname.msh
dolfin-convert $fname.msh $fname.xml
