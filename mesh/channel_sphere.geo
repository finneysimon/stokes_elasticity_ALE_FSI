///////////////////////////////////////////////////////////////////
// Gmsh file for creating a finite element mesh
// In this case, we consider a sphere of radius R in a channel
// For a great tutorial on using Gmsh, see
// https://www.youtube.com/watch?v=aFc6Wpm69xo
///////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

// default element size for the fluid
es = 4e-2;

// default element size at the fluid-solid interface
esc = 5e-3;

// default element size for the solid
esa = 2e-2;

// length and half-width of the channel
L = 10;
H = 0.5;

// radius of circle - leave as 0.45 if running generatemeshes.sh
R = 0.45;

////////////////////////////////////////////////////////////

// Create all of the points

// Points for the circle
Point(1) = {-R, 0, 0, esc};
Point(2) = {0, 0, 0, esa};
Point(3) = {R, 0, 0, esc};

// Points for the domain corners
Point(4) = {L/2, 0, 0, es};
Point(5) = {L/2, H, 0, es};
Point(6) = {-L/2, H, 0, es};
Point(7) = {-L/2, 0, 0, es};

// Create circle and lines
Circle(1) = {3, 2, 1};

Line(2) = {1, 7};
Line(3) = {7, 6};
Line(4) = {6, 5};
Line(5) = {5, 4};
Line(6) = {4, 3};

Curve Loop(1) = {1:6};
Plane Surface(1) = {1};

// now let's add the circle to the domain and mesh it

// Point(8) = {1e-2, 0, 0, esc};

Line(7) = {1, 2};
// Line(8) = {2, 8};
Line(8) = {2, 3};

Curve Loop(2) = {7, 8, 1};
Plane Surface(2) = {2};

// create physical lines (for Fenics)

// circle
Physical Curve(1) = {1};

// axis for fluid
Physical Curve(2) = {2, 6};

// inlet
Physical Curve(3) = {3};

// output
Physical Curve(4) = {5};

// channel wall
Physical Curve(5) = {4};

// axis for solid
Physical Curve(6) = {7, 8, 9};


// bulk (fluid)
Physical Surface(10) = {1};

// bulk (solid)
Physical Surface(11) = {2};
