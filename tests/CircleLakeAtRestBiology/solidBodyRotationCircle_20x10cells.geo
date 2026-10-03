//---------------------------------------------------------------------------
// Points defining the circular domain

// Center of the circle
Point(1) = {2., 2., 0., 0.1};

// Points on the boundary
Point(2) = {4., 2., 0., 0.1};   // right
Point(3) = {2., 4., 0., 0.1};   // top
Point(4) = {0., 2., 0., 0.1};   // left
Point(5) = {2., 0., 0., 0.1};   // bottom


//---------------------------------------------------------------------------
// Circular arcs forming the boundary

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};


//---------------------------------------------------------------------------
// 2D surface

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};


//---------------------------------------------------------------------------
// Physical IDs

Physical Surface(3) = {1};

// The entire circular boundary has the same boundary ID.
// We will use this as a wall boundary.
Physical Curve(0) = {1, 2, 3, 4};


//---------------------------------------------------------------------------
// Meshing parameters

Mesh.RecombineAll = 1;

Show "*";
