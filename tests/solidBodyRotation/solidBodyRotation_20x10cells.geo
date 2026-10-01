//---------------------------------------------------------------------------
// In this section we create the points that make up the domain boundary

// Points on the boundary of the domain
/*
Point(1) = {0., 0., 0, 0.1}; // (x,y,z) e 4 punto è mesh size
Point(2) = {0.5, 0., 0, 0.1};
Point(3) = {1., 0., 0, 0.1};
Point(4) = {1., 0.5, 0, 0.1};
Point(5) = {1., 1., 0, 0.1}; // (x,y,z) e 4 punto è mesh size
Point(6) = {0.5, 1., 0, 0.1};
Point(7) = {0., 1., 0, 0.1};
Point(8) = {0., 0.5, 0, 0.1};
*/
// Points on the boundary of the domain
Point(1) = {0., 0., 0, 0.1}; // (x,y,z) e 4 punto è mesh size
Point(2) = {2, 0., 0, 0.1};
Point(3) = {4., 0., 0, 0.1};
Point(4) = {4., 2., 0, 0.1};
Point(5) = {4., 4., 0, 0.1}; // (x,y,z) e 4 punto è mesh size
Point(6) = {2., 4., 0, 0.1};
Point(7) = {0., 4., 0, 0.1};
Point(8) = {0., 2., 0, 0.1};


//---------------------------------------------------------------------------
// This section contains the lines that make up the outer domain

// Lines for the boundary of the domain
Line(1) = {1, 2}; // Linea(1) = collega punto 1 a punto 2
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};

//---------------------------------------------------------------------------
// This section describes the "Plane Surfaces", i.e., the 2D surfaces for meshing

// The surface is given by all the lines composing the domain
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};
// A transfinite surface takes in input the borders POINTS of the domain, just the boundary ones
// that define the limits of the boundaries
Transfinite Surface(1) = {1, 3, 5, 7}; // Transfinite Surface mi da mesh fatta di quadrilateri
// Quadrilateri vantaggiosi per: amplitude, 
// Elementi finiti di tipo Q vs P

// Creates a physical surface.  The expression list on the right hand side is the list
// of elementary surfaces created above.  This is what makes our mesh 2D.
Physical Surface(3) = {1};

//---------------------------------------------------------------------------
// This section describes the physical IDs of certain objects. Cioè i colori dei bordi

// Assign boundary ID of 0 to the domain boundary

// an ID of 1 to the inflow boundary
Physical Curve(1) = {2,4,6,8};
// and an ID of 2 for the outflow
Physical Curve(2) = {1,3,5,7};

//---------------------------------------------------------------------------
// Parameters for the meshing

// We want a structured mesh
//Mesh.Algorithm = 0;
Mesh.RecombineAll = 1; // Attiva la ricombinazione verso i quadrilateri
//Mesh.CharacteristicLengthFactor = .6;
//Mesh.SubdivisionAlgorithm = 1;
//Mesh.Smoothing = 20;
Show "*";
