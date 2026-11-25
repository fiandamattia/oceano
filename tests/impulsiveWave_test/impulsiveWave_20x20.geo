//---------------------------------------------------------------------------
// In this section we create the points that make up the outer domain

// Points on the outer domain
Point(1) = {-5., -5., 0, 0.5};
Point(2) = {+5., -5., 0, 0.5};
Point(3) = {+5., +5., 0, 0.5};
Point(4) = {-5., +5., 0, 0.5};

//---------------------------------------------------------------------------
// This section contains the lines that make up the outer domain

// Lines for the left hand side of the domain
Line(4) = {1, 2};
Line(5) = {2, 3};
Line(6) = {3, 4};
Line(7) = {4, 1};

//---------------------------------------------------------------------------
// This section describes the "Plane Surfaces", i.e., the 2D surfaces for meshing

// The surface
Curve Loop(1) = {4, 5, 6, 7};
Plane Surface(1) = {1};
Transfinite Surface(1) = {1, 2, 3, 4};

// Creates a physical surface.  The expression list on the right hand side is the list
// of elementary surfaces created above.  This is what makes our mesh 2D.
Physical Surface(3) = {1, 2, 3, 4};

//---------------------------------------------------------------------------
// This section describes the physical IDs of certain objects.

// Assign boundary ID of 0 to the domain boundary
Physical Curve(0) = {4, 5, 6, 7};

//---------------------------------------------------------------------------
// Parameters for the meshing

// We want a structured mesh
//Mesh.Algorithm = 0;
Mesh.RecombineAll = 1;
//Mesh.CharacteristicLengthFactor = .6;
//Mesh.SubdivisionAlgorithm = 1;
//Mesh.Smoothing = 20;
Show "*";
