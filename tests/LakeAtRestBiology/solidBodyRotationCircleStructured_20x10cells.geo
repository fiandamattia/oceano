//---------------------------------------------------------------------------
// STRUCTURED QUADRILATERAL MESH OF A CIRCULAR DOMAIN
//
// Circle:
//   center = (2,2)
//   radius = 2
//
// Mesh structure:
//   1 central diamond
//   4 external curved quadrilateral blocks
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
// PARAMETERS

// Number of POINTS, not cells:
// 21 points -> 20 cells
// 11 points -> 10 cells

Ntheta  = 21;
Nradial = 11;


//---------------------------------------------------------------------------
// POINTS

// Center of the circle
// Used only to define the circular arcs
Point(1) = {2., 2., 0., 0.1};


// Inner diamond
Point(2) = {2., 1., 0., 0.1};   // bottom
Point(3) = {3., 2., 0., 0.1};   // right
Point(4) = {2., 3., 0., 0.1};   // top
Point(5) = {1., 2., 0., 0.1};   // left


// Outer circle
Point(6) = {2., 0., 0., 0.1};   // bottom
Point(7) = {4., 2., 0., 0.1};   // right
Point(8) = {2., 4., 0., 0.1};   // top
Point(9) = {0., 2., 0., 0.1};   // left


//---------------------------------------------------------------------------
// INNER DIAMOND

Line(1) = {2, 3};   // bottom -> right
Line(2) = {3, 4};   // right  -> top
Line(3) = {4, 5};   // top    -> left
Line(4) = {5, 2};   // left   -> bottom


//---------------------------------------------------------------------------
// RADIAL CONNECTIONS

Line(5) = {2, 6};   // inner bottom -> outer bottom
Line(6) = {3, 7};   // inner right  -> outer right
Line(7) = {4, 8};   // inner top    -> outer top
Line(8) = {5, 9};   // inner left   -> outer left


//---------------------------------------------------------------------------
// CIRCULAR BOUNDARY

Circle(9)  = {6, 1, 7};   // bottom -> right
Circle(10) = {7, 1, 8};   // right  -> top
Circle(11) = {8, 1, 9};   // top    -> left
Circle(12) = {9, 1, 6};   // left   -> bottom


//---------------------------------------------------------------------------
// CENTRAL BLOCK

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};


//---------------------------------------------------------------------------
// BOTTOM-RIGHT BLOCK

Curve Loop(2) = {1, 6, -9, -5};
Plane Surface(2) = {2};


//---------------------------------------------------------------------------
// TOP-RIGHT BLOCK

Curve Loop(3) = {2, 7, -10, -6};
Plane Surface(3) = {3};


//---------------------------------------------------------------------------
// TOP-LEFT BLOCK

Curve Loop(4) = {3, 8, -11, -7};
Plane Surface(4) = {4};


//---------------------------------------------------------------------------
// BOTTOM-LEFT BLOCK

Curve Loop(5) = {4, 5, -12, -8};
Plane Surface(5) = {5};


//---------------------------------------------------------------------------
// TRANSFINITE DISCRETIZATION

// Tangential direction:
// inner diamond edges and external circular arcs
Transfinite Curve {1, 2, 3, 4, 9, 10, 11, 12} = Ntheta;

// Radial direction
Transfinite Curve {5, 6, 7, 8} = Nradial;


// Define the structured topology of each surface

Transfinite Surface {1} = {2, 3, 4, 5};

Transfinite Surface {2} = {2, 3, 7, 6};

Transfinite Surface {3} = {3, 4, 8, 7};

Transfinite Surface {4} = {4, 5, 9, 8};

Transfinite Surface {5} = {5, 2, 6, 9};


//---------------------------------------------------------------------------
// QUADRILATERAL ELEMENTS

Recombine Surface {1, 2, 3, 4, 5};


//---------------------------------------------------------------------------
// PHYSICAL IDs

// Entire 2D domain
Physical Surface(3) = {1, 2, 3, 4, 5};

// Entire external circular boundary = wall boundary ID 0
Physical Curve(0) = {9, 10, 11, 12};


//---------------------------------------------------------------------------

Show "*";
