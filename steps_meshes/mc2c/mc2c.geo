SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthFactor = 10;

Mesh.MshFileVersion = 4.1;
Mesh.PartitionOldStyleMsh2 = 1;
Mesh.PartitionCreateGhostCells = 1;

Point(1) = {1869.9502, 30.15647119, -386.86829597, 30.0};
Point(2) = {1869.9502, 30.15647119, -2077.0386, 30.0};
Point(3) = {1869.9502, 2076.31733326, -386.86829597, 30.0};
Point(4) = {1869.9502, 2076.31733326, -2077.0386, 30.0};
Point(5) = {3513.61609042, 30.15647119, -386.86829597, 30.0};
Point(6) = {3513.61609042, 30.15647119, -2077.0386, 30.0};
Point(7) = {3513.61609042, 2076.31733326, -386.86829597, 30.0};
Point(8) = {3513.61609042, 2076.31733326, -2077.0386, 30.0};

Line(1) = {1, 2};
Line(2) = {1, 3};
Line(3) = {1, 5};
Line(4) = {3, 4};
Line(5) = {4, 2};
Line(6) = {2, 6};
Line(7) = {6, 5};
Line(8) = {5, 7};
Line(9) = {7, 8};
Line(10) = {8, 6};
Line(11) = {8, 4};
Line(12) = {7, 3};


Curve Loop(1) = {2, -12, -8, -3};
Plane Surface(1) = {1};
Curve Loop(2) = {8, 9, 10, 7};
Plane Surface(2) = {2};
Curve Loop(3) = {10, -6, -5, -11};
Plane Surface(3) = {3};
Curve Loop(4) = {5, -1, 2, 4};
Plane Surface(4) = {4};
Curve Loop(5) = {12, 4, -11, -9};
Plane Surface(5) = {5};
Curve Loop(6) = {3, -7, -6, -1};
Plane Surface(6) = {6};


Surface Loop(1) = {1, 4, 3, 2, 5, 6};
Volume(1) = {1};

Physical Volume("extra", 1) = {1};



