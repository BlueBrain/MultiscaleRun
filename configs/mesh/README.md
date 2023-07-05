# Generate STEPS-compatible meshes using gmsh

1. Use the gmsh GUI
2. In the Geometry > Elementary entities > Add > Box, set the box coordinates. The simplest approach is to create a mesh that coincides with the bounding box (BB) of the neurons that you are dealing with (acttually extend it a bit over the BB to avoid corner cases).
3. Create the 3D mesh, Mesh > 3D, and refine it as many times deemed necessary (Refine by splitting option in the Mesh tab).
4. Partition the mesh : https://github.com/CNS-OIST/STEPS4ModelRelease/blob/main/caBurst/mesh/README.md#generate-partitioned-mesh-for-steps34

Useful Gmsh commands can be found [here](https://bbpteam.epfl.ch/project/spaces/pages/viewpage.action?spaceKey=BBPHPC&title=Useful+Gmsh+commands).
