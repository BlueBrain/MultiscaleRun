# Metabolism Models

This directory contains files related to the metabolism simulator integrated into the `multiscale_run` Python package. The files here define the system of ordinary differential equations (ODE) that simulate metabolism. Below are the key components:

## Key Files

### `data/u0.csv`
- **Description**: Contains the initial values for the ODE system, which define the starting conditions for the metabolism simulation.
- **Original Name**: `u0steady_22nov22.csv`
- **Location**: This file is essential for initializing the simulation and setting up the model.

### `ODE_system.jl`
- **Description**: Contains the system of equations that describe the metabolism model, along with the matrices and references necessary for running the metabolism simulator.
- **Original Name**: `metabolismWithSBBFinput_ndamAdapted_opt_sys_young_202302210826_2stim.js`
- **Location**: This file defines the dynamics of the model and can be modified for different simulation scenarios.

### data/*.jl
- **Description**: Contains all the additional constants required for the model.
- **Original Name**: they all maintaned their original names.

## Historical Models

If you're interested in the earlier versions of ODE system, you can find them in the following location:

- **Location**: `/multiscale_run/multiscale_run/data/metabolismndam_reduced`
- **Commit Reference**: For the previous models, refer to commit `b9f5508` or earlier. This commit includes the old versions of the metabolism model and associated files.

