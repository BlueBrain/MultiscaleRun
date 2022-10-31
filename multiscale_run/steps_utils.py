import time
import logging
import numpy as np

import steps.interface

from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *
from steps.utils import *

import params
import dualrun.sec_mapping.sec_mapping as sec_mapping

from . import utils

import os


def gen_model():
    mdl = Model()
    with mdl:
        extraNa = VolumeSystem(name=params.Volsys0.name)
        Na = Species(name=params.Na.name)
        with extraNa:
            diff = Diffusion.Create(Na, params.Na.diffcst)
    return mdl


def gen_mesh(steps_version, mesh_path):
    if type(steps_version) is str:
        steps_version = int(steps_version)

    # STEPS default length scale is m
    # NEURON default length scale is um

    mesh = (
        DistMesh(mesh_path, scale=1e-6)
        if steps_version == 4
        else TetMesh.LoadGmsh(mesh_path, scale=1e-6)
    )

    ntets = len(mesh.tets)
    with mesh:
        if steps_version == 4:
            extra = Compartment(name=params.Mesh.compname, vsys=params.Volsys0.name)
        else:
            extra = Compartment(
                mesh.tets, name=params.Mesh.compname, vsys=params.Volsys0.name
            )

    return mesh, ntets


def init_solver(model, mesh):
    rng = RNG("mt19937", 512, int(time.time() % 4294967295))
    if isinstance(mesh, steps.API_2.geom.DistMesh):
        return Simulation(
            "DistTetOpSplit",
            model,
            mesh,
            rng,
            searchMethod=NextEventSearchMethod.GIBSON_BRUCK,
        )
    else:
        part = LinearMeshPartition(mesh, 1, 1, MPI.nhosts)
        # change MPI.EF_NONE with MPI.EF_DV_PETSC if you add membranes
        return Simulation("TetOpSplit", model, mesh, rng, MPI.EF_NONE, part)


def init_steps(steps_version, ndamus, mesh_path):


    model = gen_model()
    msh, ntets = gen_mesh(steps_version, mesh_path)

    steps_sim = init_solver(model, msh)
    steps_sim.newRun()

    # In STEPS4 the global indices are not sequential and they can be over ntets.
    # This dictionary does a mapping from 0 to ntets. In STEPS3, it is trivial, i: i.
    global_inds = {tet.idx: i for i, tet in enumerate(msh.tets)}

    logging.info("Computing segments per tet...")
    neurSecmap = sec_mapping.get_sec_mapping_collective(
        ndamus, msh, params.Na.current_var
    )

    # there are 0.001 M/mM
    steps_sim.extra.Na.Conc = 1e-3 * params.Na.conc_0 * params.CONC_FACTOR
    tetVol = np.array([i.Vol for i in msh.tets], dtype=float)

    utils.print_once(f"The total tet mesh volume is : {np.sum(tetVol)}")


    index = np.array(range(ntets), dtype=np.int64 if steps_version == 4 else np.uint32)

    return steps_sim, neurSecmap, ntets, global_inds, index, tetVol
