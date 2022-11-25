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

import config

import dualrun.sec_mapping.sec_mapping as sec_mapping

from . import utils

import os


def gen_model():
    mdl = Model()
    with mdl:
        extraNa = VolumeSystem(name=config.Volsys.name)
        Na = Species(name=config.Na.name)
        with extraNa:
            diff = Diffusion.Create(Na, config.Na.diffcst)
    return mdl


def gen_mesh():
    # STEPS default length scale is m
    # NEURON default length scale is um

    mesh = (
        DistMesh(config.steps_mesh_path, scale=1e-6)
        if config.steps_version == 4
        else TetMesh.LoadGmsh(config.steps_mesh_path, scale=1e-6)
    )

    ntets = len(mesh.tets)
    with mesh:
        if config.steps_version == 4:
            extra = Compartment(name=config.Mesh.compname, vsys=config.Volsys.name)
        else:
            extra = Compartment(
                mesh.tets, name=config.Mesh.compname, vsys=config.Volsys.name
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


def init_steps(ndamus):

    model = gen_model()
    msh, ntets = gen_mesh()

    steps_sim = init_solver(model, msh)
    steps_sim.newRun()

    # In STEPS4 the global indices are not sequential and they can be over ntets.
    # This dictionary does a mapping from 0 to ntets. In STEPS3, it is trivial, i: i.
    global_inds = {tet.idx: i for i, tet in enumerate(msh.tets)}

    logging.info("Computing segments per tet...")
    neurSecmap = sec_mapping.get_sec_mapping_collective(
        ndamus, msh, config.Na.current_var
    )

    # there are 0.001 M/mM
    steps_sim.extra.Na.Conc = 1e-3 * config.Na.conc_0 * config.CONC_FACTOR
    tetVol = np.array([i.Vol for i in msh.tets], dtype=float)

    utils.print_once(f"The total tet mesh volume is : {np.sum(tetVol)}")

    index = np.array(
        range(ntets), dtype=np.int64 if config.steps_version == 4 else np.uint32
    )

    return steps_sim, neurSecmap, ntets, global_inds, index, tetVol
