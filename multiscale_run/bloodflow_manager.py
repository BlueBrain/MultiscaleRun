from vasculatureapi import PointVasculature

import pickle
import logging
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import sparse
import yaml
from pathlib import Path
from pathlib import PurePath

from . import utils

import neurodamus
import libsonata

import steps.interface
from steps.geom import *
from steps.model import *
from steps.rng import *
from steps.saving import *
from steps.sim import *
from steps.utils import *

from bloodflow import bloodflow
from bloodflow.entry_nodes import METHODS
from bloodflow.entry_nodes import compute_impedance_matrix
from bloodflow.entry_nodes import create_entry_nodes
from bloodflow.report_writer import write_simulation_report
from bloodflow.utils import set_edge_data
from bloodflow.vtk_io import vtk_writer
from bloodflow.PetscBinaryIO import get_conf

import config

from mpi4py import MPI as MPI4PY

comm = MPI4PY.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


class MsrBloodflowManager:
    def __init__(self, vasculature_path, params):
        logging.info("init MsrBloodflowManager")
        # Sanity check!
        import petsc4py
        from petsc4py import PETSc

        _, _, complexscalars = get_conf()
        assert complexscalars, "* Use PETSc with complex number support *"

        self.params = params

        self.graph = None
        if rank == 0:
            self.load_circuit(vasculature_path)

        self.get_entry_nodes()
        self.get_input_flow()

    @utils.logs_decorator
    def get_seg_points(self, scale):
        """Get a series disjoint segments described by the extreme points

        return: np.array(2*n_segments, 3)
        """
        if self.graph is None:
            return np.array([])

        start_node_coords = self.graph.node_properties.iloc[
            self.graph.edge_properties.loc[:, "start_node"]
        ].to_numpy()[:, :3]
        end_node_coords = self.graph.node_properties.iloc[
            self.graph.edge_properties.loc[:, "end_node"]
        ].to_numpy()[:, :3]
        pts = np.empty(
            (start_node_coords.shape[0] + end_node_coords.shape[0], 3), dtype=float
        )
        pts[0::2] = start_node_coords
        pts[1::2] = end_node_coords
        pts *= scale  # we assume same scaling factor as the mesh

        return pts

    def get_flows(self):
        """Flow in each segment"""
        return self.graph.edge_properties["flow"].to_numpy()

    def get_vols(self):
        """Volume in each segment"""
        return self.graph.edge_properties.volume.to_numpy()

    @utils.cache_decorator(
        path=config.cache_path,
        is_save=config.cache_save,
        is_load=config.cache_load,
        only_rank0=False,
        field_names="entry_nodes",
    )
    @utils.logs_decorator
    def get_entry_nodes(self):
        """Bloodflow input nodes"""
        # old_entry_nodes: [1039324, 919494, 589795]
        self.entry_nodes = np.array(
            create_entry_nodes(self.graph, self.params, method=METHODS.ERB_pairs)
        )
        logging.info(f"entry nodes: {self.entry_nodes}")

    @utils.logs_decorator
    def load_circuit(self, vasculature_path):
        # According to the PETSc approach

        self.graph = PointVasculature.load_sonata(vasculature_path)

        set_edge_data(self.graph)
        self.graph.edge_properties.index = pd.MultiIndex.from_frame(
            self.graph.edge_properties.loc[:, ["section_id", "segment_id"]]
        )

    @utils.logs_decorator
    def get_input_flow(self):
        """ apply the input_v to entry nodes """

        self.input_flow = bloodflow.get_input_flow(
            self.graph,
            input_nodes=self.entry_nodes,
            input_flows=[self.params["input_v"]]*len(self.entry_nodes),
        )

    @utils.logs_decorator
    def update_static_flow(self):
        """Given graph and input update flows and volumes for the quasi-static computation"""
        bloodflow.update_static_flow_pressure(self.graph, self.input_flow)

    @utils.logs_decorator
    def set_radii(self, vasc_ids, radii):
        """
        Here we want to set the radii of the vasculature sections identified by
        vasc_ids. The problem stems from the fact that in theory we could have repeating
        indexes in vasc_ids. What radii do we set in that case?
        
        We assume that the astrocytes operate in series. Every astrocyte gets to control
        a subpart of the segment of length l/n where n is the number of astrocytes
        that act on the this particular segment. 

        We want to maintain the same volume so:

        V_r_eq = V_r0 + V_r1 + ...

            r_eq = sqrt(\sum_n r_i^2/n)
        """

        def eq_radii(v):
            v = np.array(v)
            """compute x from: 1/x^2 = \sum_n 1/(n*r_i^2)"""
            return np.sqrt(v.dot(v)/len(v))

        vasc_ids = [self.graph.edge_properties.index[i] for i in vasc_ids]

        # I did not find a more pythonic way of doing this. I am open to suggestions
        d = defaultdict(list)
        for k, v in zip(vasc_ids, radii):
            d[k].append(v)

        d = {k: eq_radii(v) for k, v in d.items()}
        # without interventions, d.keys() and d.values() are ordered in the samw way
        self.graph.edge_properties.loc[list(d.keys()), "radius"] = list(d.values())
