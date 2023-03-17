from vasculatureapi import PointVasculature

import pickle
import time
import logging
import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix
import yaml
import psutil
from pathlib import Path
from pathlib import PurePath

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
from bloodflow import utils
from bloodflow.entry_nodes import METHODS
from bloodflow.entry_nodes import compute_impedance_matrix
from bloodflow.entry_nodes import create_entry_nodes
from bloodflow.report_writer import write_simulation_report
from bloodflow.utils import set_edge_data
from bloodflow.vtk_io import vtk_writer

import config

from mpi4py import MPI as MPI4PY
MPI_COMM = MPI4PY.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()

class MsrBloodflowManager:
    def __init__(self, ndamus):
        self.ndamus = ndamus

        with open(config.bloodflow_params_path) as f:
            self.params = yaml.full_load(f)

        self.load_circuit(ndamus)

        # MPI_RANK == 0 has the graph
        if self.graph:
            tmp = [self.graph.node_properties.x.to_numpy(), self.graph.node_properties.y.to_numpy(), self.graph.node_properties.z.to_numpy()]
            tmp = np.array(tmp).transpose()

            self.v_bbox_min = tmp.min(axis=0)
            self.v_bbox_max = tmp.max(axis=0)

            print(
                "bounding box Vasculature [um] : ",
                self.v_bbox_min,
                self.v_bbox_max,
                flush=True,
            )
        else:
            self.v_bbox_min = None
            self.v_bbox_max = None

        self.get_entry_nodes()
        self.get_input_flow()

        # Dictionary with keys the tet ids and entries vectors of triplets of blood vessel section_id, segment_id & intersection ratio
        #self.tet_vasc_map = {}
        # Dictionary with keys tuples of blood vessel section_id, segment_id and entries vectors of tets & intersection ratio
        self.vasc_tet_map = {}
        self.build_tetrahedra_vasculature_mapping()

    def _logs(foo):
        def logs(*args, **kwargs):
            start = time.perf_counter()
            foo(*args, **kwargs)
            mem = psutil.Process().memory_info().rss / 1024**2
            logging.info(f"Memory in use: {mem}")
            stop = time.perf_counter()
            logging.info(f"stop stopwatch. Time passed: {stop - start}")

        return logs

    @_logs
    def get_entry_nodes(self):
        logging.info("compute entry nodes")
        self.entry_nodes = create_entry_nodes(self.graph, self.params, method=METHODS.ERB_pairs)
        logging.info(f"entry nodes: {self.entry_nodes}")

    @staticmethod
    def get_manager(ndamus):
        return ndamus.circuits.get_edge_manager(
            "vasculature", "astrocytes", neurodamus.ngv.GlioVascularManager
        )

    @_logs
    def load_circuit(self, ndamus):
        logging.info("loading circuit : vasculature")
        # According to the PETSc approach
        if MPI_RANK == 0:
            vasculature_path = self.get_manager(ndamus).circuit_conf["VasculaturePath"]
            
            # Check consistency across configuration files
            h5_file = PurePath(self.params["data_folder"], self.params["dataset"] + ".h5")
            assert str(vasculature_path) == str(h5_file)

            self.graph = PointVasculature.load_sonata(vasculature_path)
            set_edge_data(self.graph)
            self.graph.edge_properties.index = pd.MultiIndex.from_frame(
                self.graph.edge_properties.loc[:, ["section_id", "segment_id"]]
            )
        else:
            self.graph = None
        logging.info("end loading circuit : vasculature")

    @_logs
    def get_input_flow(self):
        logging.info("compute input flow")
        self.input_flow = bloodflow.get_input_flow(
            self.graph,
            input_nodes=self.entry_nodes,
            input_flows=len(self.entry_nodes) * [1.0],
        )
        logging.info("end of input flow")

    @_logs
    def get_static_flow(self):
        logging.info("compute static flow")
        bloodflow.update_static_flow_pressure(self.graph, self.input_flow)
        logging.info("end of static flow pressure")

    @_logs
    def sync(self, ndamus):
        logging.info("syncing")
        manager = self.get_manager(ndamus)
        astro_ids = manager._astro_ids

        def get_vasc_ids(astro_id):
            endfeet = manager._gliovascular.afferent_edges(astro_id)
            astrocyte = manager._cell_manager.gid2cell[astro_id + manager._gid_offset]
            if astrocyte.endfeet is None:
                return []
            if MPI_RANK == 0:
                return self.graph.edge_properties.index[
                    manager._gliovascular.source_nodes(endfeet)
                ]
            else:
                return []

        def get_radii(astro_id):
            astrocyte = manager._cell_manager.gid2cell[astro_id + manager._gid_offset]
            if astrocyte.endfeet is None:
                return []
            return [sec(0.5).vascouplingB.Rad for sec in astrocyte.endfeet]

        vasc_ids = [id for astro_id in astro_ids for id in get_vasc_ids(astro_id)]
        radii = [r for astro_id in astro_ids for r in get_radii(astro_id)]

        if MPI_RANK == 0:
            self.graph.edge_properties.loc[vasc_ids, "radius"] = radii
        logging.info("end syncing")

    @_logs
    def build_tetrahedra_vasculature_mapping(self):
        # Only MPI task 0 will build the mapping between neurons and blood vessels/vasculature!
        # Keep in mind that MPI task 0 holds the graph
        # Once it is done, we broadcast the map
        if MPI_RANK == 0:
            # Make sure that we read the whole mesh
            path_to_whole_mesh = config.steps_mesh_path
            if config.steps_version == 4 and 'split' in config.steps_mesh_path:
                path = Path(config.steps_mesh_path)
                # Emulate this kind of dir hierarchy -> https://github.com/CNS-OIST/STEPS4ModelRelease/tree/main/caBurst/mesh
                # twice because we give the mesh and not the folder
                parent = path.parent.parent
                files_list = [str(i) for i in list(parent.iterdir()) if i.is_file() and i.suffix == '.msh']

                while len(files_list) == 0:
                    parent = parent.parent
                    files_list = [str(i) for i in list(parent.iterdir()) if i.is_file() and i.suffix == '.msh']

                path_to_whole_mesh = files_list[0]

            # The whole tet mesh and not the partitioned (use of the intersect method from STEPS3)
            # Since vasculature network/graph is handled by MPI rank 0, the STEPS intersect method should be called
            # only from MPI rank 0. Therefore, we need to load the whole mesh from rank 0 only!
            # The DistMesh (STEPS4) must be called by all ranks and for this reason, the STEPS4 interface cannot
            # be used (otherwise it hangs, i.e. if called by only one rank -MPI deadlock-).
            steps_mesh_whole = (TetMesh.LoadGmsh(path_to_whole_mesh, scale=1e-6))

            start_node_coords = self.graph.node_properties.iloc[self.graph.edge_properties.loc[:, 'start_node']].to_numpy()[:,:3]
            end_node_coords = self.graph.node_properties.iloc[self.graph.edge_properties.loc[:, 'end_node']].to_numpy()[:,:3]
            
            pts = np.empty((start_node_coords.shape[0] + end_node_coords.shape[0], 3), dtype=float)
            pts[0::2] = start_node_coords
            pts[1::2] = end_node_coords
            pts *= config.micrometer2meter # Vasculature in um

            res = steps_mesh_whole.intersectIndependentSegments(pts)


            num_segs = len(start_node_coords)
            for i in range(num_segs):
                section_id = int(self.graph.edge_properties.iloc[i].section_id)
                segment_id = int(self.graph.edge_properties.iloc[i].segment_id)

                #for tet, rat in res[i]:
                #    self.tet_vasc_map.setdefault(tet.idx, []).append((section_id, segment_id, rat))
                
                self.vasc_tet_map[(section_id, segment_id)] = [(tet.idx, rat) for tet, rat in res[i]]

        self.vasc_tet_map = MPI_COMM.bcast(self.vasc_tet_map, root=0)


    def get_Mmat(self, ntets):
        """ Nmat is a n_neuron_segments X n_tets sparse matrix. neuron segment area fraction per tet"""

        nsegs = len(self.vasc_tet_map)
        Mmat = dok_matrix((nsegs, ntets))

        for i, v in enumerate(self.vasc_tet_map.values()):
            l = v if i in self.entry_nodes else v[1:]
            for tet, _ in l:
                Mmat[i, tet] = 1

        return Mmat
