from vasculatureapi import PointVasculature

import pickle
import time
import logging
import pandas as pd
from pathlib import Path
from pathlib import PurePath
import yaml
import psutil

import neurodamus
import libsonata

from bloodflow import bloodflow
from bloodflow import utils
from bloodflow.entry_nodes import METHODS
from bloodflow.entry_nodes import compute_impedance_matrix
from bloodflow.entry_nodes import create_entry_nodes
from bloodflow.report_writer import write_simulation_report
from bloodflow.utils import set_edge_data
from bloodflow.vtk_io import vtk_writer

import config


class MsrBloodflowManager:
    def __init__(self, ndamus):
        self.ndamus = ndamus
        self.min_subgraph_size = 100

        with open(config.bloodflow_params_path) as f:
            self.params = yaml.full_load(f)

        self.load_circuit(ndamus)

        self.graph.edge_properties.index = pd.MultiIndex.from_frame(
            self.graph.edge_properties.loc[:, ["section_id", "segment_id"]]
        )

        self.get_entry_nodes()
        self.get_input_flow()

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
        self.entry_nodes = create_entry_nodes(
            self.graph, self.params, method=METHODS.ERB_pairs
        )

        logging.info("end compute entry nodes")

    @staticmethod
    def get_manager(ndamus):
        return ndamus.circuits.get_edge_manager(
            "vasculature", "astrocytes", neurodamus.ngv.GlioVascularManager
        )

    @_logs
    def load_circuit(self, ndamus):
        vasculature_path = self.get_manager(ndamus).circuit_conf["VasculaturePath"]

        logging.info("loading circuit")
        self.graph = PointVasculature.load_sonata(vasculature_path)
        set_edge_data(self.graph)
        logging.info(f"end loading circuit")

    @_logs
    def get_input_flow(self):
        logging.info("input flow")

        self.input_flow = bloodflow.get_input_flow(
            self.graph,
            input_nodes=self.entry_nodes,
            input_flows=len(self.entry_nodes) * [1.0],
        )
        print("end input flow")

    @_logs
    def get_static_flow(self):
        logging.info("static flow")
        bloodflow.update_static_flow_pressure(self.graph, self.input_flow)
        logging.info("end static flow")

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
            return self.graph.edge_properties.index[
                manager._gliovascular.source_nodes(endfeet)
            ]

        def get_radii(astro_id):
            astrocyte = manager._cell_manager.gid2cell[astro_id + manager._gid_offset]
            if astrocyte.endfeet is None:
                return []
            return [sec(0.5).vascouplingB.Rad for sec in astrocyte.endfeet]

        vasc_ids = [id for astro_id in astro_ids for id in get_vasc_ids(astro_id)]
        radii = [r for astro_id in astro_ids for r in get_radii(astro_id)]

        self.graph.edge_properties.loc[vasc_ids, "radius"] = radii
        logging.info("end syncing")
