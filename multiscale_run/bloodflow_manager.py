import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from astrovascpy import bloodflow
from astrovascpy.utils import Graph, create_entry_largest_nodes
from vascpy import PointVasculature

from . import utils


class MsrBloodflowManager:
    """Manager class for handling bloodflow-related operations.

    This class is responsible for managing bloodflow data and operations within a multiscale simulation.
    It provides methods for retrieving bloodflow segment points, flows, volumes, entry nodes, and more.
    It also handles loading vasculature data and applying boundary flows to entry nodes.

    Attributes
        entry_nodes (list): List of entry nodes for bloodflow simulation.
        boundary_flows (numpy.ndarray): Boundary flows for entry nodes.
        graph (PointVasculature): The vasculature graph.
    """

    def __init__(self, vasculature_path, parameters):
        """Initialize the MsrBloodflowManager.

        This method initializes an instance of the MsrBloodflowManager class by loading vasculature data and
        calculating various bloodflow-related parameters. It also sets the entry nodes and boundary flows
        for subsequent bloodflow simulations.

        Args:
            vasculature_path (str): The path to the vasculature data file.
            parameters (dict): A dictionary containing parameters for bloodflow calculations.
        """
        logging.info("init MsrBloodflowManager")

        self.parameters = parameters

        self.graph = None
        if utils.rank0():
            self.load_circuit(vasculature_path)

        self.get_entry_nodes()
        self.get_boundary_flows()

    @utils.logs_decorator
    def get_seg_points(self, scale):
        """Get a series disjoint segments described by the extreme points

        return: np.array(2*n_segments, 3)
        """
        if self.graph is None:
            return np.array([])

        if hasattr(self, "seg_points"):
            return self.seg_points * scale

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
        self.seg_points = pts

        return self.seg_points * scale

    def get_flows(self):
        """Get the flow in each vasculature segment.

        This method retrieves the blood flow values associated with each vasculature segment in the loaded vasculature graph.

        Returns
            numpy.ndarray: An array containing the flow values for each vasculature segment.
        """
        return self.graph.edge_properties["flow"].to_numpy()

    def get_vols(self):
        """Get the volume in each vasculature segment.

        This method retrieves the volume values associated with each vasculature segment in the loaded vasculature graph.

        Returns
            numpy.ndarray: An array containing the volume values for each vasculature segment.
        """
        return self.graph.edge_properties.volume.to_numpy()

    @utils.logs_decorator
    def get_entry_nodes(self):
        """Get bloodflow input nodes (entry nodes).

        This method determines the bloodflow input nodes in the vasculature graph. It identifies these
        nodes based on criteria defined by the 'parameters' provided when initializing the MsrBloodflowManager.

        Returns
            list: A list of bloodflow input nodes as node IDs.
        """
        self.entry_nodes = create_entry_largest_nodes(
            graph=self.graph, params=self.parameters
        )
        logging.info(f"entry nodes: {self.entry_nodes}")

    @utils.logs_decorator
    def load_circuit(self, vasculature_path):
        """Load the vasculature circuit from a Sonata file.

        This method loads the vasculature circuit from the specified Sonata file at 'vasculature_path' and prepares it for further bloodflow computations.

        Parameters
            vasculature_path (str or Path): The path to the Sonata file containing vasculature data.

        Returns
            None

        Note:
            - The loaded vasculature graph is stored in the 'graph' attribute of the MsrBloodflowManager.
            - This method prepares the vasculature graph by setting edge data and creating a MultiIndex for edge properties based on section and segment IDs.
        """
        pv = PointVasculature.load_sonata(vasculature_path)
        self.graph = Graph.from_point_vasculature(pv)

        self.graph.edge_properties.index = pd.MultiIndex.from_frame(
            self.graph.edge_properties.loc[:, ["section_id", "segment_id"]]
        )

    @utils.logs_decorator
    def get_boundary_flows(self):
        """Calculate boundary flows for entry nodes.

        This method calculates the boundary flows for entry nodes based on the specified input flow
        rate and entry nodes. The calculated boundary flows are stored in the 'boundary_flows' attribute of the MsrBloodflowManager.

        Note:
            - This method requires that the vasculature graph and entry nodes have been initialized.
            - The calculated boundary flows are based on the input flow rate ('input_v') and the entry nodes in the vasculature graph.
            - The calculated boundary flows are stored in the 'boundary_flows' attribute.
        """
        input_flows = None
        if self.entry_nodes is not None:
            input_flows = [self.parameters["input_v"]] * len(self.entry_nodes)
        self.boundary_flows = bloodflow.boundary_flows_A_based(
            graph=self.graph, entry_nodes=self.entry_nodes, input_flows=input_flows
        )

    @utils.logs_decorator
    def update_static_flow(self):
        """Update flow and volume for quasi-static computation.

        This method updates the flow and volume properties of the vasculature graph based on the provided
        boundary flows, blood viscosity, and base pressure. It is used for quasi-static blood flow computations.

        Note:
            - The method takes boundary flows, blood viscosity, and base pressure into account to compute
              updated flow and volume properties for the vasculature.
            - It is typically used in the context of multiscale simulations involving blood flow.
        """
        bloodflow.update_static_flow_pressure(
            graph=self.graph,
            input_flow=self.boundary_flows,
            params=self.parameters,
        )

    @utils.logs_decorator
    def set_radii(self, idxs: list[int], vals: list[float]) -> None:
        """Set radii for vasculature sections.

        This method allows you to set radii for specific vasculature sections identified by their indices.

        Args:
            idxs: A list of vasculature section indices to set the radii for.
            vals: A list of radii corresponding to the specified vasculature sections.

        Returns:
            None

        Note:
            - The method calculates equivalent radii to distribute across the specified sections while maintaining volume.
            - The updated radii are stored in the vasculature graph's edge properties.
        """

        def eq_radii(v):
            r"""Compute x from: 1/x^2 = \sum_n 1/(n*r_i^2)"""
            v = np.array(v)
            return np.sqrt(v.dot(v) / len(v))

        # I did not find a more pythonic way of doing this. I am open to suggestions
        d = defaultdict(list)
        for idx, v in zip(idxs, vals):
            d[self.graph.edge_properties.index[idx]].append(v)

        d = {k: eq_radii(v) for k, v in d.items()}
        # without interventions, d.keys() and d.values() are ordered in the samw way
        self.graph.edge_properties.loc[list(d.keys()), "radius"] = list(d.values())
