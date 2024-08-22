import json
import logging
from pathlib import Path
import textwrap

import numpy as np
import trimesh
from bluepysnap import Circuit

from . import utils


class MsrMeshError(utils.MsrException):
    """Generic Mesh Exception"""


class MsrPreprocessor:
    """Preprocess manager for mesh generation in the Multi-Step Simulation Platform (MSP).

    The MsrPreprocessor class is responsible for various preprocessing tasks related
    to mesh generation and node set configuration for simulations in the Multi-Step
    Simulation Platform (MSP). It provides methods for automatically generating node
    sets, extracting information from a circuit configuration, and generating and
    refining mesh files for simulations.

    Attributes:
        config: A configuration object that holds various settings and file paths for preprocessing tasks.

    Note:
        - This class works in the context of the MSP, which involves neuron and vasculature simulations.
        - It is used for creating the necessary input files for simulations, such as mesh files and node set configurations.
    """

    def __init__(self, config):
        self.config = config

    def _check_filename_existence(self, path):
        """Check the existence of a file and determine whether to auto-generate it.

        This method checks whether a file at the specified path exists and returns the path for
        auto-generation if the file is missing. It also logs whether the file exists or not.

        Args:
            path (str): The path to the file.

        Returns:
            str: The path for auto-generation if the file is missing, or None if the file exists.

        Example:
            >>> result = _check_filename_existence('/path/to/file.json')
        """
        if Path(path).exists():
            logging.info(f"Use existing {path}")
            return None

        logging.info(f"{path} is missing. Auto-generate it")
        self.extract_information_from_circuit()
        return path

    @utils.logs_decorator
    def autogen_node_sets(self):
        """Generate node sets and save them as a JSON file.

        This method generates node sets based on selected neurons and astrocytes and saves the configuration as a JSON file.
        Node sets are defined using a template, and the selected neuron and astrocyte IDs are included in the configuration.

        Note:
            - Node sets are generated according to the provided template.
            - The selected neuron and astrocyte IDs are included in the node sets configuration.
            - The resulting configuration is saved as a JSON file specified in 'config.node_sets_path.'

        Example:
            >>> gen_node_sets()
        """
        output_filename = self._check_filename_existence(
            Path(self.config.node_sets_file)
        )
        if output_filename is None:
            return

        # Generate the node_sets.json
        template = {
            "testNGVSSCX_AstroMini": ["testNGVSSCX", "Astrocytes"],
            "src_cells": {"population": "All", "node_id": None},
            "testNGVSSCX": {"population": "All", "node_id": None},
            "Astrocytes": {"population": "astrocytes", "node_id": None},
        }

        template["src_cells"]["node_id"] = self.selected_neurons.tolist()
        template["testNGVSSCX"]["node_id"] = self.selected_neurons.tolist()
        template["Astrocytes"]["node_id"] = self.selected_astrocytes.tolist()

        with open(output_filename, "w") as fout:
            json.dump(template, fout, indent=4)

    @utils.logs_decorator
    def extract_information_from_circuit(self):
        """Extract information from the circuit configuration.

        This method is a helper function for generating node sets and is used to extract relevant information from the
        circuit configuration (ngv_config.json). It retrieves data about neurons and astrocytes based on the provided
        neuron population name and optional filtering criteria.

        Sets Variables:
            - neuro_df (pandas DataFrame): DataFrame with information about selected neurons.
            - selected_neurons (numpy array): Array containing the selected neuron IDs.
            - selected_astrocytes (numpy array): Array containing the selected astrocyte IDs.

        Note:
            - The method reads the circuit configuration specified in the 'config.circuit_path'.
            - It identifies astrocytes with valid endfeet and selected neurons based on the neuron population name.
            - The 'filter_neuron' flag determines whether to filter neurons connected to astrocytes.
            - Extracted data is stored in the class attributes 'neuro_df,' 'selected_neurons,' and 'selected_astrocytes.'

        Authors: Jean, Katta
        """
        if hasattr(self, "neuro_df"):
            return

        pop_name = (
            self.config.multiscale_run.preprocessor.node_sets.neuron_population_name
        )

        c = Circuit(str(self.config.config_path.parent / self.config.network))

        # Create a list of astrocyte ids, that contains all the astrocytes with endfoot
        gliovascular = c.edges["gliovascular"]
        edges_ids = np.arange(gliovascular.size, dtype=np.uint64)
        df = gliovascular.get(edges_ids, ["@target_node", "endfoot_compartment_length"])
        filtered_df = df[df.endfoot_compartment_length > 0]
        selected_astrocytes = filtered_df["@target_node"].unique()

        # Remove from thi list the astrocytes with at least one endfoot_compartment_length == 0.0
        filtered_df = df[df.endfoot_compartment_length == 0]
        astroytes_id_to_remove = filtered_df["@target_node"].to_numpy()
        indices_to_remove = np.where(
            np.in1d(selected_astrocytes, astroytes_id_to_remove)
        )[0]
        selected_astrocytes = np.delete(selected_astrocytes, indices_to_remove)
        logging.info(
            f"There are {selected_astrocytes.size} astrocytes with valid endfeet"
        )

        if self.config.multiscale_run.preprocessor.node_sets.filter_neuron:
            neuroglial = c.edges["neuroglial"]
            edges_ids = np.arange(neuroglial.size, dtype=np.uint16)
            df = neuroglial.get(edges_ids, ["@source_node", "@target_node"])
            selected_neurons = df[df["@source_node"].isin(selected_astrocytes)][
                "@target_node"
            ].unique()
            neuro_df = c.nodes[pop_name].get(selected_neurons)

        else:
            selected_neurons = np.arange(c.nodes[pop_name].size, dtype=np.uint16)
            neuro_df = c.nodes[pop_name].get(selected_neurons)

        logging.info(f"There are {selected_neurons.size} selected neurons")

        neuro_df = neuro_df.rename(columns={"population": "population_name"})
        neuro_df = neuro_df.reset_index(drop=False)

        self.neuro_df = neuro_df
        self.selected_neurons = selected_neurons
        self.selected_astrocytes = selected_astrocytes


    @utils.logs_decorator
    def _gen_bbox_msh(self, pts):
        length = (
            self.config.multiscale_run.preprocessor.mesh.base_length
        )  # Adjust as needed
        phys_vol = self.config.multiscale_run.steps.compname
        refinement_steps = self.config.multiscale_run.preprocessor.mesh.refinement_steps

        bb = utils.bbox(pts)

        # Your Geo string containing the geometric description
        newline = "\n            "
        geo_string = textwrap.dedent(f"""\
            Mesh.CharacteristicLengthFactor = 10;

            Mesh.MshFileVersion = 4.1;
            Mesh.PartitionOldStyleMsh2 = 1;
            Mesh.PartitionCreateGhostCells = 1;

            Point(1) = {{{bb[0][0]}, {bb[0][1]}, {bb[0][2]}, {length}}};
            Point(2) = {{{bb[0][0]}, {bb[0][1]}, {bb[1][2]}, {length}}};
            Point(3) = {{{bb[0][0]}, {bb[1][1]}, {bb[0][2]}, {length}}};
            Point(4) = {{{bb[0][0]}, {bb[1][1]}, {bb[1][2]}, {length}}};
            Point(5) = {{{bb[1][0]}, {bb[0][1]}, {bb[0][2]}, {length}}};
            Point(6) = {{{bb[1][0]}, {bb[0][1]}, {bb[1][2]}, {length}}};
            Point(7) = {{{bb[1][0]}, {bb[1][1]}, {bb[0][2]}, {length}}};
            Point(8) = {{{bb[1][0]}, {bb[1][1]}, {bb[1][2]}, {length}}};

            Line(1) = {{1, 2}};
            Line(2) = {{1, 3}};
            Line(3) = {{1, 5}};
            Line(4) = {{3, 4}};
            Line(5) = {{4, 2}};
            Line(6) = {{2, 6}};
            Line(7) = {{6, 5}};
            Line(8) = {{5, 7}};
            Line(9) = {{7, 8}};
            Line(10) = {{8, 6}};
            Line(11) = {{8, 4}};
            Line(12) = {{7, 3}};

            Curve Loop(1) = {{2, -12, -8, -3}};
            Plane Surface(1) = {{1}};
            Curve Loop(2) = {{8, 9, 10, 7}};
            Plane Surface(2) = {{2}};
            Curve Loop(3) = {{10, -6, -5, -11}};
            Plane Surface(3) = {{3}};
            Curve Loop(4) = {{5, -1, 2, 4}};
            Plane Surface(4) = {{4}};
            Curve Loop(5) = {{12, 4, -11, -9}};
            Plane Surface(5) = {{5}};
            Curve Loop(6) = {{3, -7, -6, -1}};
            Plane Surface(6) = {{6}};

            Surface Loop(1) = {{1, 4, 3, 2, 5, 6}};
            Volume(1) = {{1}};

            Physical Volume("{phys_vol}", 1) = {{1}};
            {newline.join(["RefineMesh;"] * refinement_steps)}
            Mesh 3;
            Save "{mesh_path}";
        """)

        mesh_path = self.config.multiscale_run.mesh_path
        geo_path = mesh_path.with_suffix(".geo")
        with geo_path.open("w") as geo_file:
            geo_file.write(geo_string)
        logging.info(f"Creating mesh '{mesh_path}' with gmsh utility")
        subprocess.check_call(["gmsh", str(geo_path)])


    @staticmethod
    def _explode_pts(pts, explode_factor):
        return explode_factor * pts + (1.0 - explode_factor) * np.mean(pts, axis=0)

    def _gen_pts(self, ndam_m=None, bf_m=None, pts=None):
        """Generate points for mesh generation.

        This method generates points for mesh generation by combining points from multiple sources, such as neurodamus
        and bloodflow data, and optionally additional custom points provided as input. The points can be scaled and centered
        based on configuration parameters.

        Args:
            ndam_m (object): An object that provides neurodamus data, or None if not available.
            bf_m (object): An object that provides bloodflow data, or None if not available.
            pts (list or array): Additional custom points to include in the mesh, if provided.

        Returns:
            numpy.ndarray: An array containing the combined points for mesh generation.

        Note:
            The method first fetches points from neurodamus and bloodflow data (if available) and scales them. If custom points
            are provided, they are also included in the combined set of points. The combined points are scaled and centered
            based on the `scaling_factor` configuration parameter.

        Example:
            >>> pts = _gen_pts(ndam_m=ndam_m, bf_m=bf_m, pts=custom_pts)
        """
        pts = pts if pts is not None else []
        scale = 1
        bf_pts = bf_m.get_seg_points(scale=scale) if bf_m else []
        if ndam_m:
            ndam_pts = ndam_m.get_seg_points(scale=scale)
            ndam_pts = utils.comm().gather(ndam_pts, root=0)
        else:
            ndam_pts = []

        if utils.rank0():
            ndam_pts = np.array([l for i in ndam_pts for j in i for l in j])

            pts = np.vstack([i for i in [pts, bf_pts, ndam_pts] if len(i)])

            logging.info(f"npts: {pts.shape[0]}")

            # scaling: points should encompass a zone slightly larger
            pts = MsrPreprocessor._explode_pts(
                pts, self.config.multiscale_run.preprocessor.mesh.explode_factor
            )
        return pts

    @property
    def ntets(self):
        """Count the total number of tets in the mesh"""
        gmsh.initialize()
        try:
            gmsh.open(str(self.config.multiscale_run.mesh_path))

            _, tets, _ = gmsh.model.mesh.getElements(dim=3)
            return sum([len(i) for i in tets])

        finally:
            gmsh.finalize()

    @utils.logs_decorator
    def check_mesh(self):
        if utils.rank0():
            n = self.ntets
            if n < utils.size():
                raise MsrMeshError(
                    f'There are less tetrahedrons than MPI ranks ({n} < {utils.size()}) and this is incompatible with Omega_h. Please increase "refinement_steps" in simulation_config.json or reduce the number of MPI ranks.'
                )

    @utils.logs_decorator
    def autogen_mesh(self, ndam_m=None, bf_m=None, pts=None):
        """Generate a STEPS mesh when it is missing based on neuron and vasculature points.

        This method creates a STEPS mesh file if it doesn't already exist. It uses neuron and vasculature points
        to generate the mesh. The scale parameter is set to 1 by default, as the code already accounts for rescaling.

        Args:
            ndam_m (object): An object that provides neurodamus data, or None if not available.
            bf_m (object): An object that provides bloodflow data, or None if not available.
            pts (list or array): Additional custom points to include in the mesh, if provided.

        Note:
            The method utilizes the `neurodamus_manager` and `bloodflow_manager` to obtain neuron and vasculature points.
            It scales the points slightly to generate a slightly larger mesh, and it logs the points used for the convex hull
            as well as the mesh generation process.

        Example:
            >>> gen_mesh()
        """
        pts = pts if pts is not None else []
        mesh_path = self.config.multiscale_run.mesh_path
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        if mesh_path.exists():
            logging.info(f"Use existing steps mesh: {mesh_path}")
            return

        pts = self._gen_pts(ndam_m=ndam_m, bf_m=bf_m, pts=pts)
        if utils.rank0():
            self._gen_bbox_msh(pts)
