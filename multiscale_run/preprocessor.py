import logging
import numpy as np
import os
import json

from bluepysnap import Circuit

from . import utils


class MsrPreprocessor:
    def __init__(self, config):
        self.config = config

    @utils.logs_decorator
    def run(self):
        self.gen_node_sets()

    def _check_filename_existance(self, path):
        if os.path.exists(path):
            logging.info(f"Use existing node_sets.json")
            return None

        logging.info(f"{path} is missing. Auto-generate it")
        self.extract_information_from_circuit()
        return path

    @utils.logs_decorator
    def gen_node_sets(self):
        output_filename = self._check_filename_existance(self.config.node_sets_path)
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
        """
        Extract information from the circuit_config.json.
        This is a helper function for gen_node_sets.
        :param circuit_path: path to the somata circuit json file (ngv_config.json)
        :param neuron_population_name: the sonata neuron population name
        :param filter_neuron: If True, the neuron ids will contain only the neuron that are connected to the astrocytes,
                            otherwise, the list will contain all the neurons ids
        :set variables:
            neuro_df: pandas dataframe with all selected neuron infromation
            selected_neurons: numpy addary of shape (N,) that contains the neuron_ids
            selected_astrocytes

        :Authors: Jean, Katta
        """
        if hasattr(self, "neuro_df"):
            return

        c = Circuit(self.config.circuit_path)

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

        if self.config.filter_neuron:
            neuroglial = c.edges["neuroglial"]
            edges_ids = np.arange(neuroglial.size, dtype=np.uint16)
            df = neuroglial.get(edges_ids, ["@source_node", "@target_node"])
            selected_neurons = df[df["@source_node"].isin(selected_astrocytes)][
                "@target_node"
            ].unique()
            neuro_df = c.nodes[self.config.neuron_population_name].get(selected_neurons)

        else:
            selected_neurons = np.arange(
                c.nodes[self.config.neuron_population_name].size, dtype=np.uint16
            )
            neuro_df = c.nodes[self.config.neuron_population_name].get(selected_neurons)

        logging.info(f"There are {selected_neurons.size} selected neurons")

        neuro_df = neuro_df.rename(columns={"population": "population_name"})
        neuro_df = neuro_df.reset_index(drop=False)

        self.neuro_df = neuro_df
        self.selected_neurons = selected_neurons
        self.selected_astrocytes = selected_astrocytes
