import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils
from .data import DATA_DIR, DEFAULT_CIRCUIT


class MsrConfig(dict):
    """Multiscale run Config class"""

    def __init__(self, config_path_or_dict=None):
        """Multiscale run Config constructor

        This class is composed from a chain of json files. We start from "config_path" which can
        be provided or deducted from the environment. We look for a file named: <config_path>/msr_config.json.
        This provides the first hook. We load the file as a dict (child) and look recursively if there is a
        "parent_config_path" marked. In that case, we add that dict as parent and merge them using the
        priority rules of utils.merge_dicts.

        All the paths are PosixPaths at the end.
        There is no check if the paths really exist except for the various config paths.

        Args:
          config_path_or_dict: The path to the top configuration
          - if `None`, then a file "msr_config.json" is expected to be found in the
          current working directory.
          - otherwise, if this is a `pathlib.Path` instance pointing to a directory,
          then a file "msr_config.json" is expected to be found in this directory.
          - Otherwise, if this is a `pathlib.Path` instance to a file, it is considered
          to be the JSON file to load.
          - Finally, the argument is expected to be a `dict` instance representing the configuration to use.
        """

        if isinstance(config_path_or_dict, dict):
            self.update(config_path_or_dict)
            return
        elif isinstance(config_path_or_dict, str):
            config_path_or_dict = Path(config_path_or_dict)

        if config_path_or_dict is None:
            config_path_or_dict = Path.cwd()

        if not isinstance(config_path_or_dict, Path):
            raise TypeError("Expected type are str, pathlib.Path, or dict")

        if config_path_or_dict.resolve().is_dir():
            config_path_or_dict /= "msr_config.json"

        self._load(config_path_or_dict)

    def __getattr__(self, key):
        """
        Provide attribute access to configuration values.

        This method allows you to access configuration values as attributes of the 'MsrConfig' object. If a configuration key exists, you can retrieve its value using attribute-style access.

        Args:
            key (str): The name of the configuration key to access.

        Returns:
            Any: The value associated with the specified configuration key.

        Raises:
            AttributeError: If the specified key does not exist in the configuration.

        Example:
            >>> value = config.some_key
        """

        if key in self:
            if isinstance(self[key], dict):
                self[key] = MsrConfig(self[key])
            if key.endswith("_path") and isinstance(self[key], str):
                self[key] = Path(self[key])

            return self[key]
        raise AttributeError(
            f"'MsrConfig' object has no attribute '{key}'. Available keys: {', '.join(self.keys())}"
        )

    def __setattr__(self, key, value):
        """
        Set configuration values using attributes.

        This method allows you to set configuration values using attributes of the 'MsrConfig' object. When you assign a value to an attribute, it is stored as a configuration key-value pair.

        Args:
            key (str): The name of the configuration key to set.
            value (Any): The value to associate with the specified configuration key.

        Example:
            >>> config.some_key = value
        """

        if isinstance(value, Path):
            self[key] = str(value)
            return
        self[key] = value

    def items(self):
        """
        Generate key-value pairs from the configuration.

        This method iterates over the configuration and generates key-value pairs, which can be used in various contexts where iteration is required.

        Yields:
            Tuple: A tuple containing a key-value pair, where the first element is the key (attribute) and the second element is the corresponding value.

        Example:
            >>> for key, value in config.items():
            ...     print(key, value)
        """
        for key in self:
            yield key, getattr(self, key)

    def values(self):
        """
        Generate values from the configuration.

        This method iterates over the configuration and generates the values associated with each key. It can be used when you only need to access the values in the configuration.

        Yields:
            Any: The value associated with a specific configuration key.

        Example:
            >>> for value in config.values():
            ...     print(value)
        """

        for key in self:
            yield getattr(self, key)

    def merge_without_priority(self, d):
        """
        Add a dictionary to the configuration without overriding existing entries.

        This method allows you to merge an external dictionary into the configuration. It adds new key-value pairs from the provided dictionary to the configuration, but it does not override existing keys.

        Args:
            d (dict): A dictionary to be added to the configuration.

        Example:
            >>> config.merge_without_priority({"new_key": "new_value"})
        """

        self.update({k: v for k, v in d.items() if k not in self})

    @staticmethod
    def dump(config_path, to_path, replace_dict, indent=4):
        """Convenience function to dump the config in a file, collapsing the jsons in case it is necessary

        No sbustitutions are performed except for the parent path. Parent path is removed.
        """
        d = utils.load_jsons(config_path, parent_path_key="parent_config_path")
        d = utils.merge_dicts(child=replace_dict, parent=d)
        with open(to_path, "w") as file:
            json.dump(d, file, indent=indent)

    def _load(self, config_path):
        """
        Convenience function to load the configuration files recursively.

        This method is a convenience function that triggers the recursive
        loading of configuration files to compose the final configuration.
        It processes the JSON files, looks for parent configurations, and
        resolves relative paths.

        """
        d = utils.load_jsons(config_path, parent_path_key="parent_config_path")
        base_path = Path(d["base_path"])
        if not base_path.is_absolute():
            d["base_path"] = str(config_path.parent / base_path)
        d["pkg_data_path"] = str(DATA_DIR)
        d["timestr"] = time.strftime("%Y%m%d%H")
        utils.resolve_replaces(d)
        self.update(d)

        # finalize computing a few additional entries
        if "msr_debug" in self and self.msr_debug:
            self.debug_overrides()

        # get msr_dts and fix dts
        self.compute_msr_ndts()
        # do it again, after the overrides
        self.compute_msr_ndts()

    def debug_overrides(self):
        """
        Set debug-specific configuration overrides.

        This method sets debug-specific configuration overrides by modifying specific configuration entries. It is intended for debugging and development purposes.

        Example:
            >>> config.debug_overrides()
        """

        self.metabolism_ndts = 10
        self.bloodflow_ndts = 10

    def dt(self, token="neurodamus"):
        """
        Get the time step (dt) for a given token.

        Parameters:
        - token (str, optional): Token specifying the type of time step.
        Defaults to "neurodamus".

        Returns:
        - float: The time step for the specified token.
        """
        if token == "neurodamus":
            return self.DT

        ndts = getattr(self, f"{token}_ndts")
        return ndts * self.DT

    def compute_msr_ndts(self):
        """
        Compute multiscale run n dts based on the active steps, metabolism, and bloodflow ndts.

        This method calculates the number of neurodamus dts required to synchronize simulations based on active simulation steps (if enabled), metabolism, and bloodflow.

        Note:
            - The 'msr_ndts' attribute is updated with the calculated value.

        Example:
            >>> config.compute_msr_ndts()
        """

        if "metabolism_ndts" in self:
            if "steps_ndts" in self and self.steps_ndts > self.metabolism_ndts:
                logging.info(
                    f"steps_ndts reduced to match metabolism: {self.steps_ndts} -> {self.metabolism_ndts}"
                )
                self.steps_ndts = self.metabolism_ndts
            if "bloodflow_ndts" in self and self.bloodflow_ndts > self.metabolism_ndts:
                logging.info(
                    f"bloodflow_ndts reduced to match metabolism: {self.bloodflow_ndts} -> {self.metabolism_ndts}"
                )
                self.bloodflow_ndts = self.metabolism_ndts

        l = []
        if "with_steps" in self and self.with_steps:
            l.append(self.steps_ndts)
        if "with_metabolism" in self and self.with_metabolism:
            l.append(self.metabolism_ndts)
        if "with_bloodflow" in self and self.with_bloodflow:
            l.append(self.bloodflow_ndts)

        if "msr_ndts" in self:
            l.append(self.msr_ndts)

        self.msr_ndts = int(np.gcd.reduce(l if len(l) else 10000))

    def __str__(self):
        """
        Convert the configuration to a formatted string.

        This method generates a formatted string representation of the configuration. It's useful for printing the configuration for inspection and debugging.

        Returns:
            str: A formatted string representing the configuration.

        Example:
            >>> config_str = str(config)
            >>> print(config_str)
        """
        d = {
            k: self[k] if not isinstance(self[k], Path) else str(self[k]) for k in self
        }

        s = f"""
    -----------------------------------------------------
    --- MSR CONFIG ---
{json.dumps(d, indent=4)}
    --- MSR CONFIG ---
    -----------------------------------------------------
    """
        return s

    def dt_info(self):
        """
        Return a DataFrame of DTS (Delta Time Step) information.

        This method generates a DataFrame with information about delta time steps (DTS) for different simulation components, such as neurodamus, metabolism, bloodflow, and more.

        Returns:
            str: A string containing the DataFrame with DTS information.

        Example:
            >>> dt_info_str = config.dt_info()
            >>> print(dt_info_str)
        """

        def get_line(v):
            ndts = getattr(self, f"{v}_ndts")
            return {"DT (ms)": self.dt(v), "ndts": ndts}

        data = {}

        data["ndam"] = {
            "DT (ms)": self.DT,
        }

        data["msr"] = get_line("msr")

        if self.with_steps:
            data["steps"] = get_line("steps")

        if self.with_steps:
            data["metabolism"] = get_line("metabolism")

        if self.with_steps:
            data["bloodflow"] = get_line("bloodflow")

        df = pd.DataFrame(data).T

        s = f"""
    -----------------------------------------------------
    --- DTS ---
{df}\n
SIM_END: {self.msr_sim_end} ms
    --- DTS ---
    -----------------------------------------------------
    """

        return s

    @classmethod
    def rat_sscxS1HL_V6(cls):
        """
        Returns:
            MsrConfig: Default configuration using the rat v6 circuit
        """
        return cls(DEFAULT_CIRCUIT)
