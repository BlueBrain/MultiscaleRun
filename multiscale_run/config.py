import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils
from .data import DATA_DIR, DEFAULT_CIRCUIT


class MsrConfig(dict):
    """Multiscale run Config class

    This class is composed from a chain of json files. We start from "base_path" which can
    be provided or deducted from environment. We look for a file named: <base_path>/msr_config.json.
    This provides the first hook. We load the file as a dict (child) and look recursively if there is a
    "parent_config_path" marked. In that case we add that dict as parent and merge them using the
    priority rules of utils.merge_dicts. After all of that, the assigned env variables that match
    the entries of this class are replaced.

    All the paths are PosixPaths at the end.
    There is no check if the paths really exist except for the various config paths.

    We use json files (over json) because in this way we can have proper comments
    """

    def __init__(self, base_path_or_dict=None):
        """Multiscale run Config class

        This class is composed from a chain of json files. We start from "base_path" which can
        be provided or deducted from the environment. We look for a file named: <base_path>/msr_config.json.
        This provides the first hook. We load the file as a dict (child) and look recursively if there is a
        "parent_config_path" marked. In that case, we add that dict as parent and merge them using the
        priority rules of utils.merge_dicts.

        All the paths are PosixPaths at the end.
        There is no check if the paths really exist except for the various config paths.

        We use json files (over json) because in this way we can have proper comments
        """

        if isinstance(base_path_or_dict, dict):
            self.update(base_path_or_dict)
            return

        self.cwd_path = Path.cwd()

        self.base_path = base_path_or_dict
        if self.base_path is None:
            self.base_path = utils.load_from_env(
                "base_path",
                (self.cwd_path / "config"),
                lambda x: str(x),
            )
        self.base_path = Path(self.base_path)
        self.config_path = self.base_path / "msr_config.json"

        self.load()

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
                return MsrConfig(self[key])
            if key.endswith("_path") and isinstance(self[key], str):
                return Path(self[key])

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

    def _get_dict_from_json(self, path, rd):
        """
        Recursively load and compose a dictionary from JSON files with priority merging.

        This method recursively loads JSON configuration files and composes a dictionary. The merging of configurations
        follows the priority dictated by the 'utils.merge_dicts' function, giving higher priority to 'parent_config_path'
        in the JSON files.

        Args:
            path (Path): The path of the current configuration file to load and merge.

        Returns:
            dict: A dictionary representing the composed configuration.

        Note:
            - The method recursively processes JSON files, starting from the specified 'path.'
            - Configuration merging considers the presence of a 'parent_config_path' in the JSON files, which allows for
            inheritance of settings from a parent configuration.
            - The 'base_path' is used to resolve relative paths within the loaded configuration.

        Example:
            >>> config = _get_dict_from_json(Path("config.json"))
        """

        child = utils.get_dict_from_json(path)
        utils.recursive_replace(child, rd)

        parent = {}
        if "parent_config_path" in child:
            parent = self._get_dict_from_json(child["parent_config_path"], rd)
            utils.recursive_replace(parent, rd)
            del child["parent_config_path"]

        return utils.merge_dicts(parent=parent, child=child)

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

    def load(self):
        """
        Convenience function to load the configuration files recursively.

        This method is a convenience function that triggers the recursive loading of configuration files to compose the final configuration. It processes the JSON files, looks for parent configurations, and resolves relative paths.

        Example:
            >>> config.load()
        """
        rd = dict(
            base_path=str(self.base_path),
            pkg_data_path=str(DATA_DIR),
            timestr=time.strftime("%Y%m%d%H"),
        )
        d = self._get_dict_from_json(self.config_path, rd)
        self.update(d)

        # finalize computing a few additional entries
        if "mr_debug" in self and self.mr_debug:
            self.debug_overrides()

        # get msr_dts and fix dts
        self.compute_msr_ndts()
        self.env_overrides()
        # do it again, after the overrides
        self.compute_msr_ndts()

    def env_overrides(self):
        """
        Apply environment variable overrides to the dictionary-like object.

        This function iterates over the items in the dictionary-like object, and for each
        key-value pair, it attempts to load a corresponding value from the environment
        variable. If the environment variable exists, it replaces the original value in
        the object with the environment variable value, converting it to the same data
        type as the original value.

        Args:
            self (dict-like object): The dictionary-like object to which environment
                variable overrides will be applied.

        Returns:
            None

        Example:
            Suppose `self` is a dictionary-like object with key-value pairs, and you
            have environment variables set for the same keys. After calling
            `env_overrides`, the values in `self` will be updated with the values from
            the corresponding environment variables.

        Note:
            This function uses a lambda to convert environment variable values to the
            same data type as the original values in the dictionary-like object.

            Since bool('0') is True in python we need to use f

        """

        def f(base, a):
            if isinstance(base, bool):
                return utils.strtobool(a)
            return type(base)(a)

        for k in self.keys():
            self[k] = utils.load_from_env(k, self[k], lambda a: f(self[k], a))

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
        d = {k: self[k] for k in self}

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
