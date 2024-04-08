import json
import logging
import time
from pathlib import Path

import numpy as np

from . import utils
from .data import DATA_DIR, DEFAULT_CIRCUIT


class MsrConfigException(Exception):
    """General error for the config object"""


class MsrConfig(dict):
    """Multiscale run Config class"""

    def __init__(self, path=None):
        """Multiscale run Config constructor

        This class is composed from a chain of json files. We start from "config_path" which can
        be provided or deducted from the environment. We look for a file named: <config_path>/msr_config.json.
        This provides the first hook. We load the file as a dict (child) and look recursively if there is a
        "parent_config_path" marked. In that case, we add that dict as parent and merge them using the
        priority rules of utils.merge_dicts.

        All the paths are PosixPaths at the end.
        There is no check if the paths really exist except for the various config paths.

        Args:
          path: The path to the top configuration:

            * if `None`, then a file "msr_config.json" is expected to be found
              in the current working directory.
            * otherwise, if this is a `pathlib.Path` instance pointing to a
              directory, then a file "msr_config.json" is expected to be found
              in this directory.
            * Otherwise, if this is a `pathlib.Path` instance to a file, it is
              considered to be the JSON file to load.
        """

        if isinstance(path, str):
            path = Path(path)

        if path is None:
            path = Path.cwd()

        if not isinstance(path, Path):
            raise TypeError("Expected type are str, pathlib.Path")

        if path.resolve().is_dir():
            path /= "simulation_config.json"

        self.config_path = path
        self._load()

    @classmethod
    def _from_dict(cls, data):
        obj = cls.__new__(cls, data)
        super(MsrConfig, obj).__init__(data)
        return obj

    def __getattr__(self, key: str):
        """
        Provide attribute access to configuration values.

        This method allows you to access configuration values as attributes of the 'MsrConfig' object.
        If a configuration key exists, you can retrieve its value using attribute-style access.

        It automatically converts:

            - dict to MsrConfig.
            - lists to lists of MsrConfigs when possible.
            - strings that represents a path (key: *_path) to `pathlib.Path`.

        Args:
            key: The name of the configuration key to access.

        Returns:
            Any: The value associated with the specified configuration key.

        Raises:
            AttributeError: If the specified key does not exist in the configuration.

        Example:
            >>> value = config.some_key.some_other_key
        """

        if key in self:
            if isinstance(self[key], dict):
                self[key] = MsrConfig._from_dict(self[key])
            if key.endswith("_path") and isinstance(self[key], str):
                self[key] = Path(self[key])
            if isinstance(self[key], list) and any(
                isinstance(item, dict) for item in self[key]
            ):
                self[key] = [
                    MsrConfig._from_dict(i) if isinstance(i, dict) else i
                    for i in self[key]
                ]

            return self[key]
        raise AttributeError(
            f"'MsrConfig' object has no attribute '{key}'. Available keys: {', '.join(self.keys())}"
        )

    def __setattr__(self, key, value):
        """
        Set configuration values using attributes.

        This method allows you to set configuration values using attributes of the 'MsrConfig' object.
        When you assign a value to an attribute, it is stored as a configuration key-value pair.

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

        Example::

            >>> for key, value in config.items():
            ...     print(key, value)
        """
        for key in self:
            yield key, getattr(self, key)

    def values(self):
        """
        Generate values from the configuration.

        This method iterates over the configuration and generates the values associated with each key.
        It can be used when you only need to access the values in the configuration.

        Yields:
            Any: The value associated with a specific configuration key.

        Example::

            >>> for value in config.values():
            ...     print(value)
        """

        for key in self:
            yield getattr(self, key)

    @staticmethod
    def dump(config_path, to_path, replace_dict, indent=4):
        """Convenience function to dump the config in a file, collapsing the jsons in case it is necessary

        No substitutions are performed except for the parent path. Parent path is removed.
        """
        d = utils.load_jsons(config_path)
        d = utils.merge_dicts(child=replace_dict, parent=d)
        with open(to_path, "w") as file:
            json.dump(d, file, indent=indent)

    def _load(self):
        """
        Convenience function to load the configuration files recursively.

        This method is a convenience function that triggers the recursive
        loading of configuration files to compose the final configuration.
        It processes the JSON files, looks for parent configurations, and
        resolves relative paths.

        """
        d = utils.load_jsons(self.config_path)
        if not "multiscale_run" in d:
            raise MsrConfigException(
                f"Missing top-level 'multiscale_run' attribute in config file: '{self.config_path}'"
            )

        d.setdefault("multiscale_run", {})["pkg_data_path"] = str(DATA_DIR)
        utils.resolve_replaces(d)
        self.update(d)

        # get msr_dts and fix dts
        self.compute_multiscale_run_ndts()

    def is_steps_active(self):
        """Convenience function to check if a steps is active"""
        if "multiscale_run" not in self or "with_steps" not in self.multiscale_run:
            return False
        return self.multiscale_run.with_steps

    def is_bloodflow_active(self):
        """Convenience function to check if a bloodflow is active"""
        if "multiscale_run" not in self or "with_bloodflow" not in self.multiscale_run:
            return False
        return self.multiscale_run.with_bloodflow

    def is_metabolism_active(self):
        """Convenience function to check if a metabolism is active"""
        if "multiscale_run" not in self or "with_metabolism" not in self.multiscale_run:
            return False
        return self.multiscale_run.with_metabolism

    def is_manager_active(self, manager_name: str):
        """Convenience function to check if a manager is active"""
        if "multiscale_run" not in self:
            return False
        if manager_name == "neurodamus":
            return True
        return self.multiscale_run.get(f"with_{manager_name}", False)

    @property
    def neurodamus_dt(self):
        """neurodamus dt"""
        if "run" in self and "dt" in self.run:
            return self.run.dt
        raise MsrConfigException(
            f"Missing 'run.dt' attribute in config file: '{self.config_path}'"
        )

    @property
    def multiscale_run_dt(self):
        """multiscale run dt. Computed based on the other dts."""
        if "multiscale_run" in self and "ndts" in self.multiscale_run:
            return self.multiscale_run.ndts * self.neurodamus_dt
        raise None

    @property
    def steps_dt(self):
        """steps dt. It is a multiple of neurodamus dts."""
        if self.is_steps_active():
            # raise the usual errors if the manager is active but we cannot access ndts
            return self.multiscale_run.steps.ndts * self.neurodamus_dt
        return None

    @property
    def bloodflow_dt(self):
        """bloodflow dt. It is a multiple of neurodamus dts."""
        if self.is_bloodflow_active():
            # raise the usual errors if the manager is active but we cannot access ndts
            return self.multiscale_run.bloodflow.ndts * self.neurodamus_dt
        return None

    @property
    def metabolism_dt(self):
        """metabolism dt. It is a multiple of neurodamus dts."""
        if self.is_metabolism_active():
            # raise the usual errors if the manager is active but we cannot access ndts
            return self.multiscale_run.metabolism.ndts * self.neurodamus_dt
        return None

    def compute_multiscale_run_ndts(self):
        """
        Compute multiscale run n dts based on the active steps, metabolism, and bloodflow ndts.

        This method calculates the number of neurodamus dts required to synchronize simulations
        based on active simulation steps (if enabled), metabolism, and bloodflow.

        """

        msr_conf = self.multiscale_run
        if self.is_metabolism_active() and self.is_steps_active():
            if msr_conf.steps.ndts > msr_conf.metabolism.ndts:
                logging.info(
                    f"steps.ndts reduced to match metabolism: {msr_conf.steps.ndts} -> {msr_conf.metabolism.ndts}"
                )
                msr_conf.steps.ndts = msr_conf.metabolism.ndts

        if self.is_metabolism_active() and self.is_bloodflow_active():
            if msr_conf.bloodflow.ndts > msr_conf.metabolism.ndts:
                logging.info(
                    f"bloodflow.ndts reduced to match metabolism: {msr_conf.bloodflow.ndts} -> {msr_conf.metabolism.ndts}"
                )
                msr_conf.bloodflow.ndts = msr_conf.metabolism.ndts

        l = [
            msr_conf[i]["ndts"]
            for i in ["steps", "metabolism", "bloodflow"]
            if self.is_manager_active(i)
        ]

        if "ndts" in msr_conf:
            l.append(msr_conf.ndts)

        self["multiscale_run"]["ndts"] = int(np.gcd.reduce(l if len(l) else 10000))

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

        s = f"""
    -----------------------------------------------------
    --- MSR CONFIG ---
{json.dumps(utils.json_sanitize(self), indent=4)}
    --- MSR CONFIG ---
    -----------------------------------------------------
    """
        return s

    def dt_info(self) -> str:
        """
        Info about the various dts of the simulation. If the simulator is inactive its dt is none.

        Returns:
            str: A string containing the a str with DTS information.

        Example:
            >>> dt_info_str = config.dt_info()
            >>> print(dt_info_str)
        """

        s = f"""
    -----------------------------------------------------
    --- DTS ---
    neurodamus dt: {self.neurodamus_dt} ms
    steps dt: {self.steps_dt} ms
    bloodflow dt: {self.bloodflow_dt} ms
    metabolism dt: {self.metabolism_dt} ms
    multiscale run dt: {self.multiscale_run_dt} ms
    SIM_END: {self.run.tstop} ms
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
