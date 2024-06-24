import collections
import functools
import json
import logging
import textwrap
from pathlib import Path

import jsonschema
import numpy as np

from . import utils
from .data import CONFIG_DIR, DATA_DIR, MSR_CONFIG_JSON, MSR_SCHEMA_JSON


class NamedCircuit(
    collections.namedtuple(
        "NamedCircuit",
        ["path", "sbatch_parameters", "config_transform"],
        defaults=[None],
    )
):
    """Define an internal circuit available within the MultiscaleRun package.
    The circuit is defined through the following fields:

    * path: its path on the filesystem
    * sbatch_parameters: recommended SLURM parameters for the SBATCH script running its simulation
    * config_transform: optional callable to apply on the loaded simulation_config.json
    """

    def config(self):
        """
        Returns
          Instance of MsrConfig based on this circuit
        """
        conf = MsrConfig(self.path)
        if self.config_transform:
            conf = self.config_transform(conf)
        return conf

    def json(self):
        """
        Returns
          the raw JSON representation of this circuit
        """
        with open(self.path / MSR_CONFIG_JSON) as istr:
            conf = json.load(istr)
        if self.config_transform:
            conf = self.config_transform(conf)
        return conf


NAMED_CIRCUITS = {
    "rat_sscxS1HL_V6": NamedCircuit(
        path=CONFIG_DIR / "rat_sscxS1HL_V6",
        sbatch_parameters=dict(
            job_name="msr_ratV6",
            nodes=1,
            time="01:00:00",
        ),
    ),
    "rat_sscxS1HL_V10": NamedCircuit(
        path=CONFIG_DIR / "rat_sscxS1HL_V10",
        sbatch_parameters=dict(
            job_name="msr_ratV10",
            nodes=64,
            time="10:00:00",
        ),
    ),
    "rat_sscxS1HL_V10_CI": NamedCircuit(
        path=CONFIG_DIR / "rat_sscxS1HL_V10",
        sbatch_parameters=dict(
            job_name="msr_ratV10_CI",
            nodes=1,
            time="01:00:00",
        ),
        config_transform=functools.partial(
            utils.replace_values,
            replacements={
                "testNGVSSCX": "testNGVSSCX_CI",
                "testNGVSSCX_AstroMini": "testNGVSSCX_AstroMini_CI",
            },
        ),
    ),
}

DEFAULT_CIRCUIT = "rat_sscxS1HL_V10_CI"


class MsrConfigException(Exception):
    """General error for the config object"""


class MsrConfigSchemaError(MsrConfigException):
    """For invalid configuration in regard to the JSON Schema"""

    def __init__(self, ve: jsonschema.exceptions.ValidationError):
        super().__init__()
        self.ve = ve

    def __str__(self):
        msg = "in the following JSON object at location '"
        path = ["multiscale_run"] + list(self.ve.path)
        msg += ".".join(str(e) for e in path)
        msg += "':\n"

        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Path):
                    return str(obj)
                return super().default(obj)

        msg += textwrap.indent(
            json.dumps(self.ve.instance, indent=4, cls=Encoder), "  "
        )
        msg += "\nError: " + self.ve.message
        return msg


class MsrConfig(dict):
    """MultiscaleRun configuration class"""

    def __init__(self, path=None):
        """Multiscale run Config constructor

        This class is composed from a chain of json files. We start from "config_path" which can
        be provided or deducted from the environment. We look for a file named: <config_path>/simulation_config.json.
        This provides the first hook. We load the file as a dict (child) and look recursively if there is a
        "parent_config_path" marked. In that case, we add that dict as parent and merge them using the
        priority rules of utils.merge_dicts.

        All the paths are PosixPaths at the end.
        There is no check if the paths really exist except for the various config paths.

        Args:
          path: The path to the top configuration:

            * if `None`, then a file "simulation_config.json" is expected to be found
              in the current working directory.
            * otherwise, if this is a `pathlib.Path` instance pointing to a
              directory, then a file "simulation_config.json" is expected to be found
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
        """Provide attribute access to configuration values.

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
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'MsrConfig' object has no attribute '{key}'. Available keys: {', '.join(self.keys())}"
            )

    def items(self):
        """Generate key-value pairs from the configuration.

        This method iterates over the configuration and generates key-value pairs, which can be used in various contexts where iteration is required.

        Yields
            Tuple: A tuple containing a key-value pair, where the first element is the key (attribute) and the second element is the corresponding value.

        Example::

            >>> for key, value in config.items():
            ...     print(key, value)
        """
        for key in self:
            yield key, getattr(self, key)

    def values(self):
        """Generate values from the configuration.

        This method iterates over the configuration and generates the values associated with each key.
        It can be used when you only need to access the values in the configuration.

        Yields
            Any: The value associated with a specific configuration key.

        Example::

            >>> for value in config.values():
            ...     print(value)
        """
        for key in self:
            yield getattr(self, key)

    def _load(self):
        """Convenience function to load the configuration files recursively.

        This method is a convenience function that triggers the recursive
        loading of configuration files to compose the final configuration.
        It processes the JSON files, looks for parent configurations, and
        resolves relative paths.

        """
        with open(self.config_path) as istr:
            d = json.load(istr)
        if "multiscale_run" not in d:
            raise MsrConfigException(
                f"Missing top-level 'multiscale_run' attribute in config file: '{self.config_path}'"
            )

        d.setdefault("multiscale_run", {})["pkg_data_path"] = str(DATA_DIR)
        utils.resolve_replaces(d)
        self.update(MsrConfig._objectify_config(None, d))

        # get msr_dts and fix dts
        self.compute_multiscale_run_ndts()

    @classmethod
    def _objectify_config(cls, key, obj):
        """Internal method recursively transforming the 'multiscale_run' data
        from 'simulation_config.json' into a Python object. The following
        transformations are:

          * dictionaries are transformed to `MsrConfig` instances
          * string values whose keys are suffixed with "_path" are
          transformed to `pathlib.Path` instances

        Args:
          key: the closest key up in the tree
          obj: the JSON object to transform recursively

        Returns:
          The transformed object.
        """
        if isinstance(obj, dict):
            return MsrConfig._from_dict(
                dict(
                    (key, cls._objectify_config(key, value))
                    for key, value in obj.items()
                )
            )
        elif isinstance(obj, str) and key.endswith("_path"):
            return Path(obj)
        elif isinstance(obj, list):
            return list(cls._objectify_config(key, item) for item in obj)
        return obj

    def check(self, schema: Path = None):
        """Validate this configuration instance in regard to the MultiscaleRun SONATA extension data schema.

        Args:
          schema: If specified, overwrites the default path to
          the `JSON Schema <https://json-schema.org/>`_
          file used to validate the configuration.

        Raises:
          MsrConfigSchemaError: If 'simulation_config' section is not valid.
        """
        with open(schema or MSR_SCHEMA_JSON) as istr:
            schema = json.load(istr)
            if isinstance(schema.get("additionalProperties"), bool):
                schema["additionalProperties"] = {}
            schema.setdefault("additionalProperties", {}).update(
                pkg_data_path=dict(
                    type="string",
                    description="Location to the 'data' directory within the MultiscaleRun installation (property set internally)",
                ),
                ndts=dict(
                    type="integer",
                    description="Number of dts to perform (property computed internally)",
                ),
            )
        try:
            cls = jsonschema.validators.validator_for(schema)
            cls.check_schema(schema)
            ext_cls = jsonschema.validators.extend(
                cls,
                type_checker=cls.TYPE_CHECKER.redefine(
                    "string",
                    lambda checker, instance: isinstance(instance, (str, Path)),
                ),
            )
            validator = ext_cls(schema)
            error = jsonschema.exceptions.best_match(
                validator.iter_errors(self["multiscale_run"])
            )
            if error is not None:
                raise error
        except jsonschema.exceptions.ValidationError as ve:
            raise MsrConfigSchemaError(ve)

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
        """Neurodamus dt"""
        if "run" in self and "dt" in self.run:
            return self.run.dt
        raise MsrConfigException(
            f"Missing 'run.dt' attribute in config file: '{self.config_path}'"
        )

    @property
    def multiscale_run_dt(self):
        """Multiscale run dt. Computed based on the other dts."""
        if "multiscale_run" in self and "ndts" in self.multiscale_run:
            return self.multiscale_run.ndts * self.neurodamus_dt
        raise None

    @property
    def steps_dt(self):
        """Steps dt. It is a multiple of neurodamus dts."""
        if self.is_steps_active():
            # raise the usual errors if the manager is active but we cannot access ndts
            return self.multiscale_run.steps.ndts * self.neurodamus_dt
        return None

    @property
    def bloodflow_dt(self):
        """Bloodflow dt. It is a multiple of neurodamus dts."""
        if self.is_bloodflow_active():
            # raise the usual errors if the manager is active but we cannot access ndts
            return self.multiscale_run.bloodflow.ndts * self.neurodamus_dt
        return None

    @property
    def metabolism_dt(self):
        """Metabolism dt. It is a multiple of neurodamus dts."""
        if self.is_metabolism_active():
            # raise the usual errors if the manager is active but we cannot access ndts
            return self.multiscale_run.metabolism.ndts * self.neurodamus_dt
        return None

    def compute_multiscale_run_ndts(self):
        """Compute MultiscaleRun n dts based on the active simulators

        Calculates the number of Neurodamus dts required to synchronize simulations.
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
        """Convert the configuration to a formatted string.

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
        """Info about the various dts of the simulation. If the simulator is inactive its dt is none.

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
    def default(cls, **kwargs):
        """Create a MsrConfig instance based on the default circuit.
        If keywords arguments are specified, then initialize a simulation
        by calling ``multiscale_run.cli.init(**kwargs)``.

        Returns:
            MsrConfig: configuration using the default circuit
        """
        if kwargs:
            sim_path = None
            if utils.rank0():
                from . import cli

                if "circuit" not in kwargs:
                    kwargs["circuit"] = DEFAULT_CIRCUIT
                sim_path = cli.init(**kwargs)
            sim_path = utils.comm().bcast(sim_path, root=0)
            return cls(sim_path)
        else:
            return NAMED_CIRCUITS[DEFAULT_CIRCUIT].config()
