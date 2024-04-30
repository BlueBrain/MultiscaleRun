Simulation Configuration
========================

This document describes the various configurations available in `simulation_config.json` in the section `multiscale_run`. The documentation about `sonata <https://sonata-extension.readthedocs.io/en/latest/sonata_simulation.html>`_ is here.

Paths
=====

A configuration key with *_path* suffix is considered as a filesystem path.
Paths are resolved and you can use keys that you defined elsewhere in the file. For example:

.. code-block::

  {
      "custom_path": "foo0",
      "new_path": "${custom_path}/foo1"
  }

`new_path` is resolved into: `foo0/foo1`.

You can also use `${pkg_data_path}` to point to a location in the MultiscaleRun installation folder.

Base section
==============

This is the base section that configures the parameters to run MultiscaleRun simulations.

- **with_steps**: Boolean. Indicates whether the STEPS simulator is included. It is in charge of the extracellular reactions/diffusions of the molecules.
- **with_bloodflow**: Boolean. Indicates whether the AstroVascPy simulator is included. It is in change of computing the blood flows.
- **with_metabolism**: Boolean. Indicates whether the Metabolism simulator is included. It is in charge of computing the Metabolism of the neurons (ATP/ADP etc. generation/consumption).
- **cache_save**: Boolean. If true, some simulation matrices are cached on the filesystem. It greatly speeds up initialization in future simulations.
- **cache_load**: Boolean. If true, some simulation matrices are loaded from the cache, if present. Used in conjunction with `cache_save`.
- **logging_level**: Integer. Defines the verbosity level of logs (fed into Neurodamus).
- **mesh_path**: String. Path to the mesh file used in simulations, e.g., "mesh/autogen_mesh.msh".
- **cache_path**: String. Filesystem location where the cached files are stored, e.g., "cache".
- **mesh_scale**: Float. Scale factor applied to the mesh dimensions, typically in the order of 1e-6.
- **config_format**: Integer. Specifies the configuration format version, e.g., 2.
- **msr_version**: String. Indicates the version of the MultiscaleRun that created this file, e.g., "1.0".
- **ndts**: Optional, Integer. Main MultiscaleRun iteration time step. Measured in Neurodamus time steps. If no entry is provided the one is calculated for you based on the time steps of the various simulators. It is suggested to let the program decide.

Preprocessor
==============

- **mesh**: This collects all the parameters required to let MultiscaleRun generates the mesh (necessary for STEPS and some connections among simulators). A custom mesh can be provided in the `mesh` folder. In that case the custom mesh is used instead.

  - **explode_factor**: Float. With 1 the auto-generating-mesh routine creates cube that touches the convex hull of all the segment extremities of the vasculature and the neurons. We need something slightly bigger to be sure of encompassing everything in the mesh. Usually kept at 1.001.
  - **base_length**: Float. Base length of mesh elements.
  - **refinement_steps**: Integer. Number of refinement steps applied to the mesh. Currently Omega_h fails if the number of tetrahedra is less than the number of ranks.

- **node_sets**

  - **filter_neuron**: Boolean. Determines if the neurons connected to astrocytes must be filtered out when generating the mesh.
  - **neuron_population_name**: String. Name of the neuron population to use. Typically: "All".

Connections among simulators
============================

The section **connect_to** outlines how data are exchanged between the different models.

Models execution
----------------

Simulations are performed alternating among simulators advancing each one based on their respective time step.
After a call to their respective `advance` method, each simulator is `synced` with the others based on the configuration in this section.
A `sync` means that some values from another simulator affect some values of the current simulator (the one that just performed an `advance`).
The keys in `connect_to` are the destination simulators: the simulators that just performed an `advance` and need to be synced with the others.
Each value is a list of `connections` that may affect some values of the `destination simulator`. The order of the list is important: later connections in these lists may override previous ones.

Configuration specification
---------------------------

Each connection must specify:

- **src_simulator**: String. Simulator, source of the values that need to be synced. Possible values are "neurodamus", "steps", "bloodflow", or "metabolism".
- **src_get_func**: String. Getter function for the source simulator.
- **src_get_kwargs**: Dict. Inputs for the getter function for the source simulator.
- **dest_get_func**: String. Setter function to the destination simulator.
- **dest_get_kwargs**: Dict. Inputs for the setter function for the destination simulator.
- **action**: String. Sync action type. It may be:

  - **set**: values from the source simulator are set into the destination simulator overriding what was there before.
  - **sum**: values from the source simulator are added to the destination simulator values.
  - **merge**: deltas (compared to the previous time step value of the destination simulator) for source and destination
    simulators are computed and added together to merge the results.
    This is typically used for syncing Neurodamus and Metabolism: Neurodamus consumes ATP while Metabolism generates ATP.
    This merging mechanism reconciles the 2 simulators every time Metabolism advances. Notice that this is the only
    syncing action that may change the values for the source simulator too. In that case also the following keys are required:

    - **src_set_func**: String. Setter function for the source simulator.
    - **src_set_kwargs**: Dict. Inputs for the Setter function for the source simulator.
    - **dest_get_func**: String. Getter function to the destination simulator.
    - **dest_get_kwargs**: Dict. Inputs for the getter function for the destination simulator.

Concrete example
----------------

.. code-block:: json

    {
        "connect_to": {
            "metabolism": [
                {
                    "src_simulator": "neurodamus",
                    "src_get_func": "get_var",
                    "src_get_kwargs": {"var": "atpi", "weight": "volume"},
                    "src_set_func": "set_var",
                    "src_set_kwargs": {"var": "atpi"},
                    "dest_get_func": "get_vm_idx",
                    "dest_get_kwargs": {"idx": 22},
                    "dest_set_func": "set_vm_idxs",
                    "dest_set_kwargs": {"idxs": [22]},
                    "action": "merge"
                }
            ]
        }
    }

In the previous block MultiscaleRun is instructed to `merge` (the action) the values from Neurodamus and Metabolism simulators (just after Metabolism calls `advance`). It follows the equation:

.. math::

    a_{n_{\text{metabolism}}+1} = a_{\text{metabolism} \; n_{\text{metabolism}}+1} + a_{\text{neurodamus} \; n_{\text{metabolism}}+1} - a_{n_{\text{metabolism}}}

All these values are based on the time step of Metabolism. :math:`n_{\text{metabolism}}` is the n\ :sup:`th` time step for Metabolism. The reconciled value at :math:`n_{\text{metabolism}}+1` is equal to the value from Metabolism plus the value from Neurodamus minus the previous reconciled value.

The remaining keys indicate functions and arguments for setters and getters for both source and destination. For example, to set the values to the destination we use the function `set_vm_idxs` and its arguments are: `"idxs": [22]`. It may be possible, like in this case, to set the value for multiple indexes simultaneously if the appropriate function accepts lists. This functionality may be expanded in the future to other setters and simulators if needed.

Data transformation
-------------------

For a few couplings, it is possible to specify data transformation operations when sending values from one simulator to another with the **conversion** JSON object.

- Bloodflow → Metabolism : matrices multiplication and Python expression
- Neurodamus → STEPS: Python expression only
- STEPS → Metabolism: Python expression only

Matrices Multiplication
^^^^^^^^^^^^^^^^^^^^^^^

The **matrices** key is an array of strings. Possible values are the names of the connection matrices defined in ``MsrConnectionManager``:

- **tetXtetMat**: tetrahedra x tetrahedra matrix to get the measure ratio compared to the mean tetrahedron measure. Useful to compute species dispersion in tetrahedra.
- **tetXbfVolsMat**: matrix representing Bloodflow segments in tetrahedra. To be multiplied only by Bloodflow volumes.
- **tetXbfFlowsMat**: matrix representing Bloodflow segments in tetrahedra. To be multiplied only by Bloodflow flows.

The operation multiplies the transferred data against the specified matrices.

Python expression
^^^^^^^^^^^^^^^^^

A Python expression whose result overrides the data transferred can be specified in the **op** configuration key.
The Python expression is executed in a restricted environment where only few symbols are usable:

- `vals`: the data being transferred
- `config`: the JSON configuration object
- `math`: the module from the standard library
- `np`: the NumPy module
- the computational Python builtins: `abs`, `min`, `max`, `pow`, `round`, and `sum`

Example of valid expressions:

- ``vals * (1.0 / (1.0e-3 * config.multiscale_run.steps.conc_factor))``
- ``abs(vals) * 5e-10``
- ``np.floor(10 * rg.random((3, 4)))``

Full example of JSON connections with transformation:

.. code-block:: json

  {
    "connect_to": {
      "metabolism": [
        {
          "src_simulator": "bloodflow",
          "src_get_func": "get_vols",
          "src_get_kwargs": {},
          "conversion": {
            "matrices": ["tetXbfVolsMat", "tetXtetMat"],
            "op": "vals * 5e-10"
          },
          "dest_set_func": "set_parameters_idxs",
          "dest_set_kwargs": {"idxs": [5]},
          "action": "set"
        }
      ],
      "steps": [
        {
          "src_simulator": "neurodamus",
          "src_get_func": "get_var",
          "src_get_kwargs": {"var": "ik","weight": "area"},
          "conversion": {"op": "vals * 1e-8"},
          "dest_set_func": "add_curr_to_conc",
          "dest_set_kwargs": {"species_name": "KK"},
          "action": "sum"
        }
      ]
    }
  }


Metabolism
==========

Configures the Metabolism simulator. The Julia model has 2 inputs: `parameters` and `vm`. The initial values of `vm` is `u0`.

- **ndts**: Integer. Time step of the simulator. Measured in number of Neurodamus time steps.
- **u0_path**: String. Path to the csv file that holds the initial values of the Metabolism model.
- **julia_code_path**: String. Path to the main Julia model file.
- **model**: Dict. Holds the additional variables for the Julia model.
    - **model_path**: String. Base path for the additional includes.
    - **pardir_path**: String. Base path for the additional parameters required by the Metabolism model.
    - **includes**: Dict. Additional includes required for the main Julia model to function.
    - **constants**: Dict. Additional constants required by the julia model.
- **constants**: Dict. Constant necessary for the Metabolism manager of multiscale run.
- **parameters**: List. List of parameters of the Metabolism model. They are the inputs (except `vm`) in order of the main Julia model file. During initialization (before any advance for any simulator), the connections to `metabolism` may replace these values. In that case, and only in this case, the `merge` action is downgraded to a `set` action.
- **solver_kwargs**: Dict. Parameters for the solver of the Metabolism model. The solver is currently: `de.Rosenbrock23`.
- **checks**: Dict. This a list of checks that are performed on the Metabolism inputs (parameters and vm) for every Metabolism time steps to verify integrity of the inputs. Items are optional. The parameters and vms that are not mentioned in this list are still checked to be normal numbers (no inf, nan is allowed). For example:

.. code-block:: json

    {
        "checks": {
                "parameters": {
                    "3": {
                        "name": "bloodflow_Fin",
                        "kwargs": {"leb": 0.0},
                        "response": "exclude_neuron"
                    }
                }
            }
    }

- **3**: Integer. Index of the checked parameter.
- **name**: String. Name of the parameter. Effectively unused in the simulation. Useful for the operator.
- **kwargs**: Dict. Arguments of the checking routine. Its entries are optional. The following entries are supported:
    - **lb**: Float. Lower bound. The value `v` must be:  \(lb < v \)
    - **leb**: Float. Lower or equal bound. The value `v` must be:  \(lb \leq  v \)
    - **hb**: Float. Higer bound. The value `v` must be:  \(v < hb \)
    - **heb**: Float. Higer or equal bound. The value `v` must be:  \(v \leq  heb \)
- **response**: String. If one of the values does not pass a check for a neuron we apply the response. Currently, the following responses are implemented:
    - **exclude_neuron**: The neuron is removed from the simulation. The rest may continue. If no neurons remain (among all ranks) the simulation is aborted at the end of a MultiscaleRun iteration.
    - **abort_simulation**: The simulation is aborted.

STEPS
=====

Parameters for the STEPS simulator.

- **ndts**: Integer. Time step of the simulator. Measured in number of Neurodamus time steps.
- **conc_factor**: Float. Rescaling factor for the number of molecules. Necessary because the mesh is very coarse and STEPS may overflow.
- **compname**: String. Name of the `compartment <https://steps.sourceforge.net/manual/API_2/API_geom.html?highlight=compartment#steps.API_2.geom.Compartment>`_.
- **Volsys**: Dict. `System volume <https://steps.sourceforge.net/manual/API_2/API_model.html?highlight=volumesystem#steps.API_2.model.VolumeSystem>`_ parameters.
    - **name**: String. Name of the system volume. It needs to be the same that was used to create appropriate physical entity in the mesh.
    - **species**: Dict. Parameters of the reaction-diffusion species.
        - **conc_0**: Float. Initial concentration in `mM`.
        - **diffcst**: Float. `Diffusion <https://steps.sourceforge.net/manual/API_2/API_model.html?highlight=diffusion#steps.API_2.model.Diffusion>`_ constant in SI units.
        - **ncharges**: Integer. Charge number of the ion.

Blood Flow
==========

Parameters for the blood flow simulator (AstroVascPy).

`astrovascpy parameters <https://astrovascpy.readthedocs.io/latest/generated/astrovascpy.typing.html#astrovascpy.typing.VasculatureParams>`_.

Reports
=======

Parameters to report the simulation outcome. Currently, MultiscaleRun reports in the same folder as Neurodamus. The location is stated in `output.output_dir`. Here we try to mimic how Neurodamus reports so that the postprocessing can digest both MultiscaleRun and Neurodamus files. Example:

.. code-block:: json

    {
        "reports": {
            "metabolism": {
                "metab_ina": {
                    "src_get_func": "get_parameters_idx",
                    "src_get_kwargs": {"idx": 0},
                    "unit": "mA/cm^2",
                    "file_name": "metab_ina.h5",
                    "when": "after_sync"
                }
            }
    }

- **src_get_func**: String. Getter function for the simulator (in this case, `metabolism`).
- **src_get_kwargs**: Dict. Inputs for the getter function.
- **unit**: String. Units of the values in the report.
- **file_name**: String. Name of the file.
- **when**: String. Since multiple simulators are active at the same time and `sync` calls may modify the values of the simulators the report may take the values just before or just after the `sync` operation. This value selects that. Possible values: `after_sync`, `before_sync`. Multiple reports (with different file names) for reporting just before and after `sync` are possible.

