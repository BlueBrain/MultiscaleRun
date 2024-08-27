How to use MultiscaleRun on BB5?
================================

MultiscaleRun is already installed on BB5 but first of all, allocate a compute node
to save the load on the login nodes, for instance:

``salloc -N 1 -A proj40 -p prod --exclusive --mem=0 -t 02:00:00 --cpus-per-task=2 --constraint=clx``

As a module (recommended)
*************************

``module load unstable py-multiscale-run``

As a spack package
******************

.. code-block:: console

  spack install py-multiscale-run@develop
  spack load py-multiscale-run

Using spack environments is recommended to work in an isolated environment with only the MultiscaleRun required spack packages.
More info about `spack environments <https://github.com/BlueBrain/spack/blob/develop/bluebrain/documentation/installing_with_environments.md>`_.

.. note:: **This may also work on your spack-powered machine!**
