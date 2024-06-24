Contributing
============

How to check a contribution?
****************************

Before submitting a contribution, it is suggested to use `tox` utility to make preliminary checks:

1. ``tox`` to run unit and integration tests.
2. ``tox -e lint`` to perform static analysis of the code and ensure it is properly formatted.
3. ``tox -e docs`` to ensure the documentation builds properly.
4. ``tox -e fixlint`` to format the code and applies ruff recommended fixes.


.. _tox-installation:
.. note:: If tox utility is not installed already, use a Python virtual environment for isolation purpose:

  .. code-block:: console

    python -mvenv venv
    . venv/bin/activate
    pip install -m tox
    tox


How to release a new version?
*****************************

MultiscaleRun relies on ``setuptools-scm`` utility to infer the Python package version from the SCM. It is not needed to increase the version manually. Anyway, there are a few things to perform before creating the git tag.

1. If the structure of the JSON configuration changed during this release (key addition, removal, ...), then increment the JSON ``config_format`` key in the files:

  * ``multiscale_run/data/config/rat_sscxS1HL_V6/simulation_config.json``
  * ``multiscale_run/data/config/rat_sscxS1HL_V10/simulation_config.json``

2. Ensure the Sphinx documentation is up-to-date. The fastest way is to check the artifacts of the ``docs`` stage in the CI.
3. Ensure the *Releases Notes* section is completed for this version.
4. Ensure the list of authors is up-to-date.
5. Ensure no spurious files were added to the repository by mistake (log files, process core dumps, ...)
6. Ensure the source distribution can be built: ``python -m build --sdist .``
7. Ensure the CI/CD pipeline passes on the ``main`` branch. Start one manually if necessary.
8. Create a git tag named after the new version, for instance: ``git tag 0.7``
9. Push the tag: ``git push origin --tags``. This operation triggers a CI/CD pipeline that builds and tests the package, and upload the new documentation to the `BBP Software Catalog`_. The documentation will appear the next day.
10. Create a new pull-request to the `BlueBrain/spack`_ GitHub repository mentioning the new version in the py-multiscale-run Spack package.
11. Ensure that the bbp workflow still works with the new version.

.. _BlueBrain/spack: https://github.com/BlueBrain/spack
.. _BBP Software Catalog: https://bbpteam.epfl.ch/documentation

How to rebuild the shared Julia environment on BB5?
***************************************************

A Julia environment providing all the packages required to execute the Metabolism model is available on BB5
at the following location ``/gpfs/bbp.cscs.ch/project/proj12/jenkins/subcellular/multiscale_run/julia-environment``.
By default, the command `multiscale-run init` uses this directory rather than creating a new Julia environment (which takes approximately 10min).

When this shared Julia environment becomes out of date (newer Julia version or newer packages), then it is required to recreate it.

**Prerequisite:** access to BBP project ``proj12``

1. go to the shared folder:

    ``cd /gpfs/bbp.cscs.ch/project/proj12/jenkins/subcellular/multiscale_run/julia-environment``

2. load the MultiscaleRun module:

    ``module load unstable py-multiscale-run``

3. Setup a new simulation with a fresh Julia environment. Usually we name the Julia environment based on the day. For example:

    ``multiscale-run init --julia=create 2024-04-22``

where `2024-04-22` is the name of the folder.

4. Remove everything in the folder that is not `.julia` or `.julia_environment`.
5. Create 2 symbolic links in the folder:

  .. code-block:: console

    cd 2024-04-22
    ln -s julia .julia
    ln -s julia_environment .julia_environment

6. Finally, link `latest` to this new folder (in `/gpfs/bbp.cscs.ch/project/proj12/jenkins/subcellular/multiscale_run/julia-environment`):

  .. code-block:: console

    cd ..
    ln -s 2024-04-22 latest

How to build the Sphinx documentation locally?
**********************************************

1. Ensure the ``tox`` utility is available (see :ref:`note above <tox-installation>` for installation)
2. Build the HTML documentation : ``tox -e docs``
3. Open the generated documentation created in: ``./docs/build/html/index.html``

.. note:: Troubleshooting if the build fails

  By default, the creation of the documentation is canceled if at least one error occurs.
  In case of unsuccessful build, either fix the issues reported by Sphinx to the console or update ``tox.ini`` to ignore
  these errors.

  .. code-block:: diff

    diff --git a/tox.ini b/tox.ini
    index 0796eba..4774331 100644
    --- a/tox.ini
    +++ b/tox.ini
    @@ -12,7 +13,7 @@ deps =
         sphinxcontrib-programoutput
         sphinx-mdinclude
         mistune<3 # there is a conflict with nbconvert
    -commands = sphinx-build -W --keep-going docs docs/build/html
    +commands = sphinx-build docs docs/build/html

  Anyway, the continuous-integration process requires the build of the documentation to pass without error.
