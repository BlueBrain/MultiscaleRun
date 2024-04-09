Contributing
============

How to check a contribution?
****************************

Before submitting a contribution, it is suggested to use `tox` utility to make preliminary checks:

1. ``tox`` to run unit and integration tests.
2. ``tox -e lint`` to perform static analysis of the code and ensure it is properly formatted.
3. ``tox -e docs`` to ensure the documentation builds properly.
4. ``tox -e fixlint`` to format the code and applies ruff recommended fixes.


.. note:: If tox utility is not installed already, use a Python virtual environment for isolation purpose:

  .. code-block:: console

    python -mvenv venv
    . venv/bin/activate
    pip install -m tox
    tox

