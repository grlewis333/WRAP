Setup
=====


Requirements
------------

WRAP requires a CUDA-enabled NVIDIA GPU to speed up projection calculations.

Otherwise, WRAP should run fine on all operating systems and coding environments.

It is optimised to be set-up using conda and run inside Jupyter Notebooks.


Installation
------------

* Option 1: Create your conda environment using the provided WRAP_env.yml file by downloading the file and typing :code:`conda env create -f WRAP_env.yml`. Note this will only work for Windows users.

* Option 2: Create a fresh conda environment and install everything all at once to avoid compatibility issues with :code:`conda install -c conda-forge -c astra-toolbox -c numba -c anaconda astra-toolbox=1.8.3 jupyterlab numpy scipy matplotlib scikit-image numba cython pywavelets jupyterlab_widgets ipympl ipywidgets`, followed by :code:`pip install libertem --ignore-installed --user`. (Note in some cases, to properly configure jupyter it is necessary to additionally run :code:`conda install -c conda-forge jupyter_client=7.1.0 jupyter_server=1.13.1`).
