[![Build Status](https://app.travis-ci.com/grlewis333/WRAP.svg?branch=main)](https://app.travis-ci.com/grlewis333/WRAP)

WRAP
====
WRAP is an improved reconstruction algorithm for magnetic electron tomography data.

To install and use WRAP, we recommned using conda to manage packages. Two options include:
- Create your conda environment using the provided .yml file by downloading the file and typing `conda env create -f WRAP_env.yml`
- Add the necessary modules to your own environment by typing `conda install -c conda-forge -c astra-toolbox -c numba -c anaconda astra-toolbox=1.8.3 jupyterlab jupyter_client=7.1.0 jupyter_server=1.13.1 numpy scipy matplotlib scikit-image numba cython pywavelets jupyterlab_widgets ipympl ipywidgets`, followed by `pip install libertem --ignore-installed --user`

Basic use of WRAP is demonstrated in the WRAP_demo.ipynb python notebook.

If using WRAP for published work, please cite the following 2 papers:

Lewis, G., Ringe, E., & Midgley, P. (2022). Imaging Nanomagnetism in 3D: Potential Improvements for Vector Electron Tomography Reconstruction. Microscopy and Microanalysis, 28(S1), 2572-2574. doi:10.1017/S1431927622009801

Lewis, G., Ringe, E., & Midgley, P. (2022). WRAP: A Compressed Sensing Reconstruction Algorithm for Magnetic Vector Electron Tomography. Ultramicroscopy, [In preparation]
