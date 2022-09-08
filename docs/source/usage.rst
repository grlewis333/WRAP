Usage
=====


Demo
----

To see a demonstration of WRAP, please look through the demo notebook on Github.

Formatting the input data
-------------------------
For WRAP we require three input parameters:

* phis: These are the extracted magnetic phase images as a stack. They should be formatted such that the middle column is the image index, as is the custom in the astra framework.

* angles: This should be a list of (ax, ay, az) angles in degrees, with each trio of angles corresponding to the tilt angle of the images in 'phis'. These angles correspond directly to (alpha, beta, gamma) tilts in the microscope.

* mesh_params: This tells WRAP the dimensions of your data. It needs to be in the format [(bbx1,bby1,bbz1),(bbx2,bby2,bbz2), (nx,ny,nz)], where bb1 defines the coordinates in metres of one corner of your bounding box, and bb2 defines the opposite corner; n then defines the number of voxels in each direction of the mesh.

Running the reconstruction
-------------------------
To run a WRAP reconstruction, use the format:

>>> bxw,byw,bzw = WRAP.reconstructions.WRAP(phis,angles,mesh_params,n_pad=44,nmaj=3,nmin=10,weight=3e-3,th1=.2,th2=.8)

* n_pad: Amount to pad image in each direction (wavelet regularisation works best when image size is a square number, and image should be padded by at least the same size to avoid fourier wrap-around artefacts.)
* nmaj: Number of major iterations
* nmin: Number of minor iterations
* weight: Wavelet regularisation weighting
* th1: WRAP lower threshold (applied in final nmaj)
* th2: WRAP upper threshold (applied for nmaj 1)

To run a conventional reconstruction, use the format:

>>> bxc,byc,bzc = WRAP.reconstructions.conventional(px,py,angx,angy,mesh_params,niter=5)

* px,py: Stack of phase images (tilt number as middle column) for x,y tilt series
* angx,angy: List of (ax,ay,az) rotations in degrees for each image for x,y tilt series
* niter: Number of SIRT iterations