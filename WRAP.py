"""
Code for wavelet regularised tomographic reconsuction of magnetic fields from 
magnetic phase shifts tilt series.

Created by George R. Lewis - 07 Sep 2022
"""

# Imports
import matplotlib.pyplot as plt                     # For normal plotting
import numpy as np                                  # For maths
from scipy import ndimage                           # For image rotations
from scipy import optimize                          # For function minimization
from scipy.ndimage import zoom                      # For image rescaling
from scipy import constants                         # For scientific constants
import copy                                         # For deepcopy
import astra                                        # For tomography framework
from libertem.utils.generate import hologram_frame  # For hologram simulation
import ipywidgets                                   # Fpr interactive plotting
from skimage.restoration import unwrap_phase        # For hologram phase unwrapping
import ToveyTomoTools_adapted as rtr                # For compressed sensing reconstructions

class phantoms():
    """ Class for creating magnetisation phantoms """
    def sphere(rad_m = 10*1e-9, Ms_Am = 797700, plan_rot=0, bbox_length_m = 100*1e-9, bbox_length_px = 100):
        """ Creates uniformly magnetised sphere
        
                Returns the x,y,z components of magnetisation as 3D arrays
                Also returns the 'mesh parameters' which are consistent with Ubermag,
                defined as [(bbox coord 1), (bbox coord 2), (voxel res)]
                where bbox coords are the x,y,z coordinates in metres of a bounding box,
                and voxel res defines the number of voxels in x,y,z directions.
        
            rad_m : Radius in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        # Initialise bounding box parameters
        p1 = (0,0,0)
        p2 = (bbox_length_m,bbox_length_m,bbox_length_m)
        n = (bbox_length_px,bbox_length_px,bbox_length_px)
        mesh_params = [p1,p2,n]
        res = bbox_length_m / bbox_length_px # resolution in m per px 
        ci = int(bbox_length_px/2) # index of bbox centre
        
        # Initialise magnetisation arrays
        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if (i-ci)**2 + (j-ci)**2 + (k-ci)**2 < (rad_m/res)**2:
                        MX[i,j,k] = np.cos(plan_rot*np.pi/180)*Ms_Am
                        MY[i,j,k] = np.sin(plan_rot*np.pi/180)*Ms_Am
        
        
        MX=MX.astype(np.float32)
        MY=MY.astype(np.float32)
        MZ=MZ.astype(np.float32)

        return MX,MY,MZ, mesh_params
    
class dataset_generator():
    def realistic_sphere(tilt=10,a_range=70):
        """" Creates a dual-axis dataset for a magnetic sphere
        
            Input:
                tilt: defines the number of images in each tilt axis
                a_range: defines the angle range in degrees for each tilt axis 
            
            Output:
                phis: Stack of magnetic phase images with noise (in astra format - middle column is image number)
                angles: Corresponding tilt angles for each image
                BX,BY,BZ: 3D array for each component of (ideal) B field
                mesh_params: Mesh parameters [bbox c1, bbox c2, resolution]
                MX,MY,MZ: 3D array for each component of (ideal) magnetisation
                AX,AY,AZ: 3D array for each component of (ideal) magnetic vector potential A
                
            Note that sphere data is in 40x40x40 array, but is padded by 44 in each direction
            to give a final size of 128x128x128 since wavelet regularisation works best
            when image size is a square number, and image should be padded by at least the same
            size to avoid fourier wrap-around artefacts.
                
            """
        # Generate sphere
        MX,MY,MZ,_ = phantoms.sphere(bbox_length_px=100,rad_m=15e-9)
        # Crop out the blank space
        MX,MY,MZ = MX[20:-20,20:-20,20:-20],MY[20:-20,20:-20,20:-20],MZ[20:-20,20:-20,20:-20]
        # Reduce resolution to make the simulation smaller
        x=.66
        MX=zoom(MX, (x,x,x))
        MY=zoom(MY, (x,x,x))
        MZ=zoom(MZ, (x,x,x))
        
        # Rotate magnetisation so that there is non-zero magnetisation in each component
        MX,MY,MZ = rotate_magnetisation(MX,MY,MZ,30,50,20)

        # Define mesh parameters
        p1 = (0,0,0)
        p2=(100*1e-9,100*1e-9,100*1e-9)
        n = 40
        n_pad=44
        mesh_params = (p1,p2,(n,n,n))

        # Calculate A and B
        AX,AY,AZ,mesh_params2 = calculate_A_3D(MX,MY,MZ,mesh_params=mesh_params,n_pad=n_pad)
        BX,BY,BZ = calculate_B_from_A(AX,AY,AZ,mesh_params=mesh_params2)
        
        # Generate projection data
        as_x,as_y,pxs,pys = dual_axis_phase_generation(MX,MY,MZ,mesh_params,n_tilt= tilt,a_range=a_range,n_pad=n_pad)
        angles_x = generate_angles(mode='x',n_tilt=tilt,alpha=a_range)
        angles_y = generate_angles(mode='y',n_tilt=tilt,beta=a_range,tilt2='beta')

        pxs_n = noisy_phase(pxs,holo=True,MX=MX,MY=MY,MZ=MZ,angles=angles_x,mesh_params=mesh_params,n_pad=44,fxc=576,fyc=529,fringe=10,rc=30,n=0.02,v=.7,c=2000)
        pys_n = noisy_phase(pys,holo=True,MX=MX,MY=MY,MZ=MZ,angles=angles_y,mesh_params=mesh_params,n_pad=44,fxc=576,fyc=529,fringe=10,rc=30,n=0.02,v=.7,c=2000)



        # Dual axis data
        bx_p,by_p = dual_axis_B_generation(pxs,pys,mesh_params)

        # Multi axis data
        angles = np.concatenate([as_x,as_y])
        phis = np.concatenate([pxs_n,pys_n],axis=1)

        return phis, angles, BX,BY,BZ, mesh_params,MX,MY,MZ,AX,AY,AZ
    
def rotate_bulk(P,ax,ay,az,mode='ndimage'):
    """ 
    Rotate magnetisation locations from rotation angles ax,ay,az 
    about the x,y,z axes (given in degrees) 
    
    Can use PIL or ndimage. ndimage def works but PIL faster (PIL implementation not finished yet)
    
    NOTE: This implementation of scipy rotations is EXTRINSIC
    Therefore, to make it compatible with our intrinsic vector
    rotation, we swap the order of rotations (i.e. x then y then z)
    """
    # Due to indexing, ay needs reversing for desired behaviour
    if mode == 'PIL':
        nx,ny,nz = np.shape(P)
        Prot = np.zeros_like(P)
        ax,ay,az=ax,-ay,az
        scale = 256/np.max(P)
        for i in range(nx):
            im = Image.fromarray(P[i,:,:]*scale).convert('L')
            im = im.rotate(ax,resample = Image.BILINEAR)
            Prot[i,:,:] = np.array(im)/scale
        for j in range(ny):
            im = Image.fromarray(Prot[:,j,:]*scale).convert('L')
            im = im.rotate(ay,resample = Image.BILINEAR)
            Prot[:,j,:] = np.array(im)/scale
        for k in range(nz):
            im = Image.fromarray(Prot[:,:,k]*scale).convert('L')
            im = im.rotate(az,resample = Image.BILINEAR)
            Prot[:,:,k] = np.array(im)/scale
            
        return Prot
    
    else:
        ay = -ay

        P = ndimage.rotate(P,ax,reshape=False,axes=(1,2),order=1)
        P = ndimage.rotate(P,ay,reshape=False,axes=(2,0),order=1)
        P = ndimage.rotate(P,az,reshape=False,axes=(0,1),order=1)

        return P
    
def rotate_magnetisation(U,V,W,ax=0,ay=0,az=0):
    """ 
    Takes 3D gridded magnetisation values as input
    and returns them after an intrinsic rotation ax,ay,az 
    about the x,y,z axes (given in degrees) 
    (Uses convention of rotating about z, then y, then x)
    """
    # Rotate the gridded locations of M values
    Ub = rotate_bulk(U,ax,ay,az)
    Vb = rotate_bulk(V,ax,ay,az)
    Wb = rotate_bulk(W,ax,ay,az)
    
    shape = np.shape(Ub)
    
    # Convert gridded values to vectors
    coor_flat = grid_to_coor(Ub,Vb,Wb)
    
    # Rotate vectors
    coor_flat_r = rotate_vector(coor_flat,ax,ay,az)
    
    # Convert vectors back to gridded values
    Ur,Vr,Wr = coor_to_grid(coor_flat_r,shape=shape)

    return Ur,Vr,Wr

def rotation_matrix(ax,ay,az,intrinsic=True):
    """ 
    Generate 3D rotation matrix from rotation angles ax,ay,az 
    about the x,y,z axes (given in degrees) 
    (Uses convention of rotating about z, then y, then x)
    """

    ax = ax * np.pi/180
    Cx = np.cos(ax)
    Sx = np.sin(ax)
    mrotx = np.array([[1,0,0],[0,Cx,-Sx],[0,Sx,Cx]])
    
    ay = ay * np.pi/180
    Cy = np.cos(ay)
    Sy = np.sin(ay)
    mroty = np.array([[Cy,0,Sy],[0,1,0],[-Sy,0,Cy]])
    
    az = az * np.pi/180
    Cz = np.cos(az)
    Sz = np.sin(az)
    mrotz = np.array([[Cz,-Sz,0],[Sz,Cz,0],[0,0,1]])
    
    if intrinsic == True:
        mrot = mrotz.dot(mroty).dot(mrotx)
    else:
        # To define mrot in an extrinsic space, matching
        # our desire for intrinsic rotation, we need
        # to swap the order of the applied rotations
        mrot = mrotx.dot(mroty).dot(mrotz)
    
    return mrot

def grid_to_coor(U,V,W):
    """ Convert gridded 3D data (3,n,n,n) into coordinates (n^3, 3) """
    coor_flat = []
    nx = np.shape(U)[0]
    ny = np.shape(U)[1]
    nz = np.shape(U)[2]
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x = U[ix,iy,iz]
                y = V[ix,iy,iz]
                z = W[ix,iy,iz]
                coor_flat.append([x,y,z])
                
    return coor_flat

def coor_to_grid(coor_flat,shape=None):
    """ Convert coordinates (n^3, 3) into gridded 3D data (3,n,n,n) """
    if shape == None:
        n = int(np.round(np.shape(coor_flat)[0]**(1/3)))
        shape = (n,n,n)
    nx,ny,nz = shape
    
    x = np.take(coor_flat,0,axis=1)
    y = np.take(coor_flat,1,axis=1)
    z = np.take(coor_flat,2,axis=1)
    U = x.reshape((nx,ny,nz))
    V = y.reshape((nx,ny,nz))
    W = z.reshape((nx,ny,nz))

    return U, V, W

def rotate_vector(coor_flat,ax,ay,az):
    """ Rotates vectors by specified angles ax,ay,az 
    about the x,y,z axes (given in degrees) """
    
    # Get rotation matrix
    mrot = rotation_matrix(ax,ay,az)    

    coor_flat_r = np.zeros_like(coor_flat)
    
    # Apply rotation matrix to each M vector
    for i,M in enumerate(coor_flat):
        coor_flat_r[i] = mrot.dot(M)
    
    return coor_flat_r

def calculate_A_3D(MX,MY,MZ, mesh_params=None,n_pad=100,tik_filter=0.01):
    """ Input(3D (nx,ny,nz) array for each component of M) and return
    three 3D arrays of magnetic vector potential 
    
    Note, returned arrays will remain padded since if they are used for
    projection to phase change this will make a difference. So the new padded
    mesh parameters are also returned
    
    """
    if mesh_params == None:
        p1 = (0,0,0)
        sx,sy,sz = np.shape(MX)
        p2 = (sx,sy,sx)
        n = p2
    else:
        p1,p2,n = mesh_params
    
    # zero pad M to avoid FT convolution wrap-around artefacts
    mxpad = np.pad(MX,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    mypad = np.pad(MY,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    mzpad = np.pad(MZ,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)

    # take 3D FT of M    
    ft_mx = np.fft.fftn(mxpad)
    ft_my = np.fft.fftn(mypad)
    ft_mz = np.fft.fftn(mzpad)
    
    # Generate K values
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    resz = p2[2]/n[2] # resolution in m per px 

    kx = np.fft.fftfreq(ft_mx.shape[0],d=resx)
    ky = np.fft.fftfreq(ft_my.shape[0],d=resy)
    kz = np.fft.fftfreq(ft_mz.shape[0],d=resz)
    KX, KY, KZ = np.meshgrid(kx,ky,kz, indexing='ij') # Create a grid of coordinates
    
    # vacuum permeability
    mu0 = 4*np.pi*1e-7
    
    # Calculate 1/k^2 with Tikhanov filter
    if tik_filter == 0:
        K2_inv = np.nan_to_num(((KX**2+KY**2+KZ**2)**.5)**-2)
    else:
        K2_inv = ((KX**2+KY**2+KZ**2)**.5 + tik_filter*resx)**-2
    
    # M cross K
    cross_x = ft_my*KZ - ft_mz*KY
    cross_y = -ft_mx*KZ + ft_mz*KX
    cross_z = -ft_my*KX + ft_mx*KY
    
    # Calculate A(k)
    ft_Ax = (-1j * mu0 * K2_inv) * cross_x
    ft_Ay = (-1j * mu0 * K2_inv) * cross_y
    ft_Az = (-1j * mu0 * K2_inv) * cross_z
    
    # Inverse fourier transform
    Ax = np.fft.ifftn(ft_Ax)
    AX = Ax.real
    Ay = np.fft.ifftn(ft_Ay)
    AY = Ay.real
    Az = np.fft.ifftn(ft_Az)
    AZ = Az.real
    
    # new mesh parameters (with padding)
    n = (n[0]+2*n_pad,n[1]+2*n_pad,n[2]+2*n_pad)
    p2 = (p2[0]+2*n_pad*resx,p2[1]+2*n_pad*resy,p2[2]+2*n_pad*resz)
    mesh_params=(p1,p2,n)
    
    AX=AX.astype(np.float32)
    AY=AY.astype(np.float32)
    AZ=AZ.astype(np.float32)
    
    return AX,AY,AZ,mesh_params

def calculate_B_from_A(AX,AY,AZ,mesh_params=None):
    """ Takes curl of B to get A """
    # Initialise parameters
    phase_projs = []
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(AX)
        p2 = (s[0],s[1],s[2])
        n = p2
        mesh_params = [p1,p2,n]
    p1,p2,n=mesh_params
    
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    resz = p2[2]/n[2] # resolution in m per px 
    
    BX = np.gradient(AZ,resy)[1] - np.gradient(AY,resz)[2]
    BY = np.gradient(AX,resz)[2] - np.gradient(AZ,resx)[0]
    BZ = np.gradient(AY,resx)[0] - np.gradient(AX,resy)[1]
    
    BX=BX.astype(np.float32)
    BY=BY.astype(np.float32)
    BZ=BZ.astype(np.float32)
        
    return BX/(2*np.pi),BY/(2*np.pi),BZ/(2*np.pi)

def dual_axis_phase_generation(MX,MY,MZ,mesh_params,n_tilt=40, a_range=70,n_pad = 100):
    """ Returns ax,ay, px,py (angles and phase projections from x and y tilt series)
    n_tilt = number of projections in each series
    a_range = maximum tilt angle (applies to both series)
    n_pad = padding applied during phase calculation (should be > 2x n_px) """
    angles_x = generate_angles(mode='x',n_tilt=n_tilt,alpha=a_range)
    angles_y = generate_angles(mode='y',n_tilt=n_tilt,beta=a_range,tilt2='beta')
    phases_x = generate_phase_data(MX,MY,MZ,angles_x,mesh_params=mesh_params,n_pad=n_pad,unpad=False)
    phases_y = generate_phase_data(MX,MY,MZ,angles_y,mesh_params=mesh_params,n_pad=n_pad,unpad=False)
    
    return angles_x, angles_y, phases_x, phases_y

def generate_angles(mode='x',n_tilt = 40, alpha=70,beta=40,gamma=180,dist_n2=8,tilt2='gamma'):
    """ Return a list of [ax,ay,az] lists, each corresponding to axial
    rotations applied to [0,0,1] to get a new projection direction.
    
    Modes = x, y, dual, quad, sync, dist, rand
    
    Specify the +- tilt range of alpha/beta/gamma
    
    Say total number of tilts n_tilt
    
    For dist, each alpha has 'dist_n2' 'tilt2' projections
    
    Specify if the 2nd tilt axis is beta or gamma """
    
    angles = []
    ax,ay,az = 0,0,0
    
    # x series
    if mode=='x':
        for ax in np.linspace(-alpha,alpha,n_tilt):
            angles.append([ax,ay,az])
            
    if mode=='y':
        if tilt2 == 'beta':
            for ay in np.linspace(-beta,beta,n_tilt):
                angles.append([ax,ay,az])
        if tilt2 == 'gamma':
            if gamma >= 90:
                az = 90
            else:
                az = gamma
            for ax in np.linspace(-alpha,alpha,n_tilt):
                angles.append([ax,ay,az])
            
    if mode=='dual':
        for ax in np.linspace(-alpha,alpha,int(n_tilt/2)):
            angles.append([ax,ay,az])
            
        ax,ay,az = 0,0,0
        if tilt2 == 'beta':
            for ay in np.linspace(-beta,beta,int(n_tilt/2)):
                angles.append([ax,ay,az])
        if tilt2 == 'gamma':
            if gamma >=90:
                az = 90
            else:
                az = gamma
            for ax in np.linspace(-alpha,alpha,int(n_tilt/2)):
                angles.append([ax,ay,az])
    
    if mode=='quad':
        if tilt2 == 'beta':
            for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                angles.append([ax,ay,az])
            ax,ay,az = 0,0,0
            for ay in np.linspace(-beta,beta,int(n_tilt/4)):
                angles.append([ax,ay,az])
            ay = beta
            for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                angles.append([ax,ay,az])
            ay = -beta
            for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                angles.append([ax,ay,az])
                    
        if tilt2 == 'gamma':
            if gamma >= 90:
                for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                    angles.append([ax,ay,az])
                az = 90
                for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                    angles.append([ax,ay,az])
                az = 45
                for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                    angles.append([ax,ay,az])
                az = -45
                for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                        angles.append([ax,ay,az])           
            else:
                az = gamma
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])
                az = -gamma
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])
                az = gamma/3
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])
                az = -gamma/3
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])

    # random series # g or b
    if mode=='rand':
        for i in range(n_tilt):
            ax_rand = np.random.rand()*alpha*2 - alpha
            if tilt2 == 'beta':
                ay_rand = np.random.rand()*beta*2 - beta
                angles.append([ax_rand,ay_rand,0])
            if tilt2 == 'gamma':
                az_rand = np.random.rand()*gamma*2 - gamma
                angles.append([ax_rand,0,az_rand])
            
    # alpha propto beta series # g or b
    if mode=='sync':
        if tilt2 == 'beta': 
            ax = np.linspace(-alpha,alpha,int(n_tilt/2))
            ay = np.linspace(-beta,beta,int(n_tilt/2))

            for i,a in enumerate(ax):
                angles.append([a,ay[i],0])

            for i,a in enumerate(ax):
                angles.append([a,-ay[i],0])
        if tilt2 == 'gamma': 
            ax = np.linspace(-alpha,alpha,int(n_tilt/2))
            az = np.linspace(-gamma,gamma,int(n_tilt/2))

            for i,a in enumerate(ax):
                angles.append([a,0,az[i]])

            for i,a in enumerate(ax):
                angles.append([a,0,-az[i]])
            
    # even spacing # g or b
    if mode=='dist':
        ax = np.linspace(-alpha,alpha,int(n_tilt/dist_n2))
        if alpha == 90:
            ax = np.linspace(-90,90,n_tilt/dist_n2+1)
            ax = ax[::-1]
        if tilt2 == 'beta': 
            ay = np.linspace(-beta,beta,dist_n2)
            for x in ax:
                for y in ay:
                    angles.append([x,y,0])
        if tilt2 == 'gamma': 
            if gamma < 90:
                az = np.linspace(-gamma,gamma,dist_n2)
                for x in ax:
                    for z in az:
                        angles.append([x,0,z])
            if gamma >= 90:
                az = np.linspace(-90,90,dist_n2+1)
                for x in ax:
                    for z in az[:-1]:
                        angles.append([x,0,z])
    
    return angles

def generate_phase_data(MX,MY,MZ,angles,mesh_params=None,n_pad=500,unpad=False):
    """ Returns phase projections for given M and angles
    in order [x, i_tilt, y] """
    # Initialise parameters
    phase_projs = []
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(MX)
        p2 = (s[0],s[1],s[2])
        n = p2
        mesh_params = [p1,p2,n]
    
    # Loop through projection angles
    for i in range(len(angles)):
        ax,ay,az = angles[i]
        #rotate M
        MXr,MYr,MZr = rotate_magnetisation(MX,MY,MZ,ax,ay,az)
        #calculate phase
        phase = calculate_phase_M_2D(MXr,MYr,MZr,mesh_params=mesh_params,n_pad=n_pad,unpad=unpad)
        phase = np.flipud(phase.T)

        phase_projs.append(phase)            
    
    # Prepare projections for reconstruction
    phase_projs = np.transpose(phase_projs,axes=[1,0,2]) # reshape so proj is middle column
    phase_projs=phase_projs.astype(np.float32)
    return np.array(phase_projs)

def calculate_phase_M_2D(MX,MY,MZ,mesh_params,n_pad=500,tik_filter=0.01,unpad=True):
    """ Preffered method. Takes 3D MX,MY,MZ magnetisation arrays
    and calculates phase shift in rads in z direction.
    First projects M from 3D to 2D which speeds up calculations """
    p1,p2,n=mesh_params
    
    # J. Loudon et al, magnetic imaging, eq. 29
    const = .5 * 1j * 4*np.pi*1e-7 / constants.codata.value('mag. flux quantum')

    # Define resolution from mesh parameters
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    resz = p2[2]/n[2] # resolution in m per px 
    
    # Project magnetisation array
    mx = project_along_z(MX,mesh_params=mesh_params)
    my = project_along_z(MY,mesh_params=mesh_params)
    
    # Take fourier transform of M
    # Padding necessary to stop Fourier convolution wraparound (spectral leakage)
    if n_pad > 0:
        mx = np.pad(mx,[(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
        my = np.pad(my,[(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    
    ft_mx = np.fft.fft2(mx)
    ft_my = np.fft.fft2(my)
    
    # Generate K values
    kx = np.fft.fftfreq(n[0]+2*n_pad,d=resx)
    ky = np.fft.fftfreq(n[1]+2*n_pad,d=resy)
    KX, KY = np.meshgrid(kx,ky, indexing='ij') # Create a grid of coordinates
    
    # Filter to avoid division by 0
    if tik_filter == 0:
        K2_inv = np.nan_to_num(((KX**2+KY**2)**.5)**-2)
    else:
        K2_inv = ((KX**2+KY**2)**.5 + tik_filter*resx)**-2

    # Take cross product (we only need z component)
    cross_z = (-ft_my*KX + ft_mx*KY)*K2_inv
    
    # Inverse fourier transform
    phase = np.fft.ifft2(const*cross_z).real
    
    # Unpad
    if unpad == True:
        if n_pad > 0:
            phase=phase[n_pad:-n_pad,n_pad:-n_pad]
    
    return phase

def project_along_z(U,mesh_params=None):
    """ Takes a 3D array and projects along the z component 
    It does this by multiplying each layer by its thickness
    and then summing down the axis. """
    if type(mesh_params) == type(None):
        p1 = (0,0,0)
        sx,sy,sz = np.shape(U)
        p2 = (sx,sy,sz)
        n = p2
    else:
        p1,p2,n = mesh_params
    
    # Get resolution    
    z_size = p2[2]
    z_res = z_size/n[2]
    
    # project
    u_proj = np.sum(U*z_res,axis=2)
    
    return u_proj

def noisy_phase(ps, misalign = False, gaussian = False, lowpass=False,maxshift=3,noise_level=np.pi/30,freq_rad_px=20,holo=False,fringe=10,up=10,v=1,n=None,c=1000,n_pad=32,MX=None,MY=None,MZ=None,angles=None,mesh_params=None,fxc=400,fyc=400,rc=50):
    """ Makes phase images noisy!
    Misalign: Randomly shifts each image in stack by +- maxshift pixels in x and y directions.
    Gaussian: Adds Gaussian noise where noise_level corresponds to three standard deviations.
    Lowpass: Removes frequencies beyond a freq_rad_px radius in Fourier space.
    """
    ps_n = ps
    if misalign == True:
        # Randomly shift the tilt series
        ps_n = misalign_func(ps_n,maxshift=maxshift)

    if gaussian == True:
        # Add gaussian noise
        ps_n = noisy(ps_n,noise_typ='gauss',g_var=(noise_level/3)**(2))

    if lowpass == True:
        # Filter out high spatial frequencies
        ps_n = spatial_freq_filter(ps_n,rad=freq_rad_px)
        
    if holo==True:
        ps_n = hologram_noise(ps_n,MX,MY,MZ,mesh_params,angles,fxc=fxc,fyc=fyc,rc=rc,n=n,v=v,c=c,plot=False,fringe=fringe)
    
    return ps_n

def misalign_func(ps,maxshift=1):
    """ Takes a stack of images (indexed along middle column)
    and randomly translates them up/down and left/right by 
    up to a maximum of maxshift pixels """
    
    num = np.shape(ps)[1]
    
    ps_shift = []
    for ipic in range(num):
        im = ps[:,ipic,:]
        imshift = copy.deepcopy(im)

        xtrans = np.random.randint(-1,high=2)
        if xtrans!=0:
            if xtrans == 1:

                val = np.random.choice(list(range(1,maxshift+1,1)))
                imshift = imshift[val:]
                imshift = np.pad(imshift,((0,val),(0,0)),mode='edge')
                #print('right',val)
            if xtrans == -1:

                val = np.random.choice(list(range(1,maxshift+1,1)))
                imshift = imshift[:-val]
                imshift = np.pad(imshift,((val,0),(0,0)),mode='edge')
                #print('left',val)

        ytrans = np.random.randint(-1,high=2)       
        if ytrans!=0:
            if ytrans == 1:

                val = np.random.choice(list(range(1,maxshift+1,1)))
                imshift = imshift[:,val:]
                imshift = np.pad(imshift,((0,0),(0,val)),mode='edge')
                #print('up',val)
            if ytrans == -1:

                val = np.random.choice(list(range(1,maxshift+1,1)))
                imshift = imshift[:,:-val]
                imshift = np.pad(imshift,((0,0),(val,0)),mode='edge')
                #print('down',val)
                
        ps_shift.append(imshift)
    
    ps_shift = np.transpose(ps_shift,axes=[1,0,2]) # reshape so proj is middle column
    ps_shift=ps_shift.astype(np.float32)
    
    return ps_shift

def noisy(image, noise_typ='gauss',g_var = 0.1, p_sp = 0.004,val_pois = None,sp_var=1):
    """ Adapted from Stack Exchange 
    Add noise to image with choice from:
    - 'gauss' for Gaussian noise w/ variance 'g_var'
    - 's&p' for salt & pepper noise with probability 'p_sp'
    - 'poisson' for shot noise with avg count of 'val_pois'
    - 'speckle' for speckle noise w/ variance 'sp_var'"""
    if noise_typ == "gauss":
        # INDEPENDENT (ADDITIVE)
        # Draw random samples from a Gaussian distribution
        # Add these to the image
        # Higher variance = more noise
        row,col,ch= image.shape
        mean = 0
        var = g_var
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    
    elif noise_typ == "s&p":
        # INDEPENDENT
        # Salt & pepper/spike/dropout noise will either
        # set random pixels to their max (salt) or min (pepper)
        # Quantified by the % of corrupted pixels
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = p_sp
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape] # randomly select coordinates
        out[coords] = np.max(image) # set value to max

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape] # randomly select coordinates
        out[coords] = np.min(image) # set value to min
        return out
    
    elif noise_typ == "poisson":
        # DEPENDENT (MULTIPLICATIVE)
        # Poisson noise or shot noise arises due to the quantised
        # nature of particle detection.
        # Each pixel changes from its original value to 
        # a value taken from a Poisson distrubution with
        # the same mean (multiplied by vals)
        # So val can be thought of as the avg no. of electrons
        # contributing to that pixel of the image (low = noisy)
        if val_pois == None:
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
        else:
            vals = val_pois
            
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    
    elif noise_typ =="speckle":
        # DEPENDENT (MULTIPLICATIVE)
        # Random value multiplications of the image pixels
        
        # Generate array in shape of image but with values
        # drawn from a Gaussian distribution
        row,col,ch= image.shape
        mean = 0
        var = sp_var
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        
        # Multiply image by dist. and add to image
        noisy = image + image * gauss
        return noisy
    
def spatial_freq_filter(ps, rad=10):
    """ Takes a stack of images (indexed in middle column)
    and applys a circular cutoff in Fourier space at a radius
    defined in pixels by rad """
    
    num = np.shape(ps)[1]
    
    ps_filt = []
    for ipic in range(num):
        im = ps[:,ipic,:]
        ft = np.fft.fftshift(np.fft.fft2(im))

        mask = np.zeros_like(ft,dtype='uint8')
        cent = np.shape(mask)[0]/2
        for i, mi in enumerate(mask):
            for j, mij in enumerate(mi):
                x = cent - i
                y = cent - j
                if x**2 + y**2 < rad**2:
                    mask[i,j] = 1

        ftfilt = mask*ft
        imfilt = np.real(np.fft.ifft2(np.fft.fftshift(ftfilt)))
        
        ps_filt.append(imfilt)
        
    ps_filt = np.transpose(ps_filt,axes=[1,0,2]) # reshape so proj is middle column
    ps_filt=ps_filt.astype(np.float32)

    return ps_filt

def hologram_noise(ps,MX,MY,MZ,mesh_params,angles,n_pad=30,v=1,n=None,c=1000,fxc=400,fyc=400,rc=50,plot=False,fringe=10,up=10):
    """ Add realistic hologram-type noise to a phase image using libertem package
    """
    num = np.shape(ps)[1]
    
    ps_n = []
    
    # Loop over all images in stack
    for ipic in range(num):
        im = ps[:,ipic,:]
        phase_m = copy.deepcopy(im)
    
        # Upsample
        phase_tot = zoom(phase_m,(up,up))

        # estimate amplitude
        a=angles[ipic]
        MXr,MYr,MZr=rotate_magnetisation(MX,MY,MZ,a[0],a[1],a[2])
        mag =(MXr**2+MYr**2+MZr**2)**.5 # magnitude of magnetisation
        mag = mag/np.max(mag) # rescale so 1 is max
        thickness = project_along_z(mag,mesh_params=mesh_params) # project
        thickness = np.pad(thickness,[(n_pad,n_pad),(n_pad,n_pad)],mode='edge')
        amp = 1-thickness/np.max(thickness)/2
        amp = zoom(amp,(up,up))

        # Create hologram and reference
        holo = hologram_frame(np.ones_like(phase_tot), phase_tot,sampling=fringe,visibility=v,poisson_noise=n,counts=c)
        ref = hologram_frame(np.ones_like(phase_tot), np.zeros_like(phase_tot),sampling=fringe,visibility=v,poisson_noise=n)

        # Extract phase
        pu = reconstruct_hologram(holo,ref,fxc=fxc,fyc=fyc,rc=rc,plot=False)

        # Downsample
        phase_recon = zoom(pu,(1/up,1/up))
        
        ps_n.append(phase_recon)
        
    ps_n = np.transpose(ps_n,axes=[1,0,2]) # reshape so proj is middle column
    ps_n=ps_n.astype(np.float32)
    
    return ps_n

def create_circular_mask(imsize, center=(None), radius=None):
    h,w = imsize,imsize
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
    mask = np.ones((h,w))
    mask[dist_from_center >= radius] = 0
    #mask = dist_from_center <= radius
    
    # blur edge of mask
    mask = abs(ndimage.gaussian_filter(mask, sigma=1))
    
    return mask



def apply_circular_mask(circ_mask,im_ft):
    masked_im = im_ft.copy()
    masked_im = masked_im*circ_mask
    #masked_im[~circ_mask] = 0
    return masked_im

def centre_sideband(masked_im,fxc,fyc,rc):
    """ Centre by cropping around sideband centre then padding back out with zeros """
    size = np.shape(masked_im)[0]
    cropped_im = masked_im[fyc-2*rc:fyc+2*rc,fxc-2*rc:fxc+2*rc]
    size_cropped = np.shape(cropped_im)[0]
    size_pad = size-size_cropped
    centred_im = np.pad(cropped_im,int(size_pad/2),mode='constant')
    return centred_im

def extract_wf_and_phase(ob,ref):

    extracted_ob = np.fft.ifft2(ob)
    extracted_ref = np.fft.ifft2(ref)

    psi_0 = extracted_ob / extracted_ref
    phase = np.arctan2(np.imag(psi_0),np.real(psi_0)) # or np.angle(psi_0)

    return psi_0, phase

def reconstruct_hologram(holo,ref,fxc=135, fyc = 125,rc = 8,plot=False):
    ## hamming filter
    # create window
    size = np.shape(holo)[0]
    ham1d = np.hamming(size)
    ham2d = (np.outer(ham1d,ham1d))**1 # 0.5 normalises but might not remove cross

    # apply window
    ob = ham2d*holo
    ref = ham2d*ref
    
    ## FT
    ob_ft = np.fft.fft2(ob)
    ob_ft = np.fft.fftshift(ob_ft)
    ref_ft = np.fft.fft2(ref)
    ref_ft = np.fft.fftshift(ref_ft)
    
    ## select sideband
    circ_mask = create_circular_mask(size,center=[fxc,fyc],radius=rc)
    
    masked_ob = apply_circular_mask(circ_mask,ob_ft)
    masked_ref = apply_circular_mask(circ_mask,ref_ft)
    
    # auto-find centre
    maxval = np.amax(masked_ob)
    maxind = np.where(masked_ob == maxval)
    fxc,fyc = maxind[1][0], maxind[0][0]


    # auto-find radius
    dx = size/2 - fxc
    dy = size/2 - fyc
    rc = (dx**2 + dy**2)**0.5 / 2 #- 10

    # remask
    circ_mask = create_circular_mask(size,center=[fxc,fyc],radius=rc)
    masked_ob = apply_circular_mask(circ_mask,ob_ft)
    masked_ref = apply_circular_mask(circ_mask,ref_ft)
    

    if plot==True:
        plt.imshow(np.log10(abs(ob_ft)))
        plt.imshow(np.log10(abs(masked_ob)),cmap='Blues_r')
        plt.show()

    centred_ob = centre_sideband(masked_ob,fxc,fyc,int(rc))
    centred_ref = centre_sideband(masked_ref,fxc,fyc,int(rc))
    
    ## reconstruct wave function
    psi_0, phase = extract_wf_and_phase(centred_ob,centred_ref)
    
    ## unwrap phase
    pu = unwrap_phase(phase)
    #pu = (pu+abs(np.min(pu))) 
    
    return pu

def generate_A_projection(AX,AY,AZ,angles,mesh_params=None,unpad=False,reorient = False):
    """ Returns A projections for given angles
    in order [x, i_tilt, y] """
    # Initialise parameters
    ax_projs = []
    ay_projs = []
    az_projs = []
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(MX)
        p2 = (s[0],s[1],s[2])
        n = p2
        mesh_params = [p1,p2,n]
    
    # Loop through projection angles
    for i in range(len(angles)):
        ax,ay,az = angles[i]
        #rotate A
        AXr = rotate_bulk(AX,ax,ay,az)
        AYr = rotate_bulk(AY,ax,ay,az)
        AZr = rotate_bulk(AZ,ax,ay,az)
        ax_proj = project_along_z(AXr,mesh_params=mesh_params)
        ay_proj = project_along_z(AYr,mesh_params=mesh_params)
        az_proj = project_along_z(AZr,mesh_params=mesh_params)
        
        # reorient to match phase_projs
        if reorient == True:
            ax_proj = np.flipud(ax_proj.T)
            ay_proj = np.flipud(ay_proj.T)
            az_proj = np.flipud(az_proj.T)
        
        
        #calculate phase
        ax_projs.append(ax_proj)
        ay_projs.append(ay_proj)
        az_projs.append(az_proj)
    
    # Prepare projections for reconstruction
    ax_projs = np.transpose(ax_projs,axes=[1,0,2]) # reshape so proj is middle column
    ax_projs=ax_projs.astype(np.float32)
    ay_projs = np.transpose(ay_projs,axes=[1,0,2]) # reshape so proj is middle column
    ay_projs=ay_projs.astype(np.float32)
    az_projs = np.transpose(az_projs,axes=[1,0,2]) # reshape so proj is middle column
    az_projs=az_projs.astype(np.float32)
    
    return np.array(ax_projs),np.array(ay_projs),np.array(az_projs)

def generate_A_projection_fast(AX,AY,AZ,angles,mesh_params=None,unpad=False,reorient = True):
    """ Returns A projections for given angles
    in order [x, i_tilt, y], using astra forward projector """
    
    # Define some astra-specific things
    # Fairly sure that these ones in mm don't matter for parallel geom
    distance_source_origin = 300  # [mm]
    distance_origin_detector = 100  # [mm]
    detector_pixel_size = 1.05  # [mm]
    detector_rows = AX.shape[0]  # Vertical size of detector [pixels].
    detector_cols = AX.shape[0]  # Horizontal size of detector [pixels].
    
    # Create astra vectors to describe angles
    vecs = generate_vectors(angles)

    # AX
        # Create volume geometry
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                              detector_rows)
    
        # Reorient to match with old version
    AX = np.transpose(AX,[2,1,0])
    AX = AX[:,::-1,:]
    
        # Load data into astra
    phantom_idx = astra.data3d.create('-vol', vol_geom, data=AX)
    
        # Create projection geometry
    proj_geom = astra.create_proj_geom('parallel3d_vec', detector_rows, detector_cols, 
                                       np.array(vecs),(distance_source_origin + distance_origin_detector) /detector_pixel_size, 0)
        
        # Get forward projections
    projections_idx, projectionsx = astra.creators.create_sino3d_gpu(phantom_idx, proj_geom, vol_geom)
    
        # Clear astra memory
    astra.clear()
    
    # AY
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                              detector_rows)
    AY = np.transpose(AY,[2,1,0])
    AY = AY[:,::-1,:]
    phantom_idy = astra.data3d.create('-vol', vol_geom, data=AY)
    proj_geom = astra.create_proj_geom('parallel3d_vec', detector_rows, detector_cols, 
                                       np.array(vecs),(distance_source_origin + distance_origin_detector) /detector_pixel_size, 0)
    projections_idy, projectionsy = astra.creators.create_sino3d_gpu(phantom_idy, proj_geom, vol_geom)
    astra.clear()
    
    # AZ
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                              detector_rows)
    AZ = np.transpose(AZ,[2,1,0])
    AZ = AZ[:,::-1,:]
    phantom_idz = astra.data3d.create('-vol', vol_geom, data=AZ)
    proj_geom = astra.create_proj_geom('parallel3d_vec', detector_rows, detector_cols, 
                                       np.array(vecs),(distance_source_origin + distance_origin_detector) /detector_pixel_size, 0)
    projections_idz, projectionsz = astra.creators.create_sino3d_gpu(phantom_idz, proj_geom, vol_geom)
    astra.clear()
    
    # Scale correctly in line with previous version
    ax_projs = projectionsx*mesh_params[1][0]/mesh_params[2][0]
    ay_projs = projectionsy*mesh_params[1][0]/mesh_params[2][0]
    az_projs = projectionsz*mesh_params[1][0]/mesh_params[2][0]
    
    return np.array(ax_projs),np.array(ay_projs),np.array(az_projs)


def calculate_A_contributions(angles):
    """ For a given tilt angle series [[-70,0,0], [-60,0,0],...]
    calculate the weighting that the x,y,z components of A will contribute
    to each phase image in the series """
    
    ws = []
    
    for i, a in enumerate(angles):
        # calculate rotation matrix
        mrot = rotation_matrix(a[0],a[1],a[2])

        # Calculate position of x,y,z axes after rotation
        nx = np.dot(mrot,[1,0,0])
        ny = np.dot(mrot,[0,1,0])
        nz = np.dot(mrot,[0,0,1])
        
        #print(nx)

        # calculate how aligned the new x,y,z axes are with the beam direction
        # i.e. how much does this component contribute to the phase image?
        nx = np.dot(nx,[0,0,1])
        ny = np.dot(ny,[0,0,1])
        nz = np.dot(nz,[0,0,1])
        
        ws.append([nx,ny,nz])
    return np.array(ws)

def weight_phases(projs,ws):
    """ For a specific projection component, and its set of weights,
    multiplies those weights through the projection data 
    # checked and this definitely returns new_ps in same orientation as phase_proj """
    new_ps = []
    for i,w in enumerate(ws):
        p = projs[:,i,:]
        new_p = p*w
        new_ps.append(new_p)
        
    new_ps = np.transpose(new_ps,axes=[1,0,2]) # reshape so proj is middle column
    
    return new_ps

def update_weighted_proj_data(phase_projs,a_weighted_x,a_weighted_y,a_weighted_z,ws):
    """ Given a set of phase projection data, and the current best guess for each component af A,
        returns a new set projection data for each component, which is the raw data after removing
        the contribution of the other 2 components and reweighting it back to unity """
    const = -np.pi/constants.codata.value('mag. flux quantum')/(2*np.pi)
    new_x = phase_projs*1/const - weight_phases(a_weighted_y,ws[:,1]) - weight_phases(a_weighted_z,ws[:,2])
    new_x = weight_phases(new_x,1/ws[:,0])

    new_y = phase_projs*1/const - weight_phases(a_weighted_x,ws[:,0]) - weight_phases(a_weighted_z,ws[:,2])
    new_y = weight_phases(new_y,1/ws[:,1])

    new_z = phase_projs*1/const - weight_phases(a_weighted_y,ws[:,1]) - weight_phases(a_weighted_x,ws[:,0])
    new_z = weight_phases(new_z,1/ws[:,2])
    
    return new_x,new_y,new_z

def recon_step(a_projs,ws,angles,mesh_params, thresh=0.706, algorithm = 'SIRT3D_CUDA', niter=40, weight = 0.001,
                            balance = 1, steps = 'backtrack', callback_freq = 0):
    """ Given a set of A-component projections along with their weights and angles, does a SIRT reconstruction on it.
    It will only use projections where the component accounted for >threshold % of the original data in that slice.
    
    Input: Tilt series for a component of A, with it's associated weightings, angles and mesh parameters
    Specify: The threshold for this roudn of nmaj (thresh) and the number of iterations (niter)
    Return: A 3D reconstruction of 1 component of A """
    
    # Initialise parameters
    p1,p2,nn=mesh_params
    res=p2[0]/nn[0]
    a_thresh = []
    angles_thresh = []
    
    # Threshold out data with low weighting
    for i,w in enumerate(ws):
        if abs(w) > thresh:
            a_thresh.append(a_projs[:,i,:])
            angles_thresh.append(angles[i])
            
    angles_thresh=np.array(angles_thresh)
    a_thresh = np.transpose(a_thresh,axes=[1,0,2]) # reshape so proj is middle column  

    # Perform SIRT reconstruction on remaining data
    vecs = generate_vectors(angles_thresh)
    recon = generate_reconstruction(a_thresh,vecs, algorithm = algorithm, niter=niter, weight = weight,
                                balance = balance, steps = steps, callback_freq = callback_freq)
    
    # reformat to match structure of input data  
    recon = np.transpose(recon,axes=(2,1,0))[:,::-1,:]
    
    # Rescale intensities to account for pixel size
    recon = recon/res
    
    # Ensure astra doesn't fill up the RAM
    astra.clear()
    
    return recon

def iterative_update_algorithm(phase_projs,angles,mesh_params,n_pad,n_full_iter=1,n_step_iter=5, 
                               algorithm = 'SIRT3D_CUDA', weight = 0.001,thresh_range=(.01,.7),callback=False):
    """ Puts everything together for the multi-axis reconstruction procedure 
    Input: Phase tilt series, associated angles, mesh parameters, and pad count in pixels
    Specify: nmaj (n_full_iter), nmin (n_step_iter), and threshold range (tmin,tmax)
    Returns: Reconstructed Ax, Ay, Az arrays """
    
    if callback == True:
        callback_freq = 1
    else:
        callback_freq = 0
    
    # Calculate weightings for each tilt angle
    ws = calculate_A_contributions(angles)
    
    # In default run, threshold will initially be high (tmax)
    tmin,tmax = thresh_range
    # generate linearly spaced threshold list from low to high threshold for each nmaj
    possible_ts = np.linspace(tmin,tmax,n_full_iter-1)
    thresh = tmax
    
    # initialize new arrays
    a_weighted_x = np.zeros_like(phase_projs)
    a_weighted_y = np.zeros_like(phase_projs)
    a_weighted_z = np.zeros_like(phase_projs)
    
    # Generate A(0) tilt series
    a_weighted_x, a_weighted_y, a_weighted_z = update_weighted_proj_data(phase_projs,a_weighted_x,a_weighted_y,a_weighted_z,ws)
    
    if callback == True:
        print("Initialised")
      
    # do first step of reconstruction
    Ax_recon = recon_step(a_weighted_x,ws[:,0],angles,mesh_params,niter=n_step_iter,algorithm=algorithm,weight=weight,thresh=thresh, callback_freq =callback_freq) 
    Ay_recon = recon_step(a_weighted_y,ws[:,1],angles,mesh_params,niter=n_step_iter,algorithm=algorithm,weight=weight,thresh=thresh, callback_freq =callback_freq) 
    Az_recon = recon_step(a_weighted_z,ws[:,2],angles,mesh_params,niter=n_step_iter,algorithm=algorithm,weight=weight,thresh=thresh, callback_freq =callback_freq) 
    
    if callback == True:
        print("Iteration 1 finished")
    
    # Cycle through iterations for nmaj>1
    for i in range(n_full_iter-1):
        
        # Repeat t=tmax for nmaj=2, then decrease t for subsequent iterations
        thresh = possible_ts[-(i+1)]
        # recalculate projection data
       
        # project current A to get A_p(n)
        n=n_pad
        a_weighted_x,a_weighted_y,a_weighted_z = generate_A_projection_fast(Ax_recon[n:-n,n:-n,n:-n],Ay_recon[n:-n,n:-n,n:-n],Az_recon[n:-n,n:-n,n:-n],angles,mesh_params=mesh_params,reorient=True)
        
        # Update to get A_p(n+1)
        a_weighted_x = np.pad(a_weighted_x,[(n_pad,n_pad),(0,0),(n_pad,n_pad)], mode='constant', constant_values=0)
        a_weighted_y = np.pad(a_weighted_y,[(n_pad,n_pad),(0,0),(n_pad,n_pad)], mode='constant', constant_values=0)
        a_weighted_z = np.pad(a_weighted_z,[(n_pad,n_pad),(0,0),(n_pad,n_pad)], mode='constant', constant_values=0)
        a_weighted_x, a_weighted_y, a_weighted_z = update_weighted_proj_data(phase_projs,a_weighted_x,a_weighted_y,a_weighted_z,ws)
        
        # SIRT reconstruct to get A(n+1)
        Ax_recon = recon_step(a_weighted_x,ws[:,0],angles,mesh_params,niter=n_step_iter,thresh=thresh,algorithm=algorithm,weight=weight, callback_freq =callback_freq) 
        Ay_recon = recon_step(a_weighted_y,ws[:,1],angles,mesh_params,niter=n_step_iter,thresh=thresh,algorithm=algorithm,weight=weight, callback_freq =callback_freq) 
        Az_recon = recon_step(a_weighted_z,ws[:,2],angles,mesh_params,niter=n_step_iter,thresh=thresh,algorithm=algorithm,weight=weight, callback_freq =callback_freq) 
        
        # ensure astra doesn't clog up the RAM
        astra.clear()
        
        if callback == True:
            print("Iteration ",i+2," finished")
    
    return Ax_recon,Ay_recon,Az_recon

def dual_axis_B_generation(pxs,pys,mesh_params):
    """ Returns bxs, bys (projected B fields for tilt series) 
    Calculates the BX/BY component from the x/y tilt series
    """
    p1,p2,n = mesh_params
        
    x_size = p2[0]
    x_res = x_size/n[0]
    
    b_const = (constants.codata.value('mag. flux quantum')/(np.pi))
    bxs = []
    
    # calculate b component at each tilt
    for i in range(np.shape(pxs)[1]):
        p=pxs[:,i,:]
        # minus not needed as it goes bottom to top instead of top to bottom
        bx = b_const*np.gradient(p,x_res)[0]
        bxs.append(bx)
    
    bys = []
    # calculate b component at each tilt
    for i in range(np.shape(pys)[1]):
        p=pys[:,i,:]
        by = b_const*np.gradient(p,x_res)[1]
        bys.append(by)
    
    # reorder for tomo
    bxs = np.transpose(bxs,axes=[1,0,2])
    bys = np.transpose(bys,axes=[1,0,2])
    
    return bxs,bys

def dual_axis_reconstruction(xdata,ydata,axs,ays,mesh_params,algorithm = 'SIRT3D_CUDA', niter=100, weight = 0.001,
                            balance = 1, steps = 'backtrack', callback_freq = 0):
    """ Perform reconstruction of X/Y components on either phase or magnetic projections from dual axis series """
    p1,p2,nn=mesh_params
    resx=p2[0]/nn[0]
    resy=p2[1]/nn[1]

    # X series reconstruction
    vecs = generate_vectors(axs)
    rx = generate_reconstruction(xdata,vecs, algorithm = algorithm, niter=niter, weight = weight,
                                balance = balance, steps = steps, callback_freq = callback_freq)
    
    astra.clear()
    
    # Y series reconstruction
    vecs = generate_vectors(ays)
    ry = generate_reconstruction(ydata,vecs, algorithm = algorithm, niter=niter, weight =weight,
                                balance = balance, steps = steps, callback_freq =callback_freq)
    
    # Restructure data to match input M/A/B/p
    rx = np.transpose(rx,axes=(2,1,0))[:,::-1,:]
    ry = np.transpose(ry,axes=(2,1,0))[:,::-1,:]

    rx = rx/resx
    ry = ry/resy
    
    astra.clear()
    return rx,ry

def dual_axis_bz_from_bxby(bx,by):
    """ Calculates BZ from BX,BY using div(B)=0 """
    bz = []
    old=0
    for i in range(np.shape(bx)[2]):
        dbx = bx[:,:,i]
        dby = by[:,:,i]
        dbz = -1*(np.gradient(dbx)[0]+np.gradient(dby)[1])
        cum = dbz+old
        bz.append(cum)
        old=cum    

    bz = np.array(bz)    
    bz = np.transpose(bz,axes=[1,2,0])
    return bz

def generate_vectors(angles):
    """ Converts list of 3D projection angles into
    list of astra-compatible projection vectors,
    with [r,d,u,v] vectors on each row. """
    vectors = []
    for [ax,ay,az] in angles:
        vector = get_astravec(ax,ay,az)
        vectors.append(vector)
    
    return vectors

def get_astravec(ax,ay,az):
    """ Given angles in degrees, return r,d,u,v as a concatenation
    of four 3-component vectors"""
    # Since we us flipud on y axis, ay needs reversing for desired behaviour
    ay = -ay 
    
    # centre of detector
    d = [0,0,0]
    
    # 3D rotation matrix - EXTRINSIC!
    mrot = np.array(rotation_matrix(ax,ay,az,intrinsic=False))
    
    # ray direction r
    r = mrot.dot([0,0,1])*-1 # -1 to match astra definitions
    # u (det +x)
    u = mrot.dot([1,0,0])
    # v (det +y)
    v = mrot.dot([0,1,0])

    return np.concatenate((r,d,u,v))

def generate_reconstruction(raw_data,vectors, algorithm = 'SIRT3D_CUDA', niter=10, weight = 0.01,
                            balance = 1, steps = 'backtrack', callback_freq = 0):
    """ Chooise from 'SIRT3D_CUDA','FP3D_CUDA','BP3D_CUDA','CGLS3D_CUDA' or 'TV1'"""
    # Astra default algorithms
    if algorithm in ['SIRT3D_CUDA','FP3D_CUDA','BP3D_CUDA','CGLS3D_CUDA']:
        # Load data objects into astra C layer
        proj_geom = astra.create_proj_geom('parallel3d_vec',np.shape(raw_data)[0],np.shape(raw_data)[2],np.array(vectors))
        projections_id = astra.data3d.create('-sino', proj_geom, raw_data)
        vol_geom = astra.creators.create_vol_geom(np.shape(raw_data)[0], np.shape(raw_data)[0],
                                                  np.shape(raw_data)[2])
        reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
        alg_cfg = astra.astra_dict(algorithm)
        alg_cfg['ProjectionDataId'] = projections_id
        alg_cfg['ReconstructionDataId'] = reconstruction_id
        algorithm_id = astra.algorithm.create(alg_cfg)

        astra.algorithm.run(algorithm_id,iterations=niter)
        recon = astra.data3d.get(reconstruction_id)
    
    # CS TV using RTR
    if algorithm == 'TV1':
        data = rtr.tomo_data(raw_data, np.array(vectors), degrees=True,
                    tilt_axis=0, stack_dim=1)

        vol_shape = (data.shape[0],data.shape[0],data.shape[2])
        projector = data.getOperator(vol_shape=vol_shape,
                                    backend='astra',GPU=True)
        alg = rtr.TV(vol_shape, order=1)
        
        if callback_freq == 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,
                    callback=None)
            
        if callback_freq != 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,callback_freq = callback_freq,
                    callback=('primal','gap','violation','step'))[0]
            
    if algorithm == 'TV1_unnorm':
        data = rtr.tomo_data(raw_data, np.array(vectors), degrees=True,
                    tilt_axis=0, stack_dim=1)

        vol_shape = (data.shape[0],data.shape[0],data.shape[2])
        projector = data.getOperator(vol_shape=vol_shape,
                                    backend='astra',GPU=True)
        alg = rtr.TV_unnorm(vol_shape, order=1)
        
        if callback_freq == 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,
                    callback=None)
            
        if callback_freq != 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,callback_freq = callback_freq,
                    callback=('primal','gap','violation','step'))[0]
    
    if algorithm == 'TV2':
        data = rtr.tomo_data(raw_data, np.array(vectors), degrees=True,
                    tilt_axis=0, stack_dim=1)

        vol_shape = (data.shape[0],data.shape[0],data.shape[2])
        projector = data.getOperator(vol_shape=vol_shape,
                                    backend='astra',GPU=True)
        alg = rtr.TV(vol_shape, order=2)
        
        if callback_freq == 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,
                    callback=None)
            
        if callback_freq != 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,callback_freq = callback_freq,
                    callback=('primal','gap','violation','step'))[0]
            
    if 'wavelet' in algorithm:
        _,w = algorithm.split(sep='_')
        data = rtr.tomo_data(raw_data, np.array(vectors), degrees=True,
                    tilt_axis=0, stack_dim=1)

        vol_shape = (data.shape[0],data.shape[0],data.shape[2])
        projector = data.getOperator(vol_shape=vol_shape,
                                    backend='astra',GPU=True)
        alg = rtr.Wavelet(vol_shape, wavelet=w)
        
        if callback_freq == 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps='classic',
                    callback=None)
            
        if callback_freq != 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps='classic',callback_freq = callback_freq,
                    callback=('primal','gap','violation','step'))[0]
    
    return recon
    
    return recon

class reconstructions():
    def WRAP(phis,angles,mesh_params,n_pad=44,nmaj=3,nmin=15,weight=2e-3,th1=.2,th2=.8,callback=False):
        """ Reconstructs B field from magnetic phase images using WRAP 
            
            Inputs:
                phis: Stack of phase images (tilt number as middle column)
                angles: List of (ax,ay,az) rotations in degrees for each image
                mesh_params: [(bbx1,bby1,bbz1),(bbx2,bby2,bbz2), (nx,ny,nz)]
                             Parameters defining bounding box extent and resolution
                
                n_pad: Amount to pad image in each direction 
                       (wavelet regularisation works best
                        when image size is a square number, 
                        and image should be padded by at least the same
                        size to avoid fourier wrap-around artefacts.)
                nmaj: Number of major iterations
                nmin: Number of minor iterations
                weight: Wavelet regularisation weighting
                th1: WRAP lower threshold (applied in final nmaj)
                th2: WRAP upper threshold (applied for nmaj 1)
        """
        a1,a2,a3=iterative_update_algorithm(phis,angles,mesh_params,n_pad,n_full_iter=nmaj,n_step_iter=nmin, 
                                           thresh_range=(th1,th2),algorithm='wavelet_coif1',weight=weight,callback=callback)
        b1m,b2m,b3m = calculate_B_from_A(a1,a2,a3,mesh_params=mesh_params)
        
        return b1m,b2m,b3m
    
    def conventional(px,py,angx,angy,mesh_params,n_pad=44,niter=15,callback=False):
        """ Reconstructs B field from magnetic phase images using WRAP 
            
            Inputs:
                px,py: Stack of phase images (tilt number as middle column) for x,y tilt series
                angx,angy: List of (ax,ay,az) rotations in degrees for each image for x,y tilt series
                mesh_params: [(bbx1,bby1,bbz1),(bbx2,bby2,bbz2), (nx,ny,nz)]
                             Parameters defining bounding box extent and resolution
                
                niter: Number of SIRT iterations
        """
        bx_p,by_p = dual_axis_B_generation(px,py,mesh_params)
        if callback==True:
            callback_freq=5
        else:
            callback_freq = 0
        b1d,b2d = dual_axis_reconstruction(bx_p,by_p ,angx,angy,mesh_params,niter=niter,algorithm='SIRT3D_CUDA',callback_freq=callback_freq)
        b3d = dual_axis_bz_from_bxby(b1d,b2d)

        return b1d,b2d,b3d
    
def plot_component_orthoslices(X,Y,Z, vmin=False, vmax=False,npad=False, i = None, oslice = 'z'):
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    f,axs = plt.subplots(ncols=3,figsize=(14,4))
    
    # Check whether to set intensity limits to maximal extent
    if vmin == True:
        vmin = np.min([X,Y,Z])
        vmax = np.max([X,Y,Z])
            
    if vmin == False:
        vmin=None
        vmax = None
            
    # Choose what to plot
    if npad != False: # unpad
        x = X[npad:-npad,npad:-npad,npad:-npad]
        y = Y[npad:-npad,npad:-npad,npad:-npad]
        z = Z[npad:-npad,npad:-npad,npad:-npad]
    if i == None: # find central slice
        i = int(np.shape(x)[0]/2)
    if oslice=='z':
        x = x[:,:,i]
        y = y[:,:,i]
        z = z[:,:,i]
    if oslice=='y':
        x = x[:,i,:]
        y = y[:,i,:]
        z = z[:,i,:]
    if oslice=='x':
        x = x[i,:,:]
        y = y[i,:,:]
        z = z[i,:,:]
    
    # Plot X
    im1 = axs[0].imshow(x,vmin=vmin,vmax=vmax)
    
    if vmin == None:
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(im1, cax=cax, orientation='vertical')
            
    # Plot Y
    im2 = axs[1].imshow(y,vmin=vmin,vmax=vmax)
    
    if vmin == None:
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(im2, cax=cax, orientation='vertical')
            
    # Plot Z
    im3 = axs[2].imshow(z,vmin=vmin,vmax=vmax)
    
    divider = make_axes_locatable(axs[2]) # switch on colorbar
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im3, cax=cax, orientation='vertical')
    
    for ax in axs:
        ax.axis('off')
        
    f.patch.set_facecolor('white')
    
    plt.tight_layout()
    
def find_lim(X,Y,Z):
    """ Returns the biggest of abs(min) and max of magnitude of X,Y,Z """
    bmag = (X**2+Y**2+Z**2)**0.5
    bmax = np.max(bmag)
    bmin = np.min(bmag)
    blim = np.max([abs(bmax),abs(bmin)])
    
    bmag=np.max([X,Y,Z])
    bmin=np.min([X,Y,Z])
    
    return blim, bmax, bmin

class plotting():
    def interactive_phases(phis,angles=None):
        """ Plots phase shifts in an interactive way """
        lim = np.shape(phis)[1]-1

        def update(i):
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(phis[:,int(i),:],cmap='Greys_r',vmin=np.min(phis),vmax=np.max(phis))
            cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
            cbar.ax.set_title('Phase shift / rads', rotation=0,fontsize=14)
            vals = [100, 30,10]
            if type(angles) != type(None):
                title = r'($%.1f^{\circ},%.1f^{\circ},%.1f^{\circ})$' % tuple(angles[int(i)].tolist())
                plt.title(title,fontsize=13)
            for val in vals:
                if np.pi/val < np.max(phis):
                    cbar.ax.plot([-1,1], [np.pi/val,np.pi/val],'r-',markersize=50)
                    cbar.ax.text(1.3,np.pi/val,r'$\pi$ / %.i' % val,fontsize=13,color='r')
                    ax.axis('off')

        ipywidgets.interact(update,i=(0,lim,1))
        
    def compare_orthoslices(BX,BY,BZ,bxw,byw,bzw,bxc,byc,bzc,i=20,n=44): 
        """ Compare ideal B, WRAP B, and conventional B orthoslices
            i = Orthoslice index
            n = padding value     
        """
        b1m,b2m,b3m,b1d,b2d,b3d = bxw,byw,bzw,bxc,byc,bzc
        # Plot orthoslices of B with COD values
        blim,bmax,bmin = find_lim(BX,BY,BZ)

        plot_component_orthoslices(BX,BY,BZ,npad=n,i=i,vmax=bmax,vmin=bmin)
        plt.title('Phantom B \n COD X, Y, Z (Mean)    ',fontsize=20)

        codx = COD(BX[n:-n,n:-n,n:-n],b1m[n:-n,n:-n,n:-n])
        cody = COD(BY[n:-n,n:-n,n:-n],b2m[n:-n,n:-n,n:-n])
        codz = COD(BZ[n:-n,n:-n,n:-n],b3m[n:-n,n:-n,n:-n])
        plot_component_orthoslices(b1m,b2m,b3m,npad=n,i=i,vmax=blim,vmin=-blim)
        plt.title('WRAP algorithm \n %.2f, %.2f, %.2f (%.2f)' % (codx,cody,codz,np.mean([codx,cody,codz])),fontsize=20)

        codx = COD(BX[n:-n,n:-n,n:-n],b1d[n:-n,n:-n,n:-n])
        cody = COD(BY[n:-n,n:-n,n:-n],b2d[n:-n,n:-n,n:-n])
        codz = COD(BZ[n:-n,n:-n,n:-n],b3d[n:-n,n:-n,n:-n])
        plot_component_orthoslices(b1d,b2d,b3d,npad=n,i=i,vmax=blim,vmin=-blim)
        plt.title('Conventional algorithm \n %.2f, %.2f, %.2f (%.2f)' % (codx,cody,codz,np.mean([codx,cody,codz])),fontsize=20)
        
def COD(P,recon):
    """ Calculate the coefficinet of determination (1 perfect, 0 shit)"""
    P_mean = np.mean(P)
    R_mean = np.mean(recon)
    sumprod = np.sum((P-P_mean)*(recon-R_mean))
    geom_mean = np.sqrt(np.sum((P-P_mean)**2)*np.sum((recon-R_mean)**2))
    coeff_norm = sumprod/geom_mean
    COD = coeff_norm**2
    
    return COD
        
        
    