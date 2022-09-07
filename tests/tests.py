import numpy as np
import WRAP

phis_s, angles_s, BX_s,BY_s,BZ_s, mesh_params_s,MX_s,MY_s,MZ_s,AX_s,AY_s,AZ_s = WRAP.dataset_generator.realistic_sphere()
bxw,byw,bzw = WRAP.reconstructions.WRAP(phis_s,angles_s,mesh_params_s,n_pad=44,nmaj=3,nmin=10,weight=3e-3,th1=.2,th2=.8)
bxc,byc,bzc = WRAP.reconstructions.conventional(px,py,angx,angy,mesh_params_s,niter=5)