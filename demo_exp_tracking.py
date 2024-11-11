# ======================================================================================================================
# Demo 2 demonstrates how to use a model trained on simulations to localize experimental data. Again, the experimental
# frames are taken from our STORM experiment (Fig. 3 main text). Note that the frames saved in the folder
# <Experimental_Data/Tetrapod_demo2/> are those after minimum subtraction.
#
# IMPORTANT: The order of the channels is reversed due to a previous bug in the training code
# For a new tracking experiment, the concatenation order should be (im_psf1, im_psf2) in data_utils.py, line 372 inside class "ExpDataset"
# ======================================================================================================================

# import related script and packages to load the exp. image
import os
import numpy as np
import matplotlib.pyplot as plt
from DeepNebulae.Testing_Localization_Model import test_model

# pre-trained weights on simulations
path_curr = os.getcwd()
path_results = path_curr + '/Demos/Tracking/'

# path to experimental images
path_exp_data1 = path_curr + '/Data/Experimental_Data/LiveCell/psf1/'
path_exp_data2 = path_curr + '/Data/Experimental_Data/LiveCell/psf2/'

# postprocessing parameters
postprocessing_params = {'thresh': 80, 'radius': 4, 'keep_singlez': True}

# test the model on a sequence of experimental image pairs inside a live U2OS cell
scale_test, warp2to1 = False, False
xyz_rec, conf_rec, results_rec = test_model(path_results, postprocessing_params, scale_test, warp2to1, path_exp_data1, path_exp_data2)
t_rec = results_rec[:, 0]

# plt both 3D recoveries over time
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
p3d = ax3D.scatter(xyz_rec[:, 0], xyz_rec[:, 1], xyz_rec[:, 2], s=30, c=t_rec, marker='o', cmap=plt.cm.magma)   
ax3D.set_xlabel('X [um]')
ax3D.set_ylabel('Y [um]')
ax3D.set_zlabel('Z [um]')
ax3D.invert_yaxis()
ax3D.set_title('3D Recovered pts over time')
fig.colorbar(p3d, ax=ax3D)
plt.show()

