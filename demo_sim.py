# ======================================================================================================================
# TODO: documentation
# ======================================================================================================================

# import related script and packages to load the exp. image
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from DeepNebulae.Testing_Localization_Model import test_model

# pre-trained weights on simulations
path_curr = os.getcwd()
path_results1 = path_curr + '/Demos/Results_two_masks_oracle/'
path_results2 = path_curr + '/Demos/Results_two_masks_crlb/'
path_results3 = path_curr + '/Demos/Results_two_masks_learned/'

# postprocessing parameters
postprocessing_params = {'thresh': 80, 'radius': 4, 'keep_singlez': True}

# whether to scale the test images to match training statistics
scale_test = False  # True

# you can change this to randomize the sampled example
seed = 11 # 11, 10, 30, 33, 60

# Tetrapod + EDOF PSFs
xyz_rec_oracle, conf_rec_oracle = test_model(path_results1, postprocessing_params, scale_test, None, None, seed)

# CRLB-optimized PSFs
xyz_rec_crlb, conf_rec_crlb = test_model(path_results2, postprocessing_params, scale_test, None, None, seed)

# Learned PSFs
xyz_rec_learn, conf_rec_learn = test_model(path_results3, postprocessing_params, scale_test, None, None, seed)

# save the positions and confidence values
rec_dict = {'xyz_oracle': xyz_rec_oracle, 'conf_oracle': conf_rec_oracle, 
            'xyz_crlb': xyz_rec_crlb, 'conf_crlb': conf_rec_crlb,
            'xyz_learn': xyz_rec_learn, 'conf_learn': conf_rec_learn}
sio.savemat(path_curr + f'/results_sim_seed{seed}.mat', rec_dict)

# plt all 3D recoveries
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xyz_rec_oracle[:, 0], xyz_rec_oracle[:, 1], xyz_rec_oracle[:, 2], c='b', marker='^', label='TP + EDOF', depthshade=False)
ax.scatter(xyz_rec_crlb[:, 0], xyz_rec_crlb[:, 1], xyz_rec_crlb[:, 2], c='r', marker='P', label='CRLB', depthshade=False)
ax.scatter(xyz_rec_learn[:, 0], xyz_rec_learn[:, 1], xyz_rec_learn[:, 2], c='g', marker='o', label='Learn', depthshade=False)
ax.set_xlabel('X [um]')
ax.set_ylabel('Y [um]')
ax.set_zlabel('Z [um]')
ax.invert_yaxis()
plt.legend()
plt.title('Comparison of 3D Recovered pts')

# show all plots
plt.show()
