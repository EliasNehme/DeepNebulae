# ======================================================================================================================
# Demo 3 demonstrates how to use a model trained on simulations to localize an experimental image of telomeres within a
# U2OS cell nucleus. The PSFs in the experimental data cover all of the 3 proposed methods. For visual comparison the
# script also regenerate the input image depending on the CNN localizations.
# ======================================================================================================================

# import related script and packages to load the exp. image
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from DeepNebulae.Testing_Localization_Model import test_model

# pre-trained weights on simulations
path_curr = os.getcwd()
path_results1 = path_curr + '/Demos/Results_oracle/'
path_results2 = path_curr + '/Demos/Results_crlb/'
path_results3 = path_curr + '/Demos/Results_learned_fixed/'

# fixed cell number (can be either 1 or 2)
cell_num = 1  # 2

# path to experimental images
path_exp_data1 = path_curr + f'/Experimental_Data/FixedCell{cell_num}/tetra/'
path_exp_data2 = path_curr + f'/Experimental_Data/FixedCell{cell_num}/edof/'
path_exp_data3 = path_curr + f'/Experimental_Data/FixedCell{cell_num}/crlb1/'
path_exp_data4 = path_curr + f'/Experimental_Data/FixedCell{cell_num}/crlb2/'
path_exp_data5 = path_curr + f'/Experimental_Data/FixedCell{cell_num}/learn1/'
path_exp_data6 = path_curr + f'/Experimental_Data/FixedCell{cell_num}/learn2/'

# postprocessing parameters
postprocessing_params = {'thresh': 20, 'radius': 4, 'keep_singlez': True}

# whether to scale the test images to match training statistics
scale_test = False  # True

# Tetrapod + EDOF PSFs
xyz_rec_oracle, conf_rec_oracle = test_model(path_results1, postprocessing_params, scale_test, path_exp_data1, path_exp_data2)

# CRLB-optimized PSFs
xyz_rec_crlb, conf_rec_crlb = test_model(path_results2, postprocessing_params, scale_test, path_exp_data3, path_exp_data4)

# Learned PSFs
xyz_rec_learn, conf_rec_learn = test_model(path_results3, postprocessing_params, scale_test, path_exp_data5, path_exp_data6)

# save the positions and confidence values
rec_dict = {'xyz_oracle': xyz_rec_oracle, 'conf_oracle': conf_rec_oracle, 
            'xyz_crlb': xyz_rec_crlb, 'conf_crlb': conf_rec_crlb,
            'xyz_learn': xyz_rec_learn, 'conf_learn': conf_rec_learn}
sio.savemat(path_curr + f'/results_exp_fixedCell{cell_num}.mat', rec_dict)

# plt all 3D recoveries
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xyz_rec_oracle[:, 0], xyz_rec_oracle[:, 1], xyz_rec_oracle[:, 2], c='b', marker='^', label='TP + EDOF', depthshade=False)
ax.scatter(xyz_rec_crlb[:, 0], xyz_rec_crlb[:, 1], xyz_rec_crlb[:, 2], c='r', marker='P', label='CRLB', depthshade=False)
ax.scatter(xyz_rec_learn[:, 0], xyz_rec_learn[:, 1], xyz_rec_learn[:, 2], c='g', marker='o', label='Learned', depthshade=False)
ax.set_xlabel('X [um]')
ax.set_ylabel('Y [um]')
ax.set_zlabel('Z [um]')
ax.invert_yaxis()
plt.legend()
plt.title('Comparison of 3D Recovered pts')
plt.show()
