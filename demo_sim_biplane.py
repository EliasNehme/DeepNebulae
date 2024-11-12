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
path_results = path_curr + '/Demos/Results_biplane/'

# postprocessing parameters
postprocessing_params = {'thresh': 80, 'radius': 4, 'keep_singlez': True}

# whether to scale the test images to match training statistics
scale_test = False  # True

# you can change this to randomize the sampled example
seed = 11 # 11, 10, 30, 33, 60

# Biplane PSFs
xyz_rec_oracle, conf_rec_oracle = test_model(path_results, postprocessing_params, scale_test, None, None, seed)

# show all plots
plt.show()
