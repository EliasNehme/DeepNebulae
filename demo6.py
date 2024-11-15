# ======================================================================================================================
# Demo 6 demonstrates the usage of this code to generate a training set and learn a localization model for other PSFs
# besides the one proposed in our paper. Specifically we illsutrate the utility of this code for localizing biplane 
# PSFs, which are probably the most widely used PSF pair in microscopy, requiring only the introduction of a defocus 
# between the imaging channels, without further optical elements.
# ======================================================================================================================

# import related script and packages
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
