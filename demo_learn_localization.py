# ======================================================================================================================
# TODO: documentation
# ======================================================================================================================

# import the data generation and localization net learning functions
from Demos.parameter_setting_oracle import psf_pair_parameters
from DeepNebulae.GenerateTrainingExamples import gen_data
from DeepNebulae.Training_Localization_Model import learn_localization_cnn

# specified training parameters
setup_params = psf_pair_parameters()

# generate training data
gen_data(setup_params)

# learn a localization cnn
learn_localization_cnn(setup_params)
