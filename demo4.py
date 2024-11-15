# ======================================================================================================================
# Demo 4 demonstrates how DeepNebulae can be used to learn an optimal PSF pair for high density (Fig. 7 main text). The 
# parameters assumed for this demo match our telomere simulations.The script shows the phase masks and the corresponding 
# PSFs being updated over iterations. The masks were initialized in this demo to zero-modulation.
# ======================================================================================================================

# import the data generation and optics learning net functions
from Demos.parameter_setting_psfs_learning import psf_pair_parameters
from DeepNebulae.GenerateTrainingExamples import gen_data
from DeepNebulae.PSF_Learning import learn_masks

# specified training parameters
setup_params = psf_pair_parameters()

# generate training data
gen_data(setup_params)

# learn two phase masks optimized for high density localization
learn_masks(setup_params)
