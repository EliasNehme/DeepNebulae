# ======================================================================================================================
# TODO: documentation
# ======================================================================================================================

# import the data generation and optics learning net functions
from Demos.parameter_setting_learn_twomasks import twomasks_parameters
from DeepNebulae.GenerateTrainingExamples import gen_data
from DeepNebulae.PSF_Learning import learn_masks

# specified training parameters
setup_params = twomasks_parameters()

# generate training data
gen_data(setup_params)

# learn two phase masks optimized for high density localization
learn_masks(setup_params)
