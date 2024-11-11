# ======================================================================================================================
# Demo 4 demonstrates how to use DeepSTORM3D can be used to learn an optimal PSF for high density. The parameters
# assumed for this demo match our telomere simulations (Fig. 4 main text).The script shows the phase mask and the
# corresponding PSF being updated over iterations. The mask was initialized in this demo to zero-modulation.
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
