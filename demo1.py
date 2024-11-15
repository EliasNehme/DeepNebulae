# ======================================================================================================================
# Demo 1 demonstrates how to generate a training set and learn a localization model for a given optical setup. The
# parameters assumed in this demo are matched to our telomere imaging experiment (Fig. 3 main text). To understand 
# how we set the parameters in the function demo1_parameters() please refer to the documentation in 
# <Docs/demo1_documentation>.
# ======================================================================================================================

# import the data generation and localization net learning functions
from DeepNebulae.GenerateTrainingExamples import gen_data
from DeepNebulae.Training_Localization_Model import learn_localization_cnn

# specift desired setup parameters
from Demos.parameter_setting_oracle import setup_parameters

# specified training parameters
setup_params = setup_parameters()

# generate training data
gen_data(setup_params)

# learn a localization cnn
learn_localization_cnn(setup_params)
