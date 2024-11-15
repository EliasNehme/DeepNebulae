# this script encapsulates all needed parameters for training/learning a phase mask

# import needed libraries
import os
import numpy as np
from math import pi
import scipy.io as sio


def setup_parameters():

    # path to current directory
    path_curr_dir = os.getcwd()

    # ======================================================================================
    # initial masks and training mode
    # ======================================================================================

    # boolean flags to specify whether we are learning the masks or not
    learn_mask = [False, False]

    # initial masks for learning optimized masks or final masks for training a localization model
    # if learn_mask=True the initial mask is initialized by default to be zero-modulation
    path_masks = path_curr_dir + '/Mat_Files/masks_learned_fixed_vipr.mat'
    mask_dict = sio.loadmat(path_masks)
    mask_init = [mask_dict['mask1'], mask_dict['mask2']]

    # mask options dictionary
    mask_opts = {'learn_mask': learn_mask, 'mask_init': mask_init}

    # ======================================================================================
    # optics settings: objective, light, sensor properties
    # ======================================================================================

    lamda = 0.605  # mean emission wavelength # in [um] (1e-6*meter)
    NA = 1.49  # numerical aperture of the objective lens
    noil = 1.518  # immersion medium refractive index
    nwater = 1.33  # imaging medium refractive index
    pixel_size_CCD = 11  # sensor pixel size in [um] (including binning)
    pixel_size_SLM = 24  # SLM pixel size in [um] (after binning of 3 to reduce computational complexity)
    M = 100  # optical magnification
    f_4f = 15e4  # 4f lenses focal length in [um]

    # optical settings dictionary
    optics_dict = {'lamda': lamda, 'NA': NA, 'noil': noil, 'nwater': nwater, 'pixel_size_CCD': pixel_size_CCD,
                   'pixel_size_SLM': pixel_size_SLM, 'M': M, 'f_4f': f_4f}

    # ======================================================================================
    # sensor matching and affine transforms
    # ======================================================================================

    # transforms from no mask to a mask in each camera and between cameras
    # a transform set to identity means no change in (homogeneous) coordinates
    Tpsf1 = np.eye(3, 3)  # transform from no mask to mask 1 (for example shift in xy) 
    Tpsf2 = np.eye(3, 3)  # transform from no mask to mask 2 (for example shift in xy)
    Tpsf1to2_px = np.eye(3, 3)  # transform from mask 1 to mask 2 in px (origin = top left corner of FOV)
    Tpsf1to2_um = np.eye(3, 3)  # transform from mask 1 to mask 2 in um (origin = center of FOV)

    # global shift to cover the affine range
    FOV_shift_range = [-5.0, 5.0]  # in [um]

    # registration error between the two cameras applied to the EDOF channel
    reg_err = 0.0  # in [um] 0.05

    # sensor matching dictionary
    tform_dict = {'T1': Tpsf1, 'T2': Tpsf2, 'T12_px': Tpsf1to2_px, 'T12_um': Tpsf1to2_um, 
                  'FOV_shift_range': FOV_shift_range, 'reg_err': reg_err}

    # ======================================================================================
    # phase mask and image space dimensions for simulation
    # ======================================================================================

    # phase mask dimensions
    Hmask, Wmask = 343, 343  # in SLM [pixels]

    # single training image dimensions
    H, W = 121, 121  # in sensor [pixels]

    # safety margin from the boundary to prevent PSF truncation
    clear_dist = 20  # in sensor [pixels]

    # training z-range anf focus
    zmin = 1  # minimal z in [um] (including the axial shift)
    zmax = 4.5  # maximal z in [um] (including the axial shift)
    NFP = 3  # nominal focal plane in [um] (including the axial shift)

    # discretization in z
    D = 81  # in [voxels] spanning the axial range (zmax - zmin)

    # data dimensions dictionary
    data_dims_dict = {'Hmask': Hmask, 'Wmask': Wmask, 'H': H, 'W': W, 'clear_dist': clear_dist, 'zmin': zmin,
                      'zmax': zmax, 'NFP': NFP, 'D': D}

    # ======================================================================================
    # number of emitters in each FOV
    # ======================================================================================

    # upper and lower limits for the number fo emitters
    num_particles_range = [1, 50]

    # number of particles dictionary
    num_particles_dict = {'num_particles_range': num_particles_range}

    # ======================================================================================
    # signal counts distribution and settings
    # ======================================================================================

    # boolean that specifies whether the signal counts are uniformly distributed
    nsig_unif = True

    # range of signal counts assuming a uniform distribution
    nsig_unif_range = [20000, 120000]  # in [counts]

    # parameters for sampling signal counts assuming a gamma distribution
    nsig_gamma_params = None  # in [counts]

    # threshold on signal counts to discard positions from the training labels
    nsig_thresh = None  # in [counts]

    # range to randomize the fraction of counts in the second sensor (if Nc=2)
    nsig_ratio_range = [0.75, 0.95]  # in [counts]

    # signal counts dictionary
    nsig_dict = {'nsig_unif': nsig_unif, 'nsig_unif_range': nsig_unif_range, 'nsig_gamma_params': nsig_gamma_params,
                 'nsig_thresh': nsig_thresh, 'nsig_ratio_range': nsig_ratio_range}

    # ======================================================================================
    # blur standard deviation for smoothing PSFs to match experimental conditions
    # ======================================================================================

    # upper and lower blur standard deviation for each emitter to account for finite size
    blur_std_range = [0.75, 1]  # in sensor [pixels]

    # blur dictionary
    blur_dict = {'blur_std_range': blur_std_range}

    # ======================================================================================
    # uniform/non-uniform background settings
    # ======================================================================================

    # uniform background value per pixel
    unif_bg = 0  # in [counts]

    # boolean flag whether or not to include a non-uniform background
    nonunif_bg_flag = True

    # maximal offset for the center of the non-uniform background in pixels
    nonunif_bg_offset = [10, 10]  # in sensor [pixels]

    # peak and valley minimal values for the super-gaussian; randomized with addition of up to 50%
    nonunif_bg_minvals = [20.0, 150.0]  # in [counts]

    # minimal and maximal angle of the super-gaussian for augmentation
    nonunif_bg_theta_range = [-pi/4, pi/4]  # in [radians]

    # nonuniform background dictionary
    nonunif_bg_dict = {'nonunif_bg_flag': nonunif_bg_flag, 'unif_bg': unif_bg, 'nonunif_bg_offset': nonunif_bg_offset,
                       'nonunif_bg_minvals': nonunif_bg_minvals, 'nonunif_bg_theta_range': nonunif_bg_theta_range}

    # ======================================================================================
    # read noise settings
    # ======================================================================================

    # boolean flag whether or not to include read noise
    read_noise_flag = True

    # flag whether of not the read noise standard deviation is not uniform across the FOV
    read_noise_nonuinf = False

    # range of baseline of the min-subtracted data in STORM
    read_noise_baseline_range = [200.0, 250.0]  # in [counts]

    # read noise standard deviation upper and lower range
    read_noise_std_range = [20.0, 24.0]  # in [counts]

    # read noise dictionary
    read_noise_dict = {'read_noise_flag': read_noise_flag, 'read_noise_nonuinf': read_noise_nonuinf,
                       'read_noise_baseline_range': read_noise_baseline_range,
                       'read_noise_std_range': read_noise_std_range}

    # ======================================================================================
    # image normalization settings
    # ======================================================================================

    # boolean flag whether or not to project the images to the range [0, 1]
    project_01 = True

    # global normalization factors for STORM (subtract the first and divide by the second)
    global_factors = [None, None]  # in [counts]

    # image normalization dictionary
    norm_dict = {'project_01': project_01, 'global_factors': global_factors}

    # ======================================================================================
    # training data settings
    # ======================================================================================

    # number of training and validation examples
    ntrain = 9000
    nvalid = 1000

    # path for saving training examples: images + locations for localization net or locations + photons for PSF learning
    training_data_path = path_curr_dir + "/TrainingImages_two_masks_learned_3105/"

    # boolean flag whether to visualize examples while created
    visualize = False

    # training data dictionary
    training_dict = {'ntrain': ntrain, 'nvalid': nvalid, 'training_data_path': training_data_path, 'visualize': visualize}

    # ======================================================================================
    # learning settings
    # ======================================================================================

    # results folder to save the trained model
    results_path = path_curr_dir + "/Results_two_masks_learned_3105/"

    # maximal dilation flag when learning a localization CNN (set to None if learn_mask=True as we use a different CNN)
    dilation_flag = False  # if set to 1 then dmax=16 otherwise dmax=4

    # batch size for training a localization model (set to 1 for mask learning as examples are generated 16 at a time)
    batch_size = 4

    # maximal number of epochs
    max_epochs = 50

    # initial learning rate for adam
    initial_learning_rate = 0.0005

    # scaling factor for the loss function
    scaling_factor = 800.0

    # learning dictionary
    learning_dict = {'results_path': results_path, 'dilation_flag': dilation_flag, 'batch_size': batch_size,
                     'max_epochs': max_epochs, 'initial_learning_rate': initial_learning_rate,
                     'scaling_factor': scaling_factor}

    # ======================================================================================
    # resuming from checkpoint settings
    # ======================================================================================

    # boolean flag whether to resume training from checkpoint
    resume_training = False

    # number of epochs to resume training
    num_epochs_resume = None

    # saved checkpoint to resume from
    checkpoint_path = None

    # checkpoint dictionary
    checkpoint_dict = {'resume_training': resume_training, 'num_epochs_resume': num_epochs_resume,
                       'checkpoint_path': checkpoint_path}

    # ======================================================================================
    # final resulting dictionary including all parameters
    # ======================================================================================

    settings = {**mask_opts, **num_particles_dict, **nsig_dict, **blur_dict, **nonunif_bg_dict, **read_noise_dict,
                **norm_dict, **optics_dict, **data_dims_dict, **training_dict, **learning_dict, **checkpoint_dict,
                **tform_dict}

    return settings


if __name__ == '__main__':
    parameters = setup_parameters()
