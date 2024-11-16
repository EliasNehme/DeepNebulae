# Import modules and libraries
import os
import pickle
import argparse
import numpy as np
from PIL import Image
from skimage.transform import AffineTransform, warp
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from DeepNebulae.data_utils import generate_batch, complex_to_tensor, affine_transform_xyz, global_shift_xyz
from DeepNebulae.physics_utils import calc_bfp_grids, EmittersToPhases, PhysicalLayer
from DeepNebulae.helper_utils import CalcMeanStd_All


# generate training data (either images for localization cnn or locations and intensities for psf learning)
def gen_data(setup_params):

    # random seed for repeatability
    torch.manual_seed(999)
    np.random.seed(566)

    # calculate on GPU if available
    device = setup_params['device']
    torch.backends.cudnn.benchmark = True

    # calculate the effective sensor pixel size taking into account magnification and set the recovery pixel size to be
    # the same such that sampling of training positions is performed on this coarse grid
    setup_params['pixel_size_FOV'] = setup_params['pixel_size_CCD'] / setup_params['M']  # in [um]
    setup_params['pixel_size_rec'] = setup_params['pixel_size_FOV'] / 1  # in [um]

    # calculate the axial range and the axial pixel size depending on the volume discretization
    setup_params['axial_range'] = setup_params['zmax'] - setup_params['zmin']  # [um]
    setup_params['pixel_size_axial'] = setup_params['axial_range'] / setup_params['D']  # [um]

    # calculate back focal plane grids and needed terms for on the fly PSF calculation
    setup_params = calc_bfp_grids(setup_params)

    # training data folder for saving
    path_train = setup_params['training_data_path']
    if not (os.path.isdir(path_train)):
        os.mkdir(path_train)

    # print status
    print('=' * 50)
    print('Sampling examples for training')
    print('=' * 50)

    # batch size for generating training examples:
    # locations for phase mask learning are saved in batches of 16 for convenience
    if np.any(setup_params['learn_mask']):
        batch_size_gen = 16
    else:
        batch_size_gen = 1
    setup_params['batch_size_gen'] = batch_size_gen

    # calculate the number of training batches to sample
    ntrain_batches = int(setup_params['ntrain'] / batch_size_gen)
    setup_params['ntrain_batches'] = ntrain_batches

    # phase masks initialization and psf module for simulation
    if not(np.all(setup_params['learn_mask'])):
        mask_init = setup_params['mask_init']
        if mask_init[0] is None and mask_init[1] is None:
            print('If the training mode is not set to learning at least one phase mask then you should supply two phase masks')
            return
        else:
            mask_param1 = torch.from_numpy(mask_init[0])
            mask_param2 = torch.from_numpy(mask_init[1])
            psf_module = PhysicalLayer(setup_params)
            psf_module.eval()
    
    # generate examples for training
    labels_dict = {}
    for i in range(ntrain_batches):

        # sample a training example
        xyz_nomask, Nphotons = generate_batch(batch_size_gen, setup_params)

        # if we intend to learn a mask, simply save locations and intensities
        if np.any(setup_params['learn_mask']):

            # save xyz, N, to labels dict
            labels_dict[str(i)] = {'xyz': xyz_nomask, 'N': Nphotons}

        # otherwise, create the respective images and save only the locations
        else:

            ##### PSF 1
            # xyz positions for PSF 1 (include xy registration errors)
            xyz_psf1 = global_shift_xyz(xyz_nomask, [0.0, 0.0], setup_params['reg_err'])
            
            # calculate phases and cast them to tensors for PSF 1
            phase_emitters1 = EmittersToPhases(xyz_psf1, setup_params)
            phases_tensor1 = complex_to_tensor(phase_emitters1)
            
            # cast number of photons to tensor for PSF 1
            Nphotons_tensor1 = torch.from_numpy(Nphotons).type(torch.FloatTensor)
            
            # calculate the image PSF 1
            im_psf1 = psf_module(mask_param1.to(device), phases_tensor1.to(device), Nphotons_tensor1.to(device))
            im_psf1_np = np.squeeze(im_psf1.data.cpu().numpy())
            
            # warp image of PSF 1 to match image of PSF 2, assumption is we estimated 1->2, and 2 have lower SNR
            tform_1to2 = AffineTransform(matrix=setup_params['T12_px'])
            im_psf1_np = warp(im_psf1_np, tform_1to2.inverse, order=3, preserve_range=False)
            
            ##### PSF 2
            # shift the simulated FOV globally to cover the affine transform range
            FOV_shift_range = setup_params['FOV_shift_range']
            shift_x = np.random.rand(1)*(FOV_shift_range[1] - FOV_shift_range[0]) + FOV_shift_range[0]
            shift_y = np.random.rand(1) * (FOV_shift_range[1] - FOV_shift_range[0]) + FOV_shift_range[0]
            xyz_global_shift = global_shift_xyz(xyz_nomask, [shift_x, shift_y])

            # affine transform to get the xyz in PSF 2 up to the global shift
            xyz_psf2_global_shift = affine_transform_xyz(xyz_global_shift, setup_params['T12_um'])

            # eliminate the global shift to get the xyz in PSF 2
            xyz_psf2 = global_shift_xyz(xyz_psf2_global_shift, [-shift_x, -shift_y])
            
            # set the label to the positions of PSF 2
            xyz_label = xyz_psf2

            # calculate phases from simulated locations and cast them to a tensor
            phase_emitters2 = EmittersToPhases(xyz_psf2, setup_params)
            phases_tensor2 = complex_to_tensor(phase_emitters2)
            
            # randomly reduce the number of photons in the second sensor
            nsig_ratio_range = setup_params['nsig_ratio_range']
            photons_factor = np.random.rand(1) * (nsig_ratio_range[1] - nsig_ratio_range[0]) + nsig_ratio_range[0]
            Nphotons_tensor2 = Nphotons_tensor1 * photons_factor[0]
            
            # calculate the image PSF 2
            im_psf2 = psf_module(mask_param2.to(device), phases_tensor2.to(device), Nphotons_tensor2.to(device))
            im_psf2_np = np.squeeze(im_psf2.data.cpu().numpy())

            # normalize image according to the global factors assuming it was not projected to [0,1]
            if setup_params['project_01'] is False:
                im_psf1_np = (im_psf1_np - setup_params['global_factors'][0]) / setup_params['global_factors'][1]
                im_psf2_np = (im_psf2_np - setup_params['global_factors'][0]) / setup_params['global_factors'][1]

            # look at the images if specified
            if setup_params['visualize']:

                # squeeze batch dimension in xyz
                xyzp = np.squeeze(xyz_label, 0)
                pixel_size_FOV, W, H = setup_params['pixel_size_FOV'], setup_params['W'], setup_params['H']

                # plot the image and the simulated xy centers on top
                fig1, axs = plt.subplots(1, 2, figsize=(10, 5), num=1)
                _ = axs[0].imshow(im_psf1_np, cmap='gray')
                axs[0].plot(xyzp[:, 0] / pixel_size_FOV + np.floor(W / 2), xyzp[:, 1] / pixel_size_FOV + np.floor(H / 2), 'r+')
                axs[0].set_title(f'PSF 1 {i}')
                imfig2 = axs[1].imshow(im_psf2_np, cmap='gray')
                axs[1].plot(xyzp[:, 0] / pixel_size_FOV + np.floor(W / 2), xyzp[:, 1] / pixel_size_FOV + np.floor(H / 2), 'r+')
                axs[1].set_title(f'PSF 2 {i}')
                
                # add colorbar and draw
                fig1.subplots_adjust(right=0.8)
                cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
                fig1.colorbar(imfig2, cax=cbar_ax)
                plt.draw()
                plt.pause(0.05)
                plt.clf()

            # threshold out dim emitters if counts are gamma distributed
            if (setup_params['nsig_unif'] is False) and (xyz_label.shape[1] > 1):
                Nphotons = np.squeeze(Nphotons)
                xyz_label = xyz_label[:, Nphotons > setup_params['nsig_thresh'], :]

            # save images as a tiff file and xyz to labels dict
            img1_name_tiff = path_train + 'im_psf1_' + str(i) + '.tiff'
            img1 = Image.fromarray(im_psf1_np)
            img1.save(img1_name_tiff)
            img2_name_tiff = path_train + 'im_psf2_' + str(i) + '.tiff'
            img2 = Image.fromarray(im_psf2_np)
            img2.save(img2_name_tiff)
            labels_dict[str(i)] = xyz_label

        # print number of example
        print('Training Example [%d / %d]' % (i + 1, ntrain_batches))

    # calculate training set mean and standard deviation (if we are generating images)
    if setup_params['learn_mask'] is False:
        train_stats = CalcMeanStd_All(path_train, labels_dict)
        setup_params['train_stats'] = train_stats

    # print status
    print('=' * 50)
    print('Sampling examples for validation')
    print('=' * 50)

    # calculate the number of training batches to sample
    nvalid_batches = int(setup_params['nvalid'] // batch_size_gen)
    setup_params['nvalid_batches'] = nvalid_batches

    # set the number of particles to the middle of the range for validation
    num_particles_range = setup_params['num_particles_range']
    setup_params['num_particles_range'] = [num_particles_range[1]//2, num_particles_range[1]//2 + 1]

    # sample validation examples
    for i in range(nvalid_batches):

        # sample a training example
        xyz_nomask, Nphotons = generate_batch(batch_size_gen, setup_params)

        # if we intend to learn a mask, simply save locations and intensities
        if np.any(setup_params['learn_mask']):

            # save xyz, N, to labels dict
            labels_dict[str(i + ntrain_batches)] = {'xyz': xyz_nomask, 'N': Nphotons}

        # otherwise, create the respective image and save only the locations
        else:
            
            ##### PSF 1
            # xyz positions for PSF 1 (include xy registration errors)
            xyz_psf1 = global_shift_xyz(xyz_nomask, [0.0, 0.0], setup_params['reg_err'])
            
            # calculate phases and cast them to tensors for PSF 1
            phase_emitters1 = EmittersToPhases(xyz_psf1, setup_params)
            phases_tensor1 = complex_to_tensor(phase_emitters1)
            
            # cast number of photons to tensor for PSF 1
            Nphotons_tensor1 = torch.from_numpy(Nphotons).type(torch.FloatTensor)
            
            # calculate the image PSF 1
            im_psf1 = psf_module(mask_param1.to(device), phases_tensor1.to(device), Nphotons_tensor1.to(device))
            im_psf1_np = np.squeeze(im_psf1.data.cpu().numpy())
            
            # warp image of PSF 1 to match image of PSF 2, assumption is we estimated 1->2, and 2 have lower SNR
            tform_1to2 = AffineTransform(matrix=setup_params['T12_px'])
            im_psf1_np = warp(im_psf1_np, tform_1to2.inverse, order=3, preserve_range=False)
            
            ##### PSF 2
            # shift the simulated FOV globally to cover the affine transform range
            FOV_shift_range = setup_params['FOV_shift_range']
            shift_x = np.random.rand(1)*(FOV_shift_range[1] - FOV_shift_range[0]) + FOV_shift_range[0]
            shift_y = np.random.rand(1) * (FOV_shift_range[1] - FOV_shift_range[0]) + FOV_shift_range[0]
            xyz_global_shift = global_shift_xyz(xyz_nomask, [shift_x, shift_y])

            # affine transform to get the xyz in PSF 2 up to the global shift
            xyz_psf2_global_shift = affine_transform_xyz(xyz_global_shift, setup_params['T12_um'])

            # eliminate the global shift to get the xyz in PSF 2
            xyz_psf2 = global_shift_xyz(xyz_psf2_global_shift, [-shift_x, -shift_y])
            
            # set the label to the positions of PSF 2
            xyz_label = xyz_psf2

            # calculate phases from simulated locations and cast them to a tensor
            phase_emitters2 = EmittersToPhases(xyz_psf2, setup_params)
            phases_tensor2 = complex_to_tensor(phase_emitters2)
            
            # randomly reduce the number of photons in the second sensor
            nsig_ratio_range = setup_params['nsig_ratio_range']
            photons_factor = np.random.rand(1) * (nsig_ratio_range[1] - nsig_ratio_range[0]) + nsig_ratio_range[0]
            Nphotons_tensor2 = Nphotons_tensor1 * photons_factor[0]
            
            # calculate the image PSF 2
            im_psf2 = psf_module(mask_param2.to(device), phases_tensor2.to(device), Nphotons_tensor2.to(device))
            im_psf2_np = np.squeeze(im_psf2.data.cpu().numpy())

            # normalize image according to the global factors assuming it was not projected to [0,1]
            if setup_params['project_01'] is False:
                im_psf1_np = (im_psf1_np - setup_params['global_factors'][0]) / setup_params['global_factors'][1]
                im_psf2_np = (im_psf2_np - setup_params['global_factors'][0]) / setup_params['global_factors'][1]

            # look at the images if specified
            if setup_params['visualize']:

                # squeeze batch dimension in xyz
                xyzp = np.squeeze(xyz_label, 0)
                pixel_size_FOV, W, H = setup_params['pixel_size_FOV'], setup_params['W'], setup_params['H']

                # plot the image and the simulated xy centers on top
                fig1, axs = plt.subplots(1, 2, figsize=(10, 5), num=1)
                _ = axs[0].imshow(im_psf1_np, cmap='gray')
                axs[0].plot(xyzp[:, 0] / pixel_size_FOV + np.floor(W / 2), xyzp[:, 1] / pixel_size_FOV + np.floor(H / 2), 'r+')
                axs[0].set_title(f'PSF 1 {i}')
                imfig2 = axs[1].imshow(im_psf2_np, cmap='gray')
                axs[1].plot(xyzp[:, 0] / pixel_size_FOV + np.floor(W / 2), xyzp[:, 1] / pixel_size_FOV + np.floor(H / 2), 'r+')
                axs[1].set_title(f'PSF 2 {i}')
                
                # add colorbar and draw
                fig1.subplots_adjust(right=0.8)
                cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
                fig1.colorbar(imfig2, cax=cbar_ax)
                plt.draw()
                plt.pause(0.05)
                plt.clf()

            # threshold out dim emitters if counts are gamma distributed
            if (setup_params['nsig_unif'] is False) and (xyz_label.shape[1] > 1):
                Nphotons = np.squeeze(Nphotons)
                xyz_label = xyz_label[:, Nphotons > setup_params['nsig_thresh'], :]
            
            # save images as a tiff file and xyz to labels dict
            img1_name_tiff = path_train + 'im_psf1_' + str(i + ntrain_batches) + '.tiff'
            img1 = Image.fromarray(im_psf1_np)
            img1.save(img1_name_tiff)
            img2_name_tiff = path_train + 'im_psf2_' + str(i + ntrain_batches) + '.tiff'
            img2 = Image.fromarray(im_psf2_np)
            img2.save(img2_name_tiff)
            labels_dict[str(i + ntrain_batches)] = xyz_label

        # print number of example
        print('Validation Example [%d / %d]' % (i + 1, nvalid_batches))

    # save all xyz's dictionary as a pickle file
    path_labels = path_train + 'labels.pickle'
    with open(path_labels, 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # set the number of particles back to the specified range
    setup_params['num_particles_range'] = num_particles_range

    # partition built in simulation
    ind_all = np.arange(0, ntrain_batches + nvalid_batches, 1)
    list_all = ind_all.tolist()
    list_IDs = [str(i) for i in list_all]
    train_IDs = list_IDs[:ntrain_batches]
    valid_IDs = list_IDs[ntrain_batches:]
    partition = {'train': train_IDs, 'valid': valid_IDs}
    setup_params['partition'] = partition

    # update recovery pixel in xy to be x4 smaller if we are learning a localization net
    if not(np.all(setup_params['learn_mask'])):
        setup_params['pixel_size_rec'] = setup_params['pixel_size_FOV'] / 4  # in [um]

    # save setup parameters dictionary for training and testing
    path_setup_params = path_train + 'setup_params.pickle'
    with open(path_setup_params, 'wb') as handle:
        pickle.dump(setup_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print status
    print('Finished sampling examples!')

    # close figure if it was open for visualization
    if not(np.all(setup_params['learn_mask'])) and setup_params['visualize']:
        plt.close(fig1)


if __name__ == '__main__':

    # start a parser
    parser = argparse.ArgumentParser()

    # previously wrapped settings dictionary
    parser.add_argument('--setup_params', help='path to the parameters wrapped in the script parameter_setting', required=True)

    # parse the input arguments
    args = parser.parse_args()

    # run the data generation process
    gen_data(args.setup_params)
