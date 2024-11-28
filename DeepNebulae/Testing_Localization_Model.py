# Import modules and libraries
import numpy as np
import glob
import time
from datetime import datetime
import argparse
import csv
import pickle
import scipy.io as sio
from skimage.io import imread
from skimage.io import imread, imsave
from skimage.transform import AffineTransform, warp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from DeepNebulae.data_utils import generate_batch, complex_to_tensor, affine_transform_xyz, global_shift_xyz, ExpDataset, sort_names_tif
from DeepNebulae.cnn_utils import LocalizationCNN
from DeepNebulae.vis_utils import ShowMaskPSF, ShowRecovery3D, ShowLossJaccardAtEndOfEpoch
from DeepNebulae.vis_utils import PhysicalLayerVisualization, ShowRecNetInput
from DeepNebulae.physics_utils import EmittersToPhases, calc_bfp_grids
from DeepNebulae.postprocess_utils import Postprocess
from DeepNebulae.assessment_utils import calc_jaccard_rmse
from DeepNebulae.helper_utils import normalize_01, xyz_to_nm


def test_model(path_results, postprocess_params, scale_test=False, exp_imgs_path1=None, exp_imgs_path2=None, seed=66):

    # close all existing plots
    plt.close("all")

    # load assumed setup parameters
    path_params_pickle = path_results + 'setup_params.pickle'
    with open(path_params_pickle, 'rb') as handle:
        setup_params = pickle.load(handle)

    # run on GPU if available
    device = setup_params['device']
    torch.backends.cudnn.benchmark = True
    
    # ensure we do not assume a higher number of GPUs than available
    if device.type=='cuda':
        if device.index > torch.cuda.device_count()-1:
            device = torch.device("cuda:0")

    # phase term for PSF visualization
    vis_term, zvis = setup_params['vis_term'], setup_params['zvis']

    # phase mask for visualization
    mask_param1 = torch.from_numpy(setup_params['mask_init'][0]).to(device)
    mask_param2 = torch.from_numpy(setup_params['mask_init'][1]).to(device)

    # plot used masks and PSFs
    plt.figure(figsize=(10,5))
    ShowMaskPSF(mask_param1, vis_term, zvis)
    plt.figure(figsize=(10,5))
    ShowMaskPSF(mask_param2, vis_term, zvis)

    # load learning results
    path_learning_pickle = path_results + 'learning_results.pickle'
    with open(path_learning_pickle, 'rb') as handle:
        learning_results = pickle.load(handle)

    # plot metrics evolution in training for debugging
    plt.figure()
    ShowLossJaccardAtEndOfEpoch(learning_results, learning_results['epoch_converged'])

    # build model and convert all the weight tensors to GPU is available
    cnn = LocalizationCNN(setup_params)
    cnn.to(device)

    # load learned weights
    cnn.load_state_dict(torch.load(path_results + 'weights_best_loss.pkl', map_location=device))

    # post-processing module on CPU/GPU
    thresh, radius, keep_singlez = postprocess_params['thresh'], postprocess_params['radius'], postprocess_params['keep_singlez']
    postprocessing_module = Postprocess(setup_params, thresh, radius, keep_singlez)

    # if no experimental imgs are supplied then sample a random simulated example
    if exp_imgs_path1 is None or exp_imgs_path2 is None:

        # visualization module to visualize the 3D positions recovered by the net as images
        psf_module_vis = PhysicalLayerVisualization(setup_params, blur_flag=False, noise_flag=False, norm_flag=True)
        
        # initialize the physical layer that encodes xyz into noisy PSFs
        psf_module_net = PhysicalLayerVisualization(setup_params, blur_flag=True, noise_flag=True, norm_flag=setup_params['project_01'])
        
        # set random number generators given the seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ==============================================================================================================
        # generate a simulated test image pair
        # ==============================================================================================================

        # sample a single piece of data
        xyz_nomask, nphotons_gt = generate_batch(1, setup_params)
        
        ##### PSF 1
        # xyz positions for PSF 1 (include xy registration errors)
        setup_params['reg_err'] = 0.0  # for testing in simulation assume zero registration error
        xyz_psf1 = global_shift_xyz(xyz_nomask, [0.0, 0.0], setup_params['reg_err'])
        
        # calculate phases and cast them to tensors for PSF 1
        phase_emitters1 = EmittersToPhases(xyz_psf1, setup_params)
        phases_tensor1 = complex_to_tensor(phase_emitters1)
        
        # cast number of photons to tensor for PSF 1
        Nphotons_tensor1 = torch.from_numpy(nphotons_gt).type(torch.FloatTensor)
        
        # calculate the image PSF 1
        im_psf1 = psf_module_net(mask_param1.to(device), phases_tensor1.to(device), Nphotons_tensor1.to(device))
        im_psf1_np = np.squeeze(im_psf1.data.cpu().numpy())
        
        # warp image of PSF 1 to match image of PSF 2
        tform_1to2 = AffineTransform(matrix=setup_params['T12_px'])  # assumption is we estimated 1->2, and 2 have lower SNR
        im_psf1_np = warp(im_psf1_np, tform_1to2.inverse, order=3, preserve_range=False)
        
        # transfer image of PSF 1 back to a tensor
        im_psf1_np = np.expand_dims(im_psf1_np, 0)
        im_psf1_np = np.expand_dims(im_psf1_np, 0)
        im_psf1 = torch.FloatTensor(im_psf1_np).to(device)
        
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
        im_psf2 = psf_module_net(mask_param2.to(device), phases_tensor2.to(device), Nphotons_tensor2.to(device))

        # normalize image according to the global factors assuming it was not projected to [0,1]
        if setup_params['project_01'] is False:
            im_psf1 = (im_psf1 - setup_params['global_factors'][0]) / setup_params['global_factors'][1]
            im_psf2 = (im_psf2 - setup_params['global_factors'][0]) / setup_params['global_factors'][1]
        
        # scale image if specified to match the mean and std of the training set
        if setup_params['project_01'] is True and scale_test is True:
            im_psf1 = (im_psf1 - im_psf1.mean())/im_psf1.std()
            im_psf1 = im_psf1*setup_params['train_stats'][1][0] + setup_params['train_stats'][0][0]
            im_psf2 = (im_psf2 - im_psf2.mean())/im_psf2.std()
            im_psf2 = im_psf2*setup_params['train_stats'][1][1] + setup_params['train_stats'][0][1]
        
        # concatenate both channels to a tensor
        test_input_imgs = torch.cat((im_psf1, im_psf2), 1)

        # ==============================================================================================================
        # predict the positions by post-processing the net's output
        # ==============================================================================================================

        # prediction using model
        cnn.eval()
        with torch.set_grad_enabled(False):
            pred_volume = cnn(test_input_imgs)

        # time prediction using model after first forward pass which is slow
        tinf_start = time.time()
        with torch.set_grad_enabled(False):
            pred_volume = cnn(test_input_imgs)
        tinf_elapsed = time.time() - tinf_start
        print('Inference complete in {:.6f}s'.format(tinf_elapsed))
        
        # post-process predicted volume
        tpost_start = time.time()
        xyz_rec, conf_rec = postprocessing_module(pred_volume)
        tpost_elapsed = time.time() - tpost_start
        print('Post-processing complete in {:.6f}s'.format(tpost_elapsed))

        # take out dim emitters from GT
        if setup_params['nsig_unif'] is False:
            nemitters = xyz_label.shape[1]
            if np.not_equal(nemitters, 1):
                nphotons_gt = np.squeeze(nphotons_gt, 0)
                xyz_label = xyz_label[:, nphotons_gt > setup_params['nsig_thresh'], :]

        # plot recovered 3D positions compared to GT
        plt.figure()
        xyz_label = np.squeeze(xyz_label, 0)
        ShowRecovery3D(xyz_label, xyz_rec)

        # report the number of found emitters
        print('Found {:d} emitters out of {:d}'.format(xyz_rec.shape[0], xyz_label.shape[0]))

        # calculate quantitative metrics assuming a matching radius of 100 nm
        jaccard_index, RMSE_xy, RMSE_z, _ = calc_jaccard_rmse(xyz_label, xyz_rec, 0.1)

        # report quantitative metrics
        print('Jaccard Index = {:.2f}%, Lateral RMSE = {:.2f} nm, Axial RMSE = {:.2f}'.format(
            jaccard_index*100, RMSE_xy*1e3, RMSE_z*1e3))

        # ==============================================================================================================
        # compare the network positions to the input images
        # ==============================================================================================================
        
        # recovered positions are the positions of PSF 2
        xyz_rec2 = np.expand_dims(xyz_rec, 0)

        # turn recovered positions into phases
        phases_np2 = EmittersToPhases(xyz_rec2, setup_params)
        phases_emitter_rec2 = complex_to_tensor(phases_np2).to(device)

        # use a uniform number of photons for recovery visualization
        nphotons_rec = 5000 * torch.ones((1, xyz_rec2.shape[1])).to(device)

        # generate the recovered PSF 2 image by the net
        im_psf2_rec = psf_module_vis(mask_param2, phases_emitter_rec2, nphotons_rec)
        
        # get some approximate recovered positions for PSF 1 by applying the inverse transform
        xyz_rec1 = affine_transform_xyz(xyz_rec2, np.linalg.inv(setup_params['T12_um']))

        # turn recovered positions into phases
        phases_np1 = EmittersToPhases(xyz_rec1, setup_params)
        phases_emitter_rec1 = complex_to_tensor(phases_np1).to(device)

        # generate the recovered image by the net
        im_psf1_rec = psf_module_vis(mask_param1, phases_emitter_rec1, nphotons_rec)
        
        # compare the recovered images to the input
        ShowRecNetInput(im_psf1, 'Simulated PSF 1 Input to Localization Net')
        ShowRecNetInput(im_psf1_rec, 'Recovered PSF 1 Input Matching Net Localizations')
        ShowRecNetInput(im_psf2, 'Simulated PSF 2 Input to Localization Net')
        ShowRecNetInput(im_psf2_rec, 'Recovered PSF 2 Input Matching Net Localizations')
        
        # return recovered locations and net confidence
        return np.squeeze(xyz_rec2,0), conf_rec

    else:

        # read all imgs in the experimental data directory assuming ".tif" extension
        img_names_psf1 = glob.glob(exp_imgs_path1 + '*.tif')
        img_names_psf2 = glob.glob(exp_imgs_path2 + '*.tif')
        assert len(img_names_psf1) == len(img_names_psf2), 'Number of images in both folders must be the same'

        # if given only 1 image then show xyz in 3D and recovered image
        if len(img_names_psf1) == 1:

            # ==========================================================================================================
            # read experimental images and normalize them
            # ==========================================================================================================
            
            # read exp image in uint16 and turn to float32
            exp_im_psf1 = (imread(img_names_psf1[0])).astype("float32")
            exp_im_psf2 = (imread(img_names_psf2[0])).astype("float32")
            
            # normalize image according to the training setting
            if setup_params['project_01'] is True:
                exp_im_psf1 = normalize_01(exp_im_psf1)
                exp_im_psf2 = normalize_01(exp_im_psf2)
            else:
                exp_im_psf1 = (exp_im_psf1 - setup_params['global_factors'][0]) / setup_params['global_factors'][1]
                exp_im_psf2 = (exp_im_psf2 - setup_params['global_factors'][0]) / setup_params['global_factors'][1]

            # scale images if specified to match the mean and std of the training set
            if setup_params['project_01'] is True and scale_test is True:
                exp_im_psf1 = (exp_im_psf1 - exp_im_psf1.mean())/exp_im_psf1.std()
                exp_im_psf1 = exp_im_psf1*setup_params['train_stats'][1][0] + setup_params['train_stats'][0][0]
                exp_im_psf2 = (exp_im_psf2 - exp_im_psf2.mean())/exp_im_psf2.std()
                exp_im_psf2 = exp_im_psf2*setup_params['train_stats'][1][1] + setup_params['train_stats'][0][1]

            # warp image of PSF 1 to match image of PSF 2, assumption is we estimated 1->2, and 2 have lower SNR
            tform_1to2 = AffineTransform(matrix=setup_params['T12_px'])
            exp_im_psf1 = warp(exp_im_psf1, tform_1to2.inverse, order=3, preserve_range=False)
            
            # turn image into torch tensor with 1 channel on GPU
            exp_im_psf1 = np.expand_dims(np.expand_dims(exp_im_psf1, 0), 0)
            exp_im_psf2 = np.expand_dims(np.expand_dims(exp_im_psf2, 0), 0)
            
            # final input tensor
            exp_tensor = torch.FloatTensor(np.concatenate((exp_im_psf1, exp_im_psf2), 1)).to(device)

            # ==========================================================================================================
            # predict the positions by post-processing the net's output
            # ==========================================================================================================

            # prediction using model
            cnn.eval()
            with torch.set_grad_enabled(False):
                pred_volume = cnn(exp_tensor)

            # time prediction using model after first forward pass which is slow
            tinf_start = time.time()
            with torch.set_grad_enabled(False):
                pred_volume = cnn(exp_tensor)
            tinf_elapsed = time.time() - tinf_start
            print('Inference complete in {:.6f}s'.format(tinf_elapsed))
            
            # post-process predicted volume
            tpost_start = time.time()
            xyz_rec, conf_rec = postprocessing_module(pred_volume)
            tpost_elapsed = time.time() - tpost_start
            print('Post-processing complete in {:.6f}s'.format(tpost_elapsed))

            # plot recovered 3D positions compared to GT
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter(xyz_rec[:, 0], xyz_rec[:, 1], xyz_rec[:, 2], c='r', marker='^', label='DL', depthshade=False)
            ax.set_xlabel('X [um]')
            ax.set_ylabel('Y [um]')
            ax.set_zlabel('Z [um]')
            ax.invert_yaxis()
            ax.set_title('3D Recovered Positions')

            # report the number of found emitters
            print('Found {:d} emitters'.format(xyz_rec.shape[0]))

            # ==========================================================================================================
            # compare the network positions to the input images
            # ==========================================================================================================
            
            # visualization module to visualize the 3D positions recovered by the net as images
            H, W = exp_im_psf1.shape[-2], exp_im_psf1.shape[-1]
            setup_params['H'], setup_params['W'] = H, W
            psf_module_vis = PhysicalLayerVisualization(setup_params, blur_flag=False, noise_flag=False, norm_flag=True)
            
            # if the grid is too large then pre-compute the phase grids again
            if H > setup_params['Hmask'] or W > setup_params['Wmask']:
                setup_params['pixel_size_SLM'] /= 2
                setup_params['Hmask'] *= 2
                setup_params['Wmask'] *= 2
                setup_params = calc_bfp_grids(setup_params)
                mask_param = interpolate(mask_param.unsqueeze(0).unsqueeze(1), scale_factor=(2,2), mode="nearest")
                mask_param = mask_param.squeeze(0).squeeze(1)
                psf_module_vis = PhysicalLayerVisualization(setup_params, blur_flag=False, noise_flag=False, norm_flag=True)
            
            # recovered positions are the positions of the PSF 2
            xyz_rec2 = np.expand_dims(xyz_rec, 0)

            # turn recovered positions into phases
            phases_np2 = EmittersToPhases(xyz_rec2, setup_params)
            phases_emitter_rec2 = complex_to_tensor(phases_np2).to(device)

            # use a uniform number of photons for recovery visualization
            nphotons_rec = 5000 * torch.ones((1, xyz_rec2.shape[1])).to(device)

            # generate the recovered PSF 2 image by the net
            exp_psf2_rec = psf_module_vis(mask_param2, phases_emitter_rec2, nphotons_rec)
            
            # get some approximate recovered positions for PSF 1 by applying the inverse transform
            xyz_rec1 = affine_transform_xyz(xyz_rec2, np.linalg.inv(setup_params['T12_um']))

            # turn recovered positions into phases
            phases_np1 = EmittersToPhases(xyz_rec1, setup_params)
            phases_emitter_rec1 = complex_to_tensor(phases_np1).to(device)

            # generate the recovered image by the net
            exp_psf1_rec = psf_module_vis(mask_param1, phases_emitter_rec1, nphotons_rec)
            
            # compare the recovered images to the input
            ShowRecNetInput(exp_im_psf1, 'Exp PSF 1 Input to Localization Net')
            ShowRecNetInput(exp_psf1_rec, 'Rec PSF 1 Input Matching Net Localizations')
            ShowRecNetInput(exp_im_psf2, 'Exp PSF 2 Input to Localization Net')
            ShowRecNetInput(exp_psf2_rec, 'Rec PSF 2 Input Matching Net Localizations')
            
            # return recovered locations and net confidence
            return np.squeeze(xyz_rec2,0), conf_rec

        else:

            # ==========================================================================================================
            # create a data generator to efficiently load imgs for temporal acquisitions
            # ==========================================================================================================
            
            
            # sort images by number (assumed names are <digits>.tif)
            img_names_psf1 = sort_names_tif(img_names_psf1)
            img_names_psf2 = sort_names_tif(img_names_psf2)
            assert len(img_names_psf1) == len(img_names_psf2), 'Number of images in both folders must be the same'
            for im_name1, im_name2 in zip(img_names_psf1, img_names_psf2):
                assert im_name1.split('/')[-1] == im_name2.split('/')[-1], 'Image names must match'

            # instantiate the data class and create a data loader for testing
            num_imgs = len(img_names_psf1)
            exp_test_set = ExpDataset(img_names_psf1, img_names_psf2, setup_params, scale_test)
            exp_generator = DataLoader(exp_test_set, batch_size=1, shuffle=False)
            
            # time the entire dataset analysis
            tall_start = time.time()

            # needed pixel-size for plotting if only few images are in the folder
            pixel_size_FOV = setup_params['pixel_size_FOV']

            # needed recovery pixel size and minimal axial height for turning ums to nms
            psize_rec_xy, zmin = setup_params['pixel_size_rec'], setup_params['zmin']

            # process all experimental images
            cnn.eval()
            results = np.array(['frame', 'x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]'])
            with torch.set_grad_enabled(False):
                for im_ind, exp_im_tensor in enumerate(exp_generator):

                    # print current image number
                    print('Processing Image [%d/%d]' % (im_ind + 1, num_imgs))

                    # time each frame
                    tfrm_start = time.time()

                    # transfer normalized image to device (CPU/GPU)
                    exp_im_tensor = exp_im_tensor.to(device)

                    # predicted volume using model
                    pred_volume = cnn(exp_im_tensor)

                    # post-process result to get the xyz coordinates and their confidence
                    xyz_rec, conf_rec = postprocessing_module(pred_volume)

                    # time it takes to analyze a single frame
                    tfrm_end = time.time() - tfrm_start

                    # if this is the first image, get the dimensions and the relevant center for plotting
                    if im_ind == 0:
                        _, _, H, W = exp_im_tensor.size()
                        ch, cw = np.floor(H / 2), np.floor(W / 2)

                    # if prediction is empty then set number of found emitters to 0
                    # otherwise generate the frame column and append results for saving
                    if xyz_rec is None:
                        nemitters = 0
                    else:
                        nemitters = xyz_rec.shape[0]
                        frm_rec = (im_ind + 1)*np.ones(nemitters)
                        xyz_save = xyz_to_nm(xyz_rec, H*2, W*2, psize_rec_xy, zmin)
                        results = np.vstack((results, np.column_stack((frm_rec, xyz_save, conf_rec))))

                    # visualize the first 10 images regardless of the number of expeimental frames
                    visualize_flag = True if im_ind < 10 else (num_imgs <= 100)
                    
                    # if the number of imgs is small then plot each image in the loop with localizations
                    if visualize_flag:

                        # show input image
                        im_np = np.squeeze(exp_im_tensor.cpu().numpy())
                        im_psf1 = im_np[1, :, :]
                        im_psf2 = im_np[0, :, :]
                        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, num=100)
                        im1 = ax1.imshow(im_psf1, cmap='gray')
                        ax1.plot(xyz_rec[:, 0] / pixel_size_FOV + cw, xyz_rec[:, 1] / pixel_size_FOV + ch, 'r+')
                        ax1.set_title('PSF 1: Frame complete in {:.2f}s, found {:d} emitters'.format(tfrm_end, nemitters))
                        plt.colorbar(im1, ax=ax1)
                        im2 = ax2.imshow(im_psf2, cmap='gray')
                        ax2.plot(xyz_rec[:, 0] / pixel_size_FOV + cw, xyz_rec[:, 1] / pixel_size_FOV + ch, 'r+')
                        ax2.set_title('PSF 2: Frame complete in {:.2f}s, found {:d} emitters'.format(tfrm_end, nemitters))
                        plt.colorbar(im2, ax=ax2)
                        plt.draw()
                        plt.pause(0.05)
                        plt.clf()

                    else:

                        # print status
                        print('Single frame complete in {:.6f}s, found {:d} emitters'.format(tfrm_end, nemitters))

            # print the time it took for the entire analysis
            tall_end = time.time() - tall_start
            print('=' * 50)
            print('Analysis complete in {:.0f}h {:.0f}m {:.0f}s'.format(
                tall_end // 3600, np.floor((tall_end / 3600 - tall_end // 3600) * 60), tall_end % 60))
            print('=' * 50)
            
            # write the results to a csv file named "localizations_<date>_<time>.csv" under the exp img folder of psf 1
            row_list = results.tolist()
            curr_dt = datetime.now()
            curr_date = f'{curr_dt.day:02d}{curr_dt.month:02d}{curr_dt.year:04d}'
            curr_time = f'{curr_dt.hour:02d}{curr_dt.minute:02d}{curr_dt.second:02d}'
            loc_name = 'localizations' + '_' + curr_date + '_' + curr_time + '.csv'
            with open(exp_imgs_path1 + loc_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)

            # return the localization results for the last image
            return xyz_rec, conf_rec, results


if __name__ == '__main__':

    # start a parser
    parser = argparse.ArgumentParser()

    # previously trained model
    parser.add_argument('--path_results', help='path to the results folder for the pre-trained model', required=True)

    # previously trained model
    parser.add_argument('--postprocessing_params', help='post-processing dictionary parameters', required=True)
    
    # previously trained model
    parser.add_argument('--scale_test', action='store_true', help='whether to normalize the test images to match stats in training')

    # path to the experimental images of PSF 1
    parser.add_argument('--exp_imgs_path1', default=None, help='path to the experimental test images of PSF 1')
    
    # path to the experimental images of PSF 1
    parser.add_argument('--exp_imgs_path2', default=None, help='path to the experimental test images of PSF 2')

    # seed to run model
    parser.add_argument('--seed', default=66, help='seed for random test data generation')

    # parse the input arguments
    args = parser.parse_args()

    # run the data generation process
    xyz_rec, conf_rec = test_model(args.path_results, args.postprocessing_params, args.scale_test, 
                                   args.exp_imgs_path1, args.exp_imgs_path2, args.seed)
