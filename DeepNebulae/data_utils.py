# Import modules and libraries
import os
import numpy as np
from skimage.io import imread
from skimage.transform import AffineTransform, warp
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from DeepNebulae.physics_utils import EmittersToPhases
from DeepNebulae.helper_utils import normalize_01


# ======================================================================================================================
# numpy array conversion to variable and numpy complex conversion to 2 channel torch tensor
# ======================================================================================================================


# function converts numpy array on CPU to torch Variable on GPU
def to_var(x):
    """
    Input is a numpy array and output is a torch variable with the data tensor
    on cuda.
    """

    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# function converts numpy array on CPU to torch Variable on GPU
def complex_to_tensor(phases_np):
    Nbatch, Nemitters, Hmask, Wmask = phases_np.shape
    phases_torch = torch.zeros((Nbatch, Nemitters, Hmask, Wmask, 2)).type(torch.FloatTensor)
    phases_torch[:, :, :, :, 0] = torch.from_numpy(np.real(phases_np)).type(torch.FloatTensor)
    phases_torch[:, :, :, :, 1] = torch.from_numpy(np.imag(phases_np)).type(torch.FloatTensor)
    return phases_torch


# ======================================================================================================================
# continuous emitter positions sampling using two steps: first sampling disjoint indices on a coarse 3D grid,
# and afterwards refining each index using a local perturbation.
# ======================================================================================================================


# Define a batch data generator for training and testing
def generate_batch(batch_size, setup_params, seed=None):

    # if we're testing then seed the random generator
    if seed is not None:
        np.random.seed(seed)
    
    # randomly vary the number of emitters
    num_particles_range = setup_params['num_particles_range']
    num_particles = np.random.randint(num_particles_range[0], num_particles_range[1], 1).item()
    
    # distrbution for sampling the number of counts per emitter
    if setup_params['nsig_unif']:

        # uniformly distributed counts per emitter (counts == photons assuming gain=1)
        Nsig_range = setup_params['nsig_unif_range']
        Nphotons = np.random.randint(Nsig_range[0], Nsig_range[1], (batch_size, num_particles))
        Nphotons = Nphotons.astype('float32')
    else:

        # gamma distributed counts per emitter
        gamma_params = setup_params['nsig_gamma_params']
        Nphotons = np.random.gamma(gamma_params[0], gamma_params[1], (batch_size, num_particles))
        Nphotons = Nphotons.astype('float32')

    # calculate upsampling factor
    pixel_size_FOV, pixel_size_rec = setup_params['pixel_size_FOV'], setup_params['pixel_size_rec']
    upsampling_factor = pixel_size_FOV / pixel_size_rec

    # update dimensions
    H, W, D, clear_dist = setup_params['H'], setup_params['W'], setup_params['D'], setup_params['clear_dist']
    H, W, clear_dist = int(H * upsampling_factor), int(W * upsampling_factor), int(clear_dist * upsampling_factor)

    # for each sample in the batch generate unique grid positions
    xyz_grid = np.zeros((batch_size, num_particles, 3)).astype('int')
    for k in range(batch_size):
        
        # randomly choose num_particles linear indices
        lin_ind = np.random.randint(0, (W - clear_dist * 2) * (H - clear_dist * 2) * D,
                                    num_particles)

        # switch from linear indices to subscripts
        zgrid_vec, ygrid_vec, xgrid_vec = np.unravel_index(lin_ind, (
        D, H - 2 * clear_dist, W - 2 * clear_dist))

        # reshape subscripts to fit into the 3D on grid array
        xyz_grid[k, :, 0] = np.reshape(xgrid_vec, (1, num_particles), 'F') + clear_dist
        xyz_grid[k, :, 1] = np.reshape(ygrid_vec, (1, num_particles), 'F') + clear_dist
        xyz_grid[k, :, 2] = np.reshape(zgrid_vec, (1, num_particles), 'F')

    # for each grid position add a continuous shift inside the voxel within [-0.5,0.5)
    x_local = np.random.uniform(-0.49, 0.49, (batch_size, num_particles))
    y_local = np.random.uniform(-0.49, 0.49, (batch_size, num_particles))
    z_local = np.random.uniform(-0.49, 0.49, (batch_size, num_particles))
    
    # minimal height in z and axial pixel size
    zmin, pixel_size_axial = setup_params['zmin'], setup_params['pixel_size_axial']

    # group samples into an array of size [batch_size, emitters, xyz]
    xyz = np.zeros((batch_size, num_particles, 3))
    xyz[:, :, 0] = (xyz_grid[:, :, 0] - int(np.floor(W / 2)) + x_local + 0.5) * pixel_size_rec
    xyz[:, :, 1] = (xyz_grid[:, :, 1] - int(np.floor(H / 2)) + y_local + 0.5) * pixel_size_rec
    xyz[:, :, 2] = (xyz_grid[:, :, 2] + z_local + 0.5) * pixel_size_axial + zmin

    # resulting batch of data
    return xyz, Nphotons


# ======================================================================================================================
# applying a global shift or an affine transform to given xyz
# ======================================================================================================================


# apply affine transform to given xyz batch
def affine_transform_xyz(xyz, transform_matrix):

    # number of samples in the batch and the number of emitters
    batch_size, num_particles, _ = xyz.shape

    # create the ones column
    ones_col = np.ones((num_particles,))

    # for each sample create the xy1 matrix and transform
    xyz_affine = np.zeros((batch_size, num_particles, 3))
    for i in range(batch_size):

        # current xyz
        curr_xyz = xyz[i, :, :]

        # create the xy1 matrix
        xy1 = np.transpose(np.column_stack((curr_xyz[:,0],curr_xyz[:,1],ones_col)))

        # transform xy
        xy1_affine = np.transpose(np.matmul(transform_matrix, xy1))

        # recompose the current xyz
        new_xyz = np.column_stack((xy1_affine[:,0], xy1_affine[:,1], curr_xyz[:, 2]))

        # save sample new coords
        xyz_affine[i, :, :] = new_xyz

    # return the result
    return xyz_affine


# apply global shift to given xyz batch and optionally introduces xy registration error
def global_shift_xyz(xyz, shifts, reg_err=None):

    # start from current coords (this makes a copy, doesn't point to input xyz)
    xyz_shift = np.copy(xyz)

    # shift globally in xy
    xyz_shift[:, :, 0] += shifts[0]
    xyz_shift[:, :, 1] += shifts[1]

    # add a registration error if specified
    if reg_err is not None:

        # number of samples in the batch and the number of emitters
        batch_size, num_particles, _ = xyz.shape

        # random perturbation in x and y
        xyz_shift[:, :, 0] += np.random.rand(batch_size, num_particles) * reg_err * 2 - reg_err
        xyz_shift[:, :, 1] += np.random.rand(batch_size, num_particles) * reg_err * 2 - reg_err

    # return the shifted coords
    return xyz_shift


# ======================================================================================================================
# projection of the continuous positions on the recovery grid in order to generate the training label
# ======================================================================================================================


# converts continuous xyz locations to a boolean grid
def batch_xyz_to_boolean_grid(xyz_np, setup_params):
    
    # calculate upsampling factor
    pixel_size_FOV, pixel_size_rec = setup_params['pixel_size_FOV'], setup_params['pixel_size_rec']
    upsampling_factor = pixel_size_FOV / pixel_size_rec
    
    # axial pixel size
    pixel_size_axial = setup_params['pixel_size_axial']

    # current dimensions
    H, W, D = setup_params['H'], setup_params['W'], setup_params['D']
    
    # shift the z axis back to 0
    zshift = xyz_np[:, :, 2] - setup_params['zmin']
    
    # number of particles
    batch_size, num_particles = zshift.shape
    
    # project xyz locations on the grid and shift xy to the upper left corner
    xg = (np.floor(xyz_np[:, :, 0]/pixel_size_rec) + np.floor(W/2)*upsampling_factor).astype('int')
    yg = (np.floor(xyz_np[:, :, 1]/pixel_size_rec) + np.floor(H/2)*upsampling_factor).astype('int')
    zg = (np.floor(zshift/pixel_size_axial)).astype('int')
    
    # indices for sparse tensor
    indX, indY, indZ = (xg.flatten('F')).tolist(), (yg.flatten('F')).tolist(), (zg.flatten('F')).tolist()

    # update dimensions
    H, W = int(H * upsampling_factor), int(W * upsampling_factor)
    
    # if sampling a batch add a sample index
    if batch_size > 1:
        indS = (np.kron(np.ones(num_particles), np.arange(0, batch_size, 1)).astype('int')).tolist()
        ibool = torch.LongTensor([indS, indZ, indY, indX])
    else:
        ibool = torch.LongTensor([indZ, indY, indX])
    
    # spikes for sparse tensor
    vals = torch.ones(batch_size*num_particles)
    
    # resulting 3D boolean tensor
    if batch_size > 1:
        boolean_grid = torch.sparse_coo_tensor(ibool, vals, torch.Size([batch_size, D, H, W]), dtype=torch.float32).to_dense()
    else:
        boolean_grid = torch.sparse_coo_tensor(ibool, vals, torch.Size([D, H, W]), dtype=torch.float32).to_dense()
    
    return boolean_grid


# ======================================================================================================================
# dataset class instantiation for both pre-calculated images / training positions to accelerate data loading in training
# ======================================================================================================================


# PSF images with corresponding xyz labels dataset
class ImagesDataset(Dataset):
    
    # initialization of the dataset
    def __init__(self, root_dir, list_IDs, labels, setup_params):
        self.root_dir = root_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.setup_params = setup_params
    
    # total number of samples in the dataset
    def __len__(self):
        return len(self.list_IDs)
    
    # sampling one example from the data
    def __getitem__(self, index):
        
        # select sample
        ID = self.list_IDs[index]

        # load PSF1 image and add the channels dimension
        im_name1 = self.root_dir + '/im_psf1_' + ID + '.tiff'
        im_np1 = imread(im_name1)
        im_np1 = np.expand_dims(im_np1, 0)
        
        # load PSF2 image and add the channels dimension
        im_name2 = self.root_dir + '/im_psf2_' + ID + '.tiff'
        im_np2 = imread(im_name2)
        im_np2 = np.expand_dims(im_np2, 0)

        # concatenate the 2 images as channels and turn into torch tensor
        im_tensor = torch.from_numpy(np.concatenate((im_np1, im_np2), 0)).type(torch.FloatTensor)
        
        # corresponding xyz labels turned to a boolean tensor
        xyz_np = self.labels[ID]
        bool_grid = batch_xyz_to_boolean_grid(xyz_np, self.setup_params)
        
        return im_tensor, bool_grid


# xyz and photons turned online to fourier phases dataset
class PhasesOnlineDataset(Dataset):
    
    # initialization of the dataset
    def __init__(self, list_IDs, labels, setup_params):
        self.list_IDs = list_IDs
        self.labels = labels
        self.setup_params = setup_params
    
    # total number of samples in the dataset
    def __len__(self):
        return len(self.list_IDs)
    
    # sampling one example from the data
    def __getitem__(self, index):
        
        # select sample
        ID = self.list_IDs[index]
        
        # associated number of photons
        dict = self.labels[ID]
        Nphotons_np = dict['N']
        Nphotons = torch.from_numpy(Nphotons_np)
        
        # corresponding xyz labels turned to a boolean tensor
        xyz_np = dict['xyz']        
        bool_grid = batch_xyz_to_boolean_grid(xyz_np, self.setup_params)
        
        # calculate phases and turn them to tensors
        phases_np = EmittersToPhases(xyz_np, self.setup_params)
        phases_tensor = complex_to_tensor(phases_np)

        return phases_tensor, Nphotons, bool_grid


# Experimental images with normalization dataset
# IMPORTANT!! The order of the channels is reversed due to a previous bug in the training code
# TODO: Switch order to (im_psf1, im_psf2) for a new tracking experiement!
class ExpDataset(Dataset):

    # initialization of the dataset
    def __init__(self, im_list_psf1, im_list_psf2, setup_params, scale_test=False):
        self.im_list_psf1 = im_list_psf1
        self.im_list_psf2 = im_list_psf2
        assert len(self.im_list_psf1) == len(self.im_list_psf2), "Each measurement should have 2 psfs!"
        self.project_01 = setup_params['project_01']
        if self.project_01 is False:
            self.global_factors = setup_params['global_factors']
        self.train_stats = setup_params['train_stats']
        self.scale_test = scale_test
        
        # affine transforms from no mask to a mask in each camera and between cameras
        # a transform set to identity means no change in (homogeneous) coordinates
        self.tform_psf1 = AffineTransform(matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))  # in [px]
        self.tform_psf2 = AffineTransform(matrix=np.array([[1, 0, 2], [0, 1, 1], [0, 0, 1]]))  # in [px]
        # assumption is we estimated 1->2, and 2 have lower SNR
        self.tform_psf1to2 = AffineTransform(matrix=setup_params['Tpx'])  # in [px]

    # total number of samples in the dataset
    def __len__(self):
        return len(self.im_list_psf1)

    # sampling one example from the data
    def __getitem__(self, index):

        # load tif images of psf 1 and psf 2 and turn into float32
        im_psf1 = (imread(self.im_list_psf1[index])).astype("float32")
        im_psf2 = (imread(self.im_list_psf2[index])).astype("float32")

        # normalize image according to the training setting
        if self.project_01 is True:
            im_psf1 = normalize_01(im_psf1)
            im_psf2 = normalize_01(im_psf2)
        else:
            im_psf1 = (im_psf1 - self.global_factors[0]) / self.global_factors[1]
            im_psf2 = (im_psf2 - self.global_factors[0]) / self.global_factors[1]
            
        # if specified then align test statistics
        if self.project_01 is True and self.scale_test is True:
            im_psf1 = (im_psf1 - im_psf1.mean()) / im_psf1.std()
            im_psf1 = im_psf1 * self.train_stats[1][0] + self.train_stats[0][0]
            im_psf2 = (im_psf2 - im_psf2.mean()) / im_psf2.std()
            im_psf2 = im_psf2 * self.train_stats[1][1] + self.train_stats[0][1]
        
        # shift the images of psf 1 and 2 (due to the way the images were cropped)
        im_psf1 = warp(im_psf1, self.tform_psf1.inverse, order=3, preserve_range=False)
        im_psf2 = warp(im_psf2, self.tform_psf2.inverse, order=3, preserve_range=False)

        # warp the images of psf 1 to match the coordinates of psf 2
        im_psf1 = warp(im_psf1, self.tform_psf1to2.inverse, order=3, preserve_range=False)

        # turn images into torch tensor with 1 channel
        im_psf1 = np.expand_dims(im_psf1, 0)
        im_psf2 = np.expand_dims(im_psf2, 0)
        
        # concatenate results into a 2 channel tensor
        # IMPORTANT!! The order of the channels is reversed due to a previous bug in the training code
        # TODO: Switch order to (im_psf1, im_psf2) for a new tracking experiement!
        im_tensor = torch.from_numpy(np.concatenate((im_psf2, im_psf1), 0)).type(torch.FloatTensor)
        return im_tensor


# ======================================================================================================================
# Sorting function in case glob uploads images in the wrong order
# ======================================================================================================================

# ordering file names according to number
def sort_names_tif(img_names):
    nums = []
    for i in img_names:
        i2 = i.split(".tif")
        i3 = os.path.split(i2[0])
        nums.append(int(i3[-1]))
    indices = np.argsort(np.array(nums))
    fixed_names = [img_names[i] for i in indices]
    return fixed_names

