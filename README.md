# DeepNebulae

This code accompanies the paper: ["Learning optimal wavefront shaping for
multi-channel imaging"](https://tomer.net.technion.ac.il/files/2021/07/PAMI_Camera_Ready_Final_Version.pdf) which was presented at the IEEE 13th International Conference on Computational Photography (ICCP 21), and selected for publication in a special issue of IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).

![](Figures/ICCP.gif "This is an experimental measurement of diffusing beads with a PSF writing the word ICCP 2021.")

# Contents

- [Overview](#overview)
- [Optical setup](#optical-setup)
- [Video recording](#video-recording)
- [Channel registration](#channel-registration)
- [Proposed PSFs](#proposed-psfs)
- [System requirements](#system-requirements)
- [Installation instructions](#installation-instructions)
- [Code structure](#code-structure)
- [Demo examples](#demo-examples)
- [Learning a localization model](#learning-a-localization-model)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

# Overview

This code implements two different applications of CNNs in dense 3D localization microscopy:
1. Learning a 3D localization CNN for a given fixed pair of PSFs.

![](Figures/Localization.gif "This movie shows the application of a trained localization CNN to two simulated frames with the learned PSF-pair.")


2. Learning an optimized pair of PSFs for high density localization via end-to-end optimization.


![](Figures/masklearninganimation.gif "This movie shows the phase masks (left) and the corresponding PSFs (right) being learned over training iterations. Note that the phase masks are initialized to zero modulation, meaning the standard microscope PSF.")


There's no need to download any dataset as the code itself generates the training and the test sets. Demo 1 illustrates how to train a localization model based on retreived phase masks and channel alignment, and demo 4 illustrates how the method can be used to learn optimzied phase masks. The remaining demos evaluate pre-trained models on both simulated and experimental data.

# Optical setup

The optical setup assumed in this repository is a 

# Video recording

To get a better grasp of the system and methods proposed in our work click the image below to watch the 12 min talk presented at the IEEE 13th International Conference on Computational Photography (ICCP 21).

[![DeepNebulae on YouTube](http://img.youtube.com/vi/SxISC2O4qRI/0.jpg)](http://www.youtube.com/watch?v=SxISC2O4qRI&t=8868s "ICCP Paper Talk")

# System requirements
* The software was tested on a *Linux* system with Ubuntu version 18.0, and a *Windows* system with Windows 10 Home.  
* Training and evaluation were run on a standard workstation equipped with 32 GB of memory, an Intel(R) Core(TM) i7 − 8700, 3.20 GHz CPU, and a NVidia GeForce Titan Xp GPU with 12 GB of video memory.

# Installation instructions
1. Download this repository as a zip file (or clone it using git).
2. Go to the downloaded directory and unzip it.
3. The [conda](https://docs.conda.io/en/latest/) environment for this project is given in `environment_<os>.yml` where `<os>` should be substituted with your operating system. For example, to replicate the environment on a linux system use the command: `conda env create -f environment_linux.yml` from within the downloaded directory.
This should take a couple of minutes.
4. After activation of the environment using: `conda activate deep-nebulae`, you're set to go!

# Code structure
 
* Data generation
    * `DeepNebulae/physics_utils.py` implements the forward physical model relying on Fourier optics.
    * `DeepNebulae/GeneratingTrainingExamples.py` generates the training examples (either images + 3D locations as in demo1 or only 3D locations + intensities as in demo4). The assumed physical setup parameters are given in the script `Demos/parameter_setting_demo1.py`. This script should be duplicated and altered according to the experimental setup as detailed in `Docs/demo1_documentation.pdf`.
    * `DeepNebulae/data_utils.py` implements the positions and photons sampling, and defines the dataset classes.
    * The folder `Mat_Files` includes phase masks needed to run the demos.
* CNN models and loss function
    * `DeepNebulae/cnn_utils.py` this script contains the two CNN models used in this work.
    * `DeepNebulae/loss_utils.py` implements the loss function, and an approximation of the Jaccard index.
* Training scripts
    * `DeepNebulae/Training_Localization_Model.py` this script trains a localization model based on the pre-calculated training and validation examples in `GeneratingTrainingExamples.py`. Here, the phase mask is assumed to be fixed (either off-the-shelf or learned), and we're only interested in a dense localization model.
    * `DeepNebulae/PSF_Learning.py` this script learns an optimized PSF. The training examples in this case are only simulated 3D locations and intensities.
* Post-processing and evaluation
    * `DeepNebulae/postprocessing_utils.py` implements 3D local maxima finding and CoG estimation with a fixed spherical diameter on GPU using max-pooling.
    * `DeepNebulae/Testing_Localization_Model.py` evaluates a learned localization model on either simulated or experimental images. In demo2/demo5 this module is used with pre-trained weights to localize experimental data. In demo3 it is used to localize simulated data.
    * `DeepNebulae/assessment_utils.py` - this script contains a function that calculates the Jaccard index and the RMSE in both the axial and lateral dimensions given two sets of points in 3D.
* Visualization and saving/loading 
    * `DeepNebulae/vis_utils.py` includes plotting functions.
    * `DeepNebulae/helper_utils.py` includes saving/loading functions.
 
 # Channel registration


 # Demo examples
 
* There are 5 different demo scripts that demonstrate the use of this code:
    1. `demo1.py` - learns a CNN for localizing high-density Tetrapods under STORM conditions. The script simulates training examples before learning starts. It takes approximately 30 hours to train a model from scratch on a Titan Xp.
    2. `demo2.py` - evaluates a pre-trained CNN for localizing experimental Tetrapods (Fig. 3 main text). The script plots the input images with the localizations voerlaid as red crosses on top. The resulting localizations are saved in a csv file under the folder `Experimental_Data/Tetrapod_demo2/`. This demo takes about 1 minute on a Titan Xp.
    3. `demo3.py` - evaluates a pre-trained CNN for localizing simulated Tetrapods (Fig. 4 main text). The script plots the simulated input and the regenerated image, and also compares the recovery with the GT positons in 3D. This demo takes about 1 second on a Titan Xp.
    4. `demo4.py` - learns an optimized PSF from scratch. The learned phase mask and its corresponding PSF are plotted each 5 batches in the first 4 epochs, and afterwards only once each 50 batches. Learning takes approximately 30 hours to converge on a Titan Xp. 
    5. `demo5.py` - evaluates a pre-trained CNN for localizing an experimental snapshot of a U2OS cell nucleus with the learned PSF. The experimental image can be switched from 'frm1' to 'frm2' in `Experimental_Data/`. This demo takes about 1 second on a Titan Xp.

* The `Demos` folder includes the following:
    * `Results_Tetrapod_demo2` contains pre-trained model weights and training metrics needed to run demo2.
    * `Results_Tetrapod_demo3` contains  pre-trained model weights and training metrics needed to run demo3.
    * `Results_Learned_demo5` contains pre-trained model weights and training metrics needed to run demo5.
    * `parameter_setting_demo*` contains the specified setup parameters for each of the demos.

* The `Experimental_data` folder includes the following:
    * `Tetrapod_demo2` contains 50 experimental frames from our STORM experiment (Fig. 3 main text).
    * `Learned_demo5_frm*` contains two snapshots of a U2OS cell nucleus with the learned PSF.

# Learning a localization model

To learn a localization model for your setup, you need to supply calibrated phase masks and channel registration transforms (e.g. using beads on the coverslip), and generate a new parameter settings script similar to the ones in the `Demos` folder. The `Docs` folder includes the pdf file `demo1_documentation.pdf` with snapshots detailing the steps in `demo1.py` to ease the user introduction to DeepNebulae. Please go through these while trying `demo1.py` to get a grasp of how the code works.

# Citation

If you use this code for your research, please cite our paper:
```
@article{nehme2021learning,
  title={Learning optimal wavefront shaping for multi-channel imaging},
  author={Nehme, Elias and Ferdman, Boris and Weiss, Lucien E and Naor, Tal and Freedman, Daniel and Michaeli, Tomer and Shechtman, Yoav},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={43},
  number={7},
  pages={2179--2192},
  year={2021},
  publisher={IEEE}
}
```

# License
 
This project is covered under the [**MIT License**](https://github.com/EliasNehme/DeepNebulae/blob/master/LICENSE).

# Contact

To report any bugs, suggest improvements, or ask questions, please contact me at "seliasne@gmail.com"

