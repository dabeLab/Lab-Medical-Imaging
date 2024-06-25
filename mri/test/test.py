import monai
import torch.nn
import numpy as np
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, OrientationD, ScaleIntensityRanged, AsDiscrete

from monai.losses.ssim_loss import SSIMLoss
from monai.metrics.regression import SSIMMetric
from monai.metrics import PSNRMetric, MAEMetric, MSEMetric
from imaging.mri.class_BrainLearn import *


# 1. DATA PRE-PROCESSING

# STEP 1.1: skull stripping -> by HD BET in 3D slicer
# Skull stripping is performed on T1, 1T1 and 1/2T1 MR images. To the purpose of model training,
# save the pairs (skull, brain segment): the brain segment will be used to weight the registration and model training.

# STEP 1.2: image registration -> by SimpleElastix (SimpleITK) in Python.
# Image registration is performed on (T1, 1T1) and (T1, 1/2T1) pairs separately,
# where T1 is the fixed image (pre-contrast), and 1T1 and 1/2T1 are the moving images (post-contrast).

# STEP 1.3: intensity rescaling -> by z-score in Python
# This is done by calculating the intensity mean and standard deviation of the image sample.
# To avoid running out of memory, the algorithm process one image per time (one MR image - 0.5 GB)
# With z-score, image intensity is rescaled between -1 and +1

# STEP 1.4: image subtraction -> by dataloader/transformation?
# This could be implemented into a transformers, but it would require defining a class. As a first attempt,
# it would be easier to get the difference with numpy.

# 2. DATASET PREPARATION

# The training, validation and testing dataset must consist of contrast images plus their brain masks. The
# # latter is necessary to weight the loss-function. Additionally, each image should come with metadata
# describing the voxel dimension.

# AsDiscrete should not be necessary as the brain mask is a 0-1 tensor.
transform = AsDiscrete(argmax=True, to_onehot=None)
data = np.array([[0.0, 1.0, 0.5], [2.0, 3.0, 0.1]])
print(data)
print(transform(data))

# initialize object
unet = DeepLearningExperiment()
# all MR images come from the same scanner -> voxel size is the same for all images
unet.voxel = (1.0, 1.0, 1.0)
# MR intensity after z-score normalization is between -1 and +1
unet.intensity_min = -1
unet.intensity_min = +1
# define loss function (for training)
unet.loss_function = torch.nn.L1Loss()  # Mean absolute value error, or L1
unet.loss_function = torch.nn.MSELoss()  # Mean squared error, or L2
unet.loss_function = SSIMLoss(spatial_dims=3, data_range=1.0, kernel_type=KernelType.GAUSSIAN, win_size=11, kernel_sigma=1.5, k1=0.01, k2=0.03, reduction=LossReduction.MEAN)
# define metric function (for validation and testing)
unet.metric_function = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_type=KernelType.GAUSSIAN, win_size=11, kernel_sigma=1.5, k1=0.01, k2=0.03, reduction=MetricReduction.MEAN, get_not_nans=False)
unet.metric_function = PSNRMetric(max_val, reduction=MetricReduction.MEAN, get_not_nans=False)

unet.train()

