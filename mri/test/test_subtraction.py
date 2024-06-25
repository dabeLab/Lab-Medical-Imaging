from imaging.mri.utilities import *
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os

mri0 = sitk.ReadImage("E:/2021_local_data/2023_Gd_synthesis/dataset/2225172 T1wRC0.0.nii")
mri1 = sitk.ReadImage("E:/2021_local_data/2023_Gd_synthesis/dataset/2225172 T1wRC0.5.nii")

mri1_0 = sitk.Subtract(mri1, mri0)
check_contrast(mri0, mri1_0, [int(x // 2) for x in mri1_0.GetSize()], [10, 10, 5], 3)
plt.show()
