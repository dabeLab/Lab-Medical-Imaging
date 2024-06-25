from imaging.mri.utilities import *

msk0 = sitk.ReadImage("E:/2021_local_data/2023_Gd_synthesis/dataset/2225172 T1wRC0.0.msk.nii")
msk1 = sitk.ReadImage("E:/2021_local_data/2023_Gd_synthesis/dataset/2225172 T1wRC0.5.msk.nii")
mri0 = sitk.ReadImage("E:/2021_local_data/2023_Gd_synthesis/dataset/2225172 T1wRC0.0.nii")
mri1 = sitk.ReadImage("E:/2021_local_data/2023_Gd_synthesis/dataset/2225172 T1wRC0.5.nii")

# Get brain masks' center of mass
cm0 = get_center_of_mass(msk0)
cm1 = get_center_of_mass(msk1)
# Calculate transformation offset
offset10 = cm1 - cm0
# Align (transform) masks and MRI images
mri1 = transform_translate_rigid(mri1, offset10.tolist())
msk1 = transform_translate_rigid(msk1, offset10.tolist())

# Register MRI1 to MRI0
elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(mri0)
elastixImageFilter.SetMovingImage(mri1)
elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
elastixImageFilter.SetFixedMask(msk0)
elastixImageFilter.SetMovingMask(msk1)
elastixImageFilter.Execute()
mri1 = elastixImageFilter.GetResultImage()

# Transform msk1 to match the position of registered MRI1
parameters = elastixImageFilter.GetTransformParameterMap()
print(parameters)
sitk.PrintParameterMap(parameters)
msk1 = sitk.Transformix(msk1, parameters)

check_registration(msk0, msk1, None, [int(x // 2) for x in msk0.GetSize()], [10, 10, 5], 3)
plt.show()



