from imaging.mri.utilities import *

msk0 = sitk.ReadImage("E:/2021_local_data/2023_Gd_synthesis/dataset/2413454 T1wRC0.0.msk.nii")
msk1 = sitk.ReadImage("E:/2021_local_data/2023_Gd_synthesis/dataset/2413454 T1wRC0.5.msk.nii")
msk2 = sitk.ReadImage("E:/2021_local_data/2023_Gd_synthesis/dataset/2413454 T1wRC1.0.msk.nii")

check_registration(msk0, msk1, None, [int(x // 2) for x in msk0.GetSize()], [10, 10, 5], 3)
plt.show()

cm0 = get_center_of_mass(msk0)
cm1 = get_center_of_mass(msk1)

msk1 = transform_translate_rigid(msk1, (cm1-cm0).tolist())
check_registration(msk0, msk1, None, [int(x // 2) for x in msk0.GetSize()], [10, 10, 5], 3)
plt.show()


