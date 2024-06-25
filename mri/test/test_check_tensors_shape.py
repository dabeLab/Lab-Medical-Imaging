import glob
import os
import SimpleITK as sitk
import numpy as np

path_main = "E:\\gd_synthesis"

path_im1 = sorted(glob.glob(os.path.join(path_main, "dataset_bak", "*T1wRC0.5.nii")))
path_im2 = sorted(glob.glob(os.path.join(path_main, "dataset_bak", "*T1wRC1.0.nii")))
path_msk = sorted(glob.glob(os.path.join(path_main, "dataset_bak", "*T1wRC0.0.msk.nii")))
path_dic = [{"img1": img1, "img2": img2, "msk": msk} for img1, img2, msk in zip(path_im1, path_im2, path_msk)]

for val in path_dic:
    img1 = val["img1"], sitk.ReadImage(val["img1"]).GetSize()
    img2 = val["img2"], sitk.ReadImage(val["img2"]).GetSize()
    img3 = val["msk"], sitk.ReadImage(val["msk"]).GetSize()

    print(img1)
    print(img2)
    print(img3)
    if img1[1] == img2[1] == img3[1]:
        print("OK\n")
    else:
        print("MISMATCH\n")
