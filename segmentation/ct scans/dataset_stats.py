import numpy as np
import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from skimage.color import label2rgb
from skimage.measure import label, regionprops, marching_cubes, mesh_surface_area
from skimage.exposure import rescale_intensity
from skimage.morphology import cube, ball, remove_small_objects, binary_opening, binary_closing
import glob
import nibabel as nib

# mm^3. We consider worms 0.1x smaller and 10x larger than the average expected size
mealworm_expected_dim = np.pi * (1.5e-3)**2 * 10e-3
characteristic_dimension = mealworm_expected_dim * np.array([0.1, 1, 10])
print(characteristic_dimension)
# Map spatial units to human-readable labels
unit_mapping = {0: None, 1: 1, 2: 1e-3, 3: 1e-6}
os.chdir("/Users/berri/Desktop/ct")
dataset = ["", 64, 128, 256]
data = {"dataset": {"volume": [], "intensity": [], "sphericity": []},
        "dataset64": {"volume": [], "intensity": [], "sphericity": []},
        "dataset128": {"volume": [], "intensity": [], "sphericity": []},
        "dataset256": {"volume": [], "intensity": [], "sphericity": []}
        }


def sphericity(segmented_volume, mask):

    return sphericity


for val in dataset:
    print(f"processing dataset{val}... ", end="")
    path_img = sorted(glob.glob(os.path.join(f"dataset{val}", "img*.nii.gz")))
    path_sgm = sorted(glob.glob(os.path.join(f"dataset{val}", "sgm*.nii.gz")))
    path_dic = [{"img": img, "sgm": sgm} for img, sgm in zip(path_img, path_sgm)]
    for paths in path_dic:
        nifti_img = nib.load(paths["img"])
        nifti_sgm = nib.load(paths["sgm"])
        img = nifti_img.get_fdata()
        sgm = np.round(nifti_sgm.get_fdata())
        dx, dy, dz = nifti_img.header.get_zooms()
        units = unit_mapping[nifti_img.header['xyzt_units']&0x07]
        if units is None:
            units = 1e-3
        voxel_dimension = units * dx * units * dy * units * dz
        lbl, n = label(sgm, return_num=True)
        for idx, region in enumerate(regionprops(label_image=lbl, intensity_image=img, cache=True)):
            data[f"dataset{val}"]["volume"].append(region.area * voxel_dimension)
            data[f"dataset{val}"]["intensity"].append(region.intensity_mean)
            # if not characteristic_dimension[0] <= region.area * voxel_dimension <= characteristic_dimension[2]:
            #     print(f"{idx+1}/{n} -> outlier")
            # else:
            #     print(f"{idx+1}/{n} -> ok")
            #     ax = plt.figure().add_subplot(projection="3d")
            #     ax.voxels(region.image, alpha=0.5, edgecolor="k")
            #     plt.show()
    print("Done.")

fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(np.log10(np.array(data["dataset"]["volume"])), bins=100, label="dataset")
axs[0, 1].hist(np.log10(np.array(data["dataset64"]["volume"])), bins=100, label="dataset64")
axs[1, 0].hist(np.log10(np.array(data["dataset128"]["volume"])), bins=100, label="dataset128")
axs[1, 1].hist(np.log10(np.array(data["dataset256"]["volume"])), bins=100, label="dataset256")
for ax in axs.flatten():
    ax.set_xlabel(r"Log10(Volume)")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.axvline(np.log10(characteristic_dimension[0]), linestyle="-", color="red")
    ax.axvline(np.log10(characteristic_dimension[1]), linestyle="-.", color="black")
    ax.axvline(np.log10(characteristic_dimension[2]), linestyle="-", color="red")
    ax.set_xlim(-10, -4)
plt.tight_layout()

fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(np.array(data["dataset"]["intensity"]), bins=100, label="dataset")
axs[0, 1].hist(np.array(data["dataset64"]["intensity"]), bins=100, label="dataset64")
axs[1, 0].hist(np.array(data["dataset128"]["intensity"]), bins=100, label="dataset128")
axs[1, 1].hist(np.array(data["dataset256"]["intensity"]), bins=100, label="dataset256")
for ax in axs.flatten():
    ax.set_xlabel(r"Intensity")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.set_xlim(-600, -100)
plt.tight_layout()
plt.show()

    #sgm = binary_opening(sgm, ball(1))
    #sgm = remove_small_objects(sgm, min_size=25)
    # lbl2, n2 = label(sgm, return_num=True, connectivity=2)
    # surface_coord = []
    # volume = []
    # for region in regionprops(lbl2):
    #     # Iterate through each pixel in the region
    #     for coord in region.coords:
    #         x, y, z = coord
    #         # Check if the pixel has at least one neighboring pixel that is not part of the same region
    #         for i in range(-1, 2):
    #             for j in range(-1, 2):
    #                 for k in range(-1, 2):
    #                     x_, y_, z_ = x + i, y + j, z + k
    #                     # Check if the neighbor is within bounds
    #                     if 0 <= x_ < lbl2.shape[0] and 0 <= y_ < lbl2.shape[1] and 0 <= z_ < lbl2.shape[2]:
    #                         # Check if the neighbor has a different label
    #                         if lbl2[x_, y_, z_] != lbl2[x, y, z]:
    #                             surface_coord.append((x, y, z))
    #                             break  # Break the inner loop if a surface voxel is found
    #     # Set surface coordinates to background
    # for (x, y, z) in surface_coord:
    #     lbl2[x, y, z] = 0
    #
    # lbl2, n2 = label(lbl2, return_num=True)
    # lbl2 = remove_small_objects(lbl2, min_size=100, connectivity=1)
    # lbl2, n2 = label(lbl2, return_num=True, connectivity=2)
    # print(n1, n2, area1, area2)

# # Generate a sample 3D volume
# volume = np.zeros((10, 10, 10), dtype=np.uint8)
# volume[3:7, 3:7, 3:7] = 1
#
# # Display the original volume
# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
# ax.voxels(volume, edgecolor='k')
# ax.set_title('Original Volume')
#
# # Identify the surface using binary_erosion
# surface = volume - ndimage.binary_erosion(volume, iterations=1)
#
# # Set surface pixels to zero
# volume[surface > 0] = 0
#
# # Display the volume with surface pixels set to zero
# ax = fig.add_subplot(122, projection='3d')
# ax.voxels(volume, edgecolor='k')
# ax.set_title('Volume with Surface Pixels Set to Zero')
#
# plt.show()
