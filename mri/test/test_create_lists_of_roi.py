import SimpleITK as sitk
import glob
import os
import numpy as np
import csv

files = sorted(glob.glob(os.path.join("E:/gd_synthesis/dataset", "**msk.nii")))
for file in files:
    msk = sitk.GetArrayFromImage(sitk.ReadImage(file))
    # Calculate the bounding box coordinates by finding the minimum and maximum indices along each dimension
    nonzero_indices = np.nonzero(msk)
    min_coord = np.min(nonzero_indices, axis=1)
    max_coord = np.max(nonzero_indices, axis=1)
    # Calculate the center of the bounding box
    bbox_c = (min_coord + max_coord) // 2
    # Calculate the size of the bounding box after rescaling
    bbox_s = (max_coord - min_coord)
    # Define the filename for saving the bounding box information
    output_file = f"{file[:-8]}.info.txt"
    # Write the bounding box information to the text file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["bbox center x", "bbox center y", "bbox center z", "bbox size x", "bbox size y", "bbox size z"])
        writer.writerow([bbox_c[0], bbox_c[1], bbox_c[2], bbox_s[0], bbox_s[1], bbox_s[2]])
