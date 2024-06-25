import SimpleITK as sitk
import numpy as np
import os

def z_score_normalize(img_list):
    # img_list is a list of path to MR images
    # Initialize variables for accumulating statistics
    total_pixels = 0
    cumulative_sum = 0
    cumulative_sum_squared = 0
    # Process images one by one to calculate mean and standard deviation
    for img in img_list:
        data = sitk.GetArrayFromImage(img)
        total_pixels += data.size
        cumulative_sum += np.sum(data)
        cumulative_sum_squared += np.sum(data ** 2)
    # Calculate mean and standard deviation across the entire sample
    sample_mean = cumulative_sum / total_pixels
    sample_std = np.sqrt((cumulative_sum_squared / total_pixels) - (sample_mean ** 2))
    # Z-score normalize each image
    for img in img_list:
        data = (sitk.GetArrayFromImage(img) - sample_mean) / sample_std
        img_sitk = sitk.GetImageFromArray(data)
        img_sitk.CopyInformation(img)
        # Save the normalized image to disk
        sitk.WriteImage(img_sitk, os.path.join(output_dir, f"normalized_image_{i}.nii.gz"))
