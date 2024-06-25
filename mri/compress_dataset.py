import os
import shutil
import nibabel as nib
import numpy as np

source_folder = 'E:\\gd_synthesis\\data_large'
destination_folder = 'E:\\gd_synthesis\\data_large_compressed'
# source_folder = 'E:\\gd_synthesis\\dataset'
# destination_folder = 'E:\\gd_synthesis\\dataset_compressed'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate over all files in the source folder
for file in os.listdir(source_folder):
    print(f'Working on file {file}..')  # Print the current file being processed

    # Define the expected output path for .nii.gz files
    new_file_path = os.path.join(destination_folder, file if not file.endswith('.nii') else file + '.gz')

    # Check if the output file already exists
    if not os.path.exists(new_file_path):
        file_path = os.path.join(source_folder, file)

        # Check if the file is a .nii file
        if file.endswith('.nii'):
            nii_image = nib.load(file_path)

            # Print the data type of the image
            print(f'Data type of {file}: {nii_image.get_data_dtype()}')

            # If the data type is float32, convert the data to float16
            if nii_image.get_data_dtype() == np.float32:
                data = nii_image.get_fdata().astype(np.float16)
                nii_image = nib.Nifti1Image(data, nii_image.affine, nii_image.header)
                print(f'Converting {file} to float16.')

            data = nii_image.get_fdata()
            # Calculate new indices to crop 10% of each dimension
            crop_percent = 0.10
            start_indices = (np.array(data.shape) * crop_percent).astype(int)
            end_indices = (np.array(data.shape) * (1 - crop_percent)).astype(int)
            cropped_data = data[start_indices[0]:end_indices[0],
                           start_indices[1]:end_indices[1],
                           start_indices[2]:end_indices[2]]

            # Create a new NIfTI image with the cropped data
            cropped_img = nib.Nifti1Image(cropped_data, nii_image.affine)

            nib.save(cropped_img, new_file_path)
        else:
            # For all other files, just copy them to the new destination
            shutil.copy2(file_path, new_file_path)
    else:
        print(f'Skipping {file} as it already exists.')

print("Operation completed successfully.")
