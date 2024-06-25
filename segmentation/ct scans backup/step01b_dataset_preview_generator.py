from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.data import CacheDataset, DataLoader
import glob
import os
import numpy as np
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define the paths to your 3D NIfTI CT scan and label files
path_dataset_trn = "C:\\Users\\dabe\\Desktop\\ct\\dataset128"
path_images = sorted(glob.glob(os.path.join(path_dataset_trn, "img*.nii.gz")))
path_labels = sorted(glob.glob(os.path.join(path_dataset_trn, "sgm*.nii.gz")))
path_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(path_images, path_labels)]

# Define transforms and DataLoader
transforms = Compose([
    #CropLabelledVolumed(keys=["image", "label"]),
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
]
)
dataset_trn = CacheDataset(data=path_dicts, transform=transforms)
loader_trn = DataLoader(dataset_trn, batch_size=1) # load one sample per time

# Load and process data
for step_trn, batch_trn in enumerate(loader_trn):
    print(f"Plotting sample {step_trn}...")
    # Find the (x,y,z) coordinates of the sample center
    x = int(np.floor(batch_trn["image"].shape[2] / 2))
    y = int(np.floor(batch_trn["image"].shape[3] / 2))
    z = int(np.floor(batch_trn["image"].shape[4] / 2))
    # Convert PyTorch GPU tensors to CPU arrays
    img_array = batch_trn["image"].cpu().numpy()[0, 0, :, :, :]
    lbl_array = batch_trn["label"].cpu().numpy()[0, 0, :, :, :]
    # Perform connected component analysis U
    lbl_array = label(lbl_array, connectivity=1)
    regions = regionprops(lbl_array)
    n_rgn = len(regions)
    print(f"N. regions ground truth: {n_rgn}")
    # Generate overlay images
    lbl_array_x = lbl_array[x, :, :]
    lbl_array_y = lbl_array[:, y, :]
    lbl_array_z = lbl_array[:, :, z]
    img_array_x = img_array[x, :, :]
    img_array_y = img_array[:, y, :]
    img_array_z = img_array[:, :, z]
    img_label_overlay_x = label2rgb(label=lbl_array_x, image=rescale_intensity(img_array_x, out_range=(0, 1)), alpha=0.3, bg_label=0, bg_color=None, kind="overlay", saturation=0.6)
    img_label_overlay_y = label2rgb(label=lbl_array_y, image=rescale_intensity(img_array_y, out_range=(0, 1)), alpha=0.3, bg_label=0, bg_color=None, kind="overlay", saturation=0.6)
    img_label_overlay_z = label2rgb(label=lbl_array_z, image=rescale_intensity(img_array_z, out_range=(0, 1)), alpha=0.3, bg_label=0, bg_color=None, kind="overlay", saturation=0.6)
    #blend = blend_images(image=batch_trn["image"], label=batch_trn["label"], alpha=0.3, cmap="gray", transparent_background=True)
    fig, axs = plt.subplots(2, ncols=3)
    axs[0, 0].set_title(f"YZ plane at X = {x} px")
    axs[0, 0].imshow(img_array_x, cmap="gray")
    axs[1, 0].imshow(img_label_overlay_x)
    axs[0, 1].set_title(f"XZ plane at Y = {y} px")
    axs[0, 1].imshow(img_array_y, cmap="gray")
    axs[1, 1].imshow(img_label_overlay_y)
    axs[0, 2].set_title(f"XY plane at Z = {z} px")
    axs[0, 2].imshow(img_array_z, cmap="gray")
    axs[1, 2].imshow(img_label_overlay_z)
    # Add rectangles around regions
    regions = regionprops(lbl_array_x)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.3)
        axs[1, 0].add_patch(rect)
    regions = regionprops(lbl_array_y)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.3)
        axs[1, 1].add_patch(rect)
    regions = regionprops(lbl_array_z)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.3)
        axs[1, 2].add_patch(rect)
    # Remove x-axis and y-axis ticks, labels, and tick marks for all subplots
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    # Adjust layout for better spacing
    plt.tight_layout()
    # Save the figure to a file
    plt.savefig(os.path.join(path_dataset_trn, f"plt_{step_trn}_xyz_{x,y,z}.png"), dpi=1200)
    plt.close()