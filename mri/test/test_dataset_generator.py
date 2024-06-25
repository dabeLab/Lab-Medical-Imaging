from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, OrientationD, SaveImaged,
                              RandCropByLabelClassesd, Spacingd, ScaleIntensityRanged)
from monai.data import CacheDataset, DataLoader
from imaging.utilities.classes_cnn import DeepLearningExperiment

# Note: for this study:
# - ground truth: the CMR obtained from full-dose MR images subtraction
# - training: the CMR obtained from half-dose MR images subtraction
# Training dataset: the whole MR image. We do not want to randomly crop the MR as this could lead to artifacts
cnn = DeepLearningExperiment()  # define new object instance
cnn.n_classes = 2

# Define transforms for loading and preprocessing the NIfTI files.
transforms = Compose([
    LoadImaged(
        keys=["image", "label"]),
    EnsureChannelFirstd(
        keys=["image", "label"]),
    Spacingd(
        keys=["image", "label"],
        pixdim=(dx, dy, dz),
        mode=("bilinear", "nearest")),
    OrientationD(
        keys=["image", "label"],
        axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=cnn.intensity_min,
        a_max=cnn.intensity_max,
        b_min=0,
        b_max=1,
        clip=True),
    RandCropByLabelClassesd(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(64, 64, 64),
        ratios=params["class_ratio"],
        num_classes=params["n_classes"],
        num_samples=params["n_crops"],
        image_threshold=0),
    SaveImaged(
        keys=["image", "label"],
        output_dir=path_out,
        separate_folder=False,
        output_postfix=""),
    #ToTensor(),  # Convert the cropped image to a PyTorch tensor
])

dataset = CacheDataset(data=path_dict, transform=transforms)
loader = DataLoader(dataset)
data = {"image": path_dataraw, "label": path_label}
dataset.transform(data)


