import os
import csv
import glob
import random
import numpy as np
import torch
import nibabel as nib
import tqdm
import datetime
import SimpleITK as sitk
from utilities import closest_divisible_by_power_of_two, extract_axial_section, extract_coronal_section, extract_sagittal_section
import monai.config
from monai.networks.nets import UNet, UNETR
from monai.transforms import Compose, LoadImageD, EnsureChannelFirstD, OrientationD, ScaleIntensityRanged, \
    AsDiscreteD, AsDiscrete, Spacingd, SpatialCropD, Transform, LambdaD, ToTensorD, RandCropByPosNegLabelD, SaveImage
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference

from monai.losses import SSIMLoss, MultiScaleLoss
from monai.metrics import SSIMMetric, MultiScaleSSIMMetric

import matplotlib.pyplot as plt

class BrainLearn:
    """This class is a wrapper around MONAI. It handles dataset preparation, model training, validating
    and testing. It includes libraries for plotting (in dev.), and statistical analysis (in dev.). """
    def __init__(self):
        # The class assumes by default data are stored in main folder -> [data, dataset, model,...]
        # however these subdirectory can be changed by changing the corresponding protected attributes.
        # Note: main -> dataset includes training, validation and testing datasets
        self.path_main = None
        self._path_dataset = "dataset_compressed"
        self._path_data_raw = "raw"
        self._path_model = "model"
        self._path_results = "results"
        self.experiment = None
        # MODEL
        self.model = None
        self.n_classes = 1  # For image generation, the output classes are just one.
        self.optimizer = None
        self.max_iteration_trn = 100  # max iteration for training
        self.delta_iteration_trn = 1  # number of iterations in-between each validation step.
        self.delta_iteration_save = 1
        self.patch_size_trn = None  # Patch size for training dataset
        self.patch_size_val = None  # Patch size for validation dataset
        self.patch_size_tst = None  # Patch size for testing dataset
        self.roi_size = None
        self.patch_number: int or None = None
        self._epoch = None
        # TRANSFORMATIONS
        self.intensity_min = -1.0  # min voxel intensity value, used for intensity rescaling
        self.intensity_max = 1.0  # max voxel intensity value, used for intensity rescaling
        self.transforms_trn = None
        self.transforms_val = None
        self.transforms_tst = None
        # IMAGE METADATA
        self.voxel = None  # voxel dimensions (dx, dy, dz) in mm
        # DATA AND LOADERS
        self.data_percentage = 1
        self.dataset_trn_ratio = 0.7  # percentage of data used for training
        self.dataset_val_ratio = 0.2  # percentage of data used for validating
        self.dataset_tst_ratio = 0.1  # percentage of data used for testing
        self.batch_trn_size = 5  # training dataset batch size
        self.batch_val_size = 5  # validation dataset batch size
        self.batch_tst_size = 5  # testing dataset batch size
        self.dataset_trn: list or None = None  # list of dictionary for training the model (compiled by method)
        self.dataset_val: list or None = None  # list of dictionary for validating the model (compiled by method)
        self.dataset_tst: list or None = None  # list of dictionary for testing the model (compiled by method)
        self.loader_trn = None  # training data loader (compiled by method)
        self.loader_val = None  # validation data loader (compiled by method)
        self.loader_tst = None  # testing data loader (compiled by method)
        # MODEL PERFORMANCE INDEXES
        self.loss_function = None
        self.metric_function = None
        self.epochs = None
        self.losses = None
        self.scores = None
        # HARDWARE
        self.device = None
        # FIGURES
        self.plot_loss = None
        self.plot_score = None

        monai.config.print_config()

    # ******************* TRANSFORMATION METHODS *******************
    def compose_transforms_trn(self, augment=False):
        """Compose the transformation for the training dataset"""
        self.transforms_trn = Compose([
            LoadImageD(keys=["img1", "img2", "msk"]),
            EnsureChannelFirstD(keys=["img1", "img2", "msk"]),
            RandCropByPosNegLabelD(keys=["img1", "img2", "msk"], label_key="msk", spatial_size=self.patch_size_trn, neg=0.2, num_samples=self.patch_number),
            # self.CropImageBasedOnROI(img_keys=["img1", "img2"], roi_keys=["roi"], roi_size=self.patch_size),
            # Spacingd(keys=["img1", "img2", "msk"], pixdim=(1, 1, 1), mode=('bilinear', 'bilinear', 'nearest')),
            # ScaleIntensityRanged(keys=["img1", "img2"], a_min=self.intensity_min, a_max=self.intensity_max, b_min=0.0, b_max=1.0, clip=True),
            ToTensorD(keys=["img1", "img2", "msk"]),
        ])

    def compose_transforms_val(self):
        """compose the transformation for the validation dataset"""
        self.transforms_val = Compose([
            LoadImageD(keys=["img1", "img2", "msk"]),
            EnsureChannelFirstD(keys=["img1", "img2", "msk"]),
            # ScaleIntensityRanged(keys=["img1", "img2"], a_min=self.intensity_min, a_max=self.intensity_max, b_min=0.0, b_max=1.0, clip=True),
            ToTensorD(keys=["img1", "img2", "msk"]),
        ])

    def compose_transforms_tst(self):
        """compose the transformation for the testing dataset"""
        self.transforms_tst = Compose([
            LoadImageD(keys=["img1", "img2", "msk"]),
            EnsureChannelFirstD(keys=["img1", "img2", "msk"]),
            # ScaleIntensityRanged(keys=["img1", "img2"], a_min=self.intensity_min, a_max=self.intensity_max, b_min=0.0, b_max=1.0, clip=True),
            ToTensorD(keys=["img1", "img2", "msk"]),
        ])

    # ******************* DATA METHODS *******************
    def build_dataset(self, shuffle=True):
        """
        Build training, validation and testing datasets.
        Each sample is a dictionary {img T1wC0.5, img T1wC1.0, msk, roi}, where 'img T1wC0.5' is the path to the
        image with C0.5 contrast dose, 'img T1wC1.0' is the path to the image with C1.0 contrast dose,
        'msk' is the path to the brain mask, and 'roi' is the ROI used to crop the image for memory-efficient training.
        """

        # Create dictionary
        path_im1 = sorted(glob.glob(os.path.join(self.path_main, "dataset_compressed", "*T1wRC0.5.nii.gz")))
        path_im2 = sorted(glob.glob(os.path.join(self.path_main, "dataset_compressed", "*T1wRC1.0.nii.gz")))
        path_msk = sorted(glob.glob(os.path.join(self.path_main, "dataset_compressed", "*T1wRC0.0.msk.nii.gz")))
        # path_roi = sorted(glob.glob(os.path.join(self.path_main, "dataset_compressed", "*T1wRC0.0.info.txt")))
        # path_dic = [{"img1": img1, "img2": img2, "msk": msk, "roi": roi} for img1, img2, msk, roi in zip(path_im1, path_im2, path_msk, path_roi)]
        path_dic = [{"img1": img1, "img2": img2, "msk": msk} for img1, img2, msk in zip(path_im1, path_im2, path_msk)]
        # select subset of data
        n = int(len(path_dic) * self.data_percentage)
        n_tra = np.floor(self.dataset_trn_ratio * n).astype(int)
        n_val = np.floor(self.dataset_val_ratio * n).astype(int)
        n_tst = np.floor(self.dataset_tst_ratio * n).astype(int)
        # Shuffle the data list to randomize the order
        if shuffle is True:
            random.shuffle(path_dic)
        # Split the data into training, validation, and testing sets, and store paths in attributes
        self.dataset_trn = path_dic[:n_tra]
        self.dataset_val = path_dic[n_tra:n_tra + n_val]
        self.dataset_tst = path_dic[n_tra + n_val:n_tra + n_val + n_tst]
        # Print Dataset to screen
        print(f"Dataset trn: {self.dataset_trn}")
        print(f"Dataset val: {self.dataset_val}")
        print(f"Dataset tst: {self.dataset_tst}")

    def cache_dataset_trn(self):
        """cache training dataset and generate loader"""
        if self.dataset_trn:
            dataset_trn = CacheDataset(data=self.dataset_trn, transform=self.transforms_trn)
            self.loader_trn = DataLoader(dataset_trn, batch_size=self.batch_trn_size)

    def cache_dataset_val(self):
        """cache validation dataset and generate loader"""
        if self.dataset_val:
            dataset_val = CacheDataset(data=self.dataset_val, transform=self.transforms_val)
            self.loader_val = DataLoader(dataset_val, batch_size=self.batch_val_size)

    def cache_dataset_tst(self):
        """cache testing dataset and generate loader"""
        if self.dataset_tst:
            dataset_tst = CacheDataset(data=self.dataset_tst, transform=self.transforms_tst)
            self.loader_tst = DataLoader(dataset_tst, batch_size=self.batch_tst_size)

    # ******************* MODEL METHODS *******************
    def build_model_unet(self):  # weight decay of the Adam optimizer
        """Build a UNet model"""
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=self.n_classes,
            channels=(32, 64, 128, 256, 512),  # sequence of channels. Top block first. len(channels) >= 2
            strides=(2, 2, 2, 2),  # sequence of convolution strides. len(strides) = len(channels) - 1.
            kernel_size=3, # convolution kernel size, value(s) should be odd. If sequence, length = N. layers.
            up_kernel_size=3, # de-convolution kernel size, value(s) should be odd. If sequence, length = N. layers
            num_res_units=1,  # number of residual units. Defaults to 0.
            # act=params["activation_function"],
            dropout=0
        ).to(self.device)

    def train(self):
        """Train the model"""
        # generate the experiment name
        self.generate_experiment_name()
        # generate arrays for training and validating plotting purposes
        self.epochs = np.arange(self.max_iteration_trn)
        self.losses = np.zeros(self.max_iteration_trn)
        self.scores = np.zeros(self.max_iteration_trn)
        # Run training
        for epoch in range(self.max_iteration_trn):
            # Update epoch
            self._epoch = epoch
            # set the model to training. This has effect only on some transforms
            self.model.train()
            # initialize the epoch's loss
            epoch_loss = 0
            epoch_trn_iterator = tqdm.tqdm(self.loader_trn, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, miniters=1)
            for step_trn, batch_trn in enumerate(epoch_trn_iterator):
                # reset the optimizer (which stores the values from the previous iteration
                self.optimizer.zero_grad()
                # send the training data to device (GPU)
                inputs, targets, msk = batch_trn['img1'].to(self.device), batch_trn['img2'].to(self.device), batch_trn["msk"].to(self.device)
                # forward pass
                outputs = self.model(inputs)
                # calculate loss and add it to epoch's loss
                loss = self.loss_function(outputs * msk, targets * msk)
                # Weight the loss and get mean value
                epoch_loss += loss.mean().item()
                # backpropagation
                loss.backward()
                # update metrics
                self.optimizer.step()
                # Update the progress bar description with loss and metrics
                epoch_trn_iterator.set_description(f"Training ({self._epoch + 1} / {self.max_iteration_trn} Steps) (loss = {epoch_loss:2.5f})")
            # store epoch's loss in losses array
            self.losses[self._epoch] = epoch_loss
            # validate model every "delta_iteration"
            if self._epoch == 0 or (self._epoch + 1) % self.delta_iteration_trn == 0 or (self._epoch + 1) == self.max_iteration_trn:
                # run validation
                score = self.validate()
                # store validation metrics in metrics array
                self.scores[self._epoch] = score
            if self._epoch == 0 or (self._epoch + 1) % self.delta_iteration_save == 0 or (self._epoch + 1) == self.max_iteration_trn:
                # save model to disc
                self.save_model_dictionary()

    def validate(self):
        """Validate the model"""
        # Set validation metric to zero
        epoch_score = 0
        # Set the model to validation. This affects some transforms.
        self.model.eval()
        # Disable gradient computation (which is useless for validation)
        with torch.no_grad():
            epoch_val_iterator = tqdm.tqdm(self.loader_val, desc="Validate (X / X Steps) (metric = X.X)", dynamic_ncols=True, miniters=1)
            for step_val, batch_val in enumerate(epoch_val_iterator):
                # Send the validation data to device (GPU)
                img, tgt = batch_val["img1"].to(self.device), batch_val["img2"].to(self.device)
                # Run inference by forward passing windowed input data through the model
                prd = sliding_window_inference(img, roi_size=self.patch_size_val, sw_batch_size=1, predictor=self.model, progress=False)
                # Evaluate metric
                batch_score = self.metric_function(prd, tgt).mean().item()
                # Add batch's validation metric and then calculate average metric
                epoch_score += batch_score
                score_mean = epoch_score / (step_val + 1)
                # Update the progress bar description with metric
                epoch_val_iterator.set_description(f"Validate ({self._epoch + 1} / {self.max_iteration_trn} Steps) (metric = {score_mean:2.5f})")
        return score_mean

    def infer(self):
        """Make inference - WARNING: the saving overwrites all files in the batch"""
        # Set validation metric to zero
        epoch_score = 0
        # Set the model to validation. This affects some transforms.
        self.model.eval()
        # Disable gradient computation (which is useless for validation)
        with torch.no_grad():
            epoch_tst_iterator = tqdm.tqdm(self.loader_tst, desc="Testing (X / X Steps)", dynamic_ncols=True, miniters=1)
            for step_tst, batch_tst in enumerate(epoch_tst_iterator):
                # Send the testing data to device (GPU)
                img, tgt = batch_tst["img1"].to(self.device), batch_tst["img2"].to(self.device)
                # Run inference by forward passing windowed input data through the model
                prd = sliding_window_inference(img, roi_size=self.patch_size_tst, sw_batch_size=1, predictor=self.model)
                # Evaluate metric
                batch_score = self.metric_function(prd, tgt).mean().item()
                # Add batch's validation metric and then calculate average metric
                epoch_score += batch_score
                score_mean = epoch_score / (step_tst + 1)
                # Update the progress bar description with metric
                epoch_tst_iterator.set_description(f"Testing ({step_tst + 1} / {len(epoch_tst_iterator)} Steps)")
                # De-collate batch into samples and save images and information to disc
                samples = decollate_batch(prd)
                for sample in samples:
                    prd_array = sitk.GetImageFromArray(sample.detach().cpu().numpy().squeeze())
                    img_array = sitk.GetImageFromArray(img.detach().cpu().numpy().squeeze())
                    tgt_array = sitk.GetImageFromArray(tgt.detach().cpu().numpy().squeeze())
                    sitk.WriteImage(prd_array, os.path.join(self.path_main, self._path_results, f"{self.experiment} - tst.nii.gz"))
                    self.save_image_section_axial(prd_array, 100, "prd")
                    self.save_image_section_axial(tgt_array, 100, "tgt")

    def set_optimizer(self, optimizer="adam"):
        """Set the model optimizer."""
        if self.model is None:
            return print("Please build a model before setting the optimizer...")
        else:
            if optimizer == "adam":
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1E-3, weight_decay=1E-4)
            if optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1E-2, weight_decay=0)

    def set_loss_function(self, loss_function="l1"):
        """Set the loss function for training"""
        if loss_function == "l1":
            self.loss_function = torch.nn.L1Loss()
        if loss_function == "l2":
            self.loss_function = torch.nn.MSELoss()
        if loss_function == "ssim":
            self.loss_function = SSIMLoss(spatial_dims=3, )
        if loss_function == "mssim":
            self.loss_function = MultiScaleLoss(loss=SSIMLoss(spatial_dims=3,),
                                                scales=[1, 2, 4],
                                                kernel="gaussian",
                                                reduction=None)

    def set_metric_function(self, metric_function="l1"):
        """Set the metric function for validation and testing"""
        if metric_function == "l1":
            self.metric_function = torch.nn.L1Loss()
        if metric_function == "l2":
            self.metric_function = torch.nn.MSELoss()
        if metric_function == "ssim":
            self.metric_function = None
        if metric_function == "mssim":
            self.metric_function = None

    def set_device(self, device: str = "cpu"):
        """Set device to use"""
        if device.lower() == "cuda" and torch.cuda.is_available():
            var = "cuda"
        elif device.lower() == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            var = "mps"
        elif device.lower() == "cpu":
            var = "cpu"
        else:
            var = "cpu"
        self.device = torch.device(var)
        print(f"Running on {self.device}")

    # ******************* SAVE METHODS *******************
    def save_dataset_trn_paths_to_csv(self):
        """Save list of training data paths to csv."""
        with open(os.path.join(self.path_main, self._path_model, f"{self.experiment} - dataset trn.csv"), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Paths'])  # Writing header
            for val in self.dataset_trn:
                writer.writerow([val])

    def save_dataset_val_paths_to_csv(self):
        """Save list of training data paths to csv."""
        with open(os.path.join(self.path_main, self._path_model, f"{self.experiment} - dataset val.csv"), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Paths'])  # Writing header
            for val in self.dataset_val:
                writer.writerow([val])

    def save_dataset_tst_paths_to_csv(self):
        """Save list of training data paths to csv."""
        with open(os.path.join(self.path_main, self._path_model, f"{self.experiment} - dataset tst.csv"), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Paths'])  # Writing header
            for val in self.dataset_tst:
                writer.writerow([val])

    def save_model_dictionary(self):
        torch.save(self.model.state_dict(), os.path.join(self.path_main, self._path_model, f"{self.experiment} - iter {self._epoch + 1:03d} mdl.pth"))

    def save_model_attributes(self):
        # Get all attributes of the object
        attributes = vars(self.model)
        # Filter out protected attributes
        attributes = {k: v for k, v in attributes.items() if not k.startswith('_')}
        # Extract attribute names and values
        attribute_names = list(attributes.keys())
        attribute_values = list(attributes.values())
        # Write attributes to CSV
        with open(os.path.join(self.path_main, self._path_model, f"{self.experiment} - model attributes.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(attribute_names)
            writer.writerow(attribute_values)

    def save_loss(self):
        data = np.column_stack((self.epochs, self.losses))
        np.savetxt(os.path.join(self.path_main, self._path_model, f"{self.experiment} - losses.csv"), data, delimiter=',', header='x,y', comments='')
        plt.plot(data[:, 0], data[:, 1])
        plt.semilogy()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.path_main, self._path_model, f"{self.experiment} - losses.png"))
        plt.close()

    def save_score(self):
        data = np.column_stack((self.epochs, self.scores))
        np.savetxt(os.path.join(self.path_main, self._path_model, f"{self.experiment} - scores.csv"), data, delimiter=',', header='x,y', comments='')
        data = data[data[0:, 1] != 0]
        plt.plot(data[:, 0], data[:, 1])
        plt.semilogy()
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.savefig(os.path.join(self.path_main, self._path_model, f"{self.experiment} - scores.png"))
        plt.close()

    def save_image_section_axial(self, img: sitk.Image, n: int, text_note: str):
        """Extract an axial section at index n"""
        img_slice = sitk.Extract(img, [img.GetSize()[0], img.GetSize()[1], 0], [0, 0, n])
        img_extent = (0,
                      (0 + img_slice.GetSize()[0]) * img_slice.GetSpacing()[0],
                      (0 - img_slice.GetSize()[1]) * img_slice.GetSpacing()[1],
                      0)
        plt.figure()
        plt.imshow(sitk.GetArrayFromImage(img_slice), cmap="gray", extent=img_extent)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_main, self._path_results, f"{self.experiment} - slice xy {n} {text_note}.png"), dpi=600)

    def save_image_section_sagittal(self, img: sitk.Image, n: int, text_note: str):
        """Extract an axial section at index n"""
        img_slice = sitk.Extract(img, [0, img.GetSize()[1], img.GetSize()[2]], [n, 0, 0])
        img_extent = (0,
                      (0 + img_slice.GetSize()[0]) * img_slice.GetSpacing()[0],
                      (0 - img_slice.GetSize()[1]) * img_slice.GetSpacing()[1],
                      0)
        plt.figure()
        plt.imshow(sitk.GetArrayFromImage(img_slice), cmap="gray", extent=img_extent)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_main, self._path_results, f"{self.experiment} - slice yz {n} {text_note}.png"), dpi=600)

    def save_image_section_coronal(self, img: sitk.Image, n: int, text_note: str):
        """Extract an axial section at index n"""
        img_slice = sitk.Extract(img, [img.GetSize()[0], 0, img.GetSize()[2]], [0, n, 0])
        img_extent = (0,
                      (0 + img_slice.GetSize()[0]) * img_slice.GetSpacing()[0],
                      (0 - img_slice.GetSize()[1]) * img_slice.GetSpacing()[1],
                      0)
        plt.figure()
        plt.imshow(sitk.GetArrayFromImage(img_slice), cmap="gray", extent=img_extent)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_main, self._path_results, f"{self.experiment} - slice xz {n} {text_note}.png"), dpi=600)

    # ******************* LOAD METHODS *******************
    """Note: ALL 'load methods' require to have previously set the object 'experiment' attribute."""
    def load_model_dictionary(self, i=1000):
        """
        i: the model iteration number
        """
        # Load the state dictionary into the model
        self.model.load_state_dict(torch.load(os.path.join(self.path_main, self._path_model, f"{self.experiment} - iter {i} mdl.pth")))

    def load_dataset_trn_paths(self):
        dataset = []
        with open(os.path.join(self.path_main, self._path_model, f"{self.experiment} - dataset trn.csv"), 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                # Assuming each row contains a single string representing a dictionary
                data_str = row[0]  # Extract the string from the row
                # Remove surrounding double quotes and convert the string to a dictionary
                data_dict = eval(data_str.strip('"'))
                dataset.append(data_dict)
        self.dataset_trn = dataset

    def load_dataset_val_paths(self):
        dataset = []
        with open(os.path.join(self.path_main, self._path_model, f"{self.experiment} - dataset val.csv"), 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                # Assuming each row contains a single string representing a dictionary
                data_str = row[0]  # Extract the string from the row
                # Remove surrounding double quotes and convert the string to a dictionary
                data_dict = eval(data_str.strip('"'))
                dataset.append(data_dict)
        self.dataset_val = dataset

    def load_dataset_tst_paths(self):
        dataset = []
        with open(os.path.join(self.path_main, self._path_model, f"{self.experiment} - dataset tst.csv"), 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                # Assuming each row contains a single string representing a dictionary
                data_str = row[0]  # Extract the string from the row
                # Remove surrounding double quotes and convert the string to a dictionary
                data_dict = eval(data_str.strip('"'))
                dataset.append(data_dict)
        self.dataset_tst = dataset

    # ******************* UTILITY METHODS *******************
    def generate_experiment_name(self):
        """the experiment name is the datetime. Model info are saved in a text file"""
        self.experiment = datetime.datetime.now().strftime('%Y.%m.%d %H.%M.%S')

    def get_voxel_size(self):
        """extract voxel dimensions from first image in training dataset"""
        if not self.dataset_trn:
            print("Error: dataset_trn is empty. Build datasets first.")
            return
        self.voxel[0], self.voxel[1], self.voxel[2] = nib.load(self.dataset_trn[0]["img"]).header.get_zooms()

    # ******************* UTILITY METHODS - W.I.P. *******************
    def estimate_memory(self):
        """Estimate the memory required to train the model"""
        attributes = vars(self.model)
        if attributes["model"] == "UNet":
            print("wip")

    @staticmethod
    def _print_shape(data, prefix):
        print("\n")
        for key, val in data.items():
            print(f"{prefix} - {key}: {val.shape}")
        return data

    def roi_set_size_bak(self):
        """
        Find the volume of all brain masks in the dataset,
        and return an ROI which sides are the largest among all brain masks.
        """
        roi_max_size = np.array([0, 0, 0])
        for img, msk in {**self.dataset_trn, **self.dataset_val, **self.dataset_tst}:
            val = sitk.GetArrayViewFromImage(sitk.ReadImage(msk))
            nonzero_indices = np.nonzero(val)
            min_coord = np.min(nonzero_indices, axis=1)
            max_coord = np.max(nonzero_indices, axis=1)
            # Calculate the size of the bounding box
            roi_size = tuple(np.array(max_coord) - np.array(min_coord))
            # Calculate the center of the bounding box
            roi_center = tuple((np.array(min_coord) + np.array(max_coord)) / 2)
            print(img, roi_center, roi_size)
            # Update ROI size
            xyz_map = roi_size > roi_max_size
            roi_max_size[xyz_map] = roi_size[xyz_map]
        self.roi_size = roi_max_size.astype(int)

    def set_roi(self, n=5, x=1.0):
        """
        Get all ROI size from all information files in the dataset, and define the ROI size to use for
        training, which must include all possible ROIs and be divisible by 2^n, where n is the number
        of layers in the UNet model.
        """
        roi_size = np.zeros(3)
        for val in self.dataset_trn + self.dataset_val + self.dataset_tst:
            roi = val["roi"]
            # Read the bounding box information from the CSV file
            with open(roi, "r") as file:
                reader = csv.reader(file)
                # Skip the header row
                next(reader)
                # Read each row containing bounding box information
                for row in reader:
                    bbox_c_x, bbox_c_y, bbox_c_z, bbox_s_x, bbox_s_y, bbox_s_z = row
                    # Convert string values to integers
                    bbox_s = np.array([int(bbox_s_x), int(bbox_s_y), int(bbox_s_z)])
                    # Append bounding box information to the list
                    roi_size[bbox_s > roi_size] = bbox_s[bbox_s > roi_size]
        # Rescale the ROI
        roi_size = np.floor(roi_size * x)
        # Find the smallest ROI which is divisible by 2^n
        roi_size_x = closest_divisible_by_power_of_two(roi_size[0], n)
        roi_size_y = closest_divisible_by_power_of_two(roi_size[1], n)
        roi_size_z = closest_divisible_by_power_of_two(roi_size[2], n)
        self.roi_size = np.array([roi_size_x, roi_size_y, roi_size_z]).astype(int)

    class CropImageBasedOnROI(Transform):
        """Work in Progress"""
        def __init__(self, img_keys: list[str], roi_keys: list[str], roi_size):
            super().__init__()
            self.img_keys = img_keys
            self.roi_keys = roi_keys
            self.roi_size: np.array = roi_size
            self.roi_center: np.array or None = None

        def __call__(self, data):
            for roi_key in self.roi_keys:
                if roi_key in data:
                    roi_center = data[roi_key]
                    cropped_img = SpatialCropD(keys=self.img_keys, roi_center=roi_center, roi_size=self.roi_size)(data)
                    return cropped_img

    class LoadImageAndTextD(LoadImageD):
        """Work in Progress"""
        def __init__(self, image_keys: list[str], text_keys: list[str]):
            super().__init__(keys=image_keys)
            self.text_keys = text_keys

        def __call__(self, data):
            data = super().__call__(data)  # Call the parent class method to load image data

            for key in self.text_keys:
                if key in data:
                    text_path = data[key]
                    if os.path.exists(text_path) and os.path.isfile(text_path):
                        with open(text_path, 'r') as file:
                            csv_reader = csv.reader(file)
                            # Skip the header row
                            next(csv_reader)
                            # Read the second row and combine the first three elements into an array
                            row = next(csv_reader)
                            roi_center = np.array(row[:3]).astype(int)
                            data[key] = roi_center
                    else:
                        raise ValueError(f"Text file not found or is not a regular file: {text_path}")
            return data
