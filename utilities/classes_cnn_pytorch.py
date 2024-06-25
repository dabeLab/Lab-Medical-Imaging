import torch.nn as nn
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from torchvision import transforms

class MealwormsCTDataset(Dataset):
    # dataset is a list of dictionary tuples ("image path", "labels path")
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_head, img_data, lbl_head, lbl_data = self.load_nifti(idx)
        if self.transform:
            image = self.transform(img_data)
            label = self.transform(lbl_data)
        image = image.unsqueeze(0) # add a dimension corresponding to the numbe of channels
        label = label.unsqueeze(0) # add a dimension corresponding to the numbe of channels
        return image, label

    def load_nifti(self, idx):
        sample = self.data[idx]
        path_img = sample['image']
        path_lbl = sample['label']
        nifti_img = nib.load(path_img)
        nifti_lbl = nib.load(path_lbl)
        nifti_img_data = nifti_img.get_fdata()
        nifti_img_head = nifti_img.header
        nifti_lbl_data = nifti_lbl.get_fdata()
        nifti_lbl_head = nifti_lbl.header
        return nifti_img_head, nifti_img_data, nifti_lbl_head, nifti_lbl_data

# Define your CNN architecture by creating a class that inherits from nn.Module.
#This class should include the layers and operations you want in your CNN.
def conv_block(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=3, padding=1, dtype=torch.float64),
        nn.ReLU(),
        nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1, dtype=torch.float64),
        nn.ReLU(),
    )

def encoder_block(in_channel, out_channels):
    return nn.Sequential(
        conv_block(in_channel, out_channels),
        nn.MaxPool3d(kernel_size=2, stride=2),
    )

def decoder_block(in_channels, out_channels, connected_features):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2),
        nn.Conv2d(in_channels + connected_features.size(1), out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )

class FeatureCountingCNN(nn.Module):
    def __init__(self):
        super(FeatureCountingCNN, self).__init__()
        # encoder
        # input: 200x200x1
        self.enc1a = nn.Conv3d(1, 64, kernel_size=3, padding=1, dtype=torch.float64)     # output:
        self.enc1b = nn.Conv3d(64, 64, kernel_size=3, padding=1, dtype=torch.float64)
        self.enc1c = nn.MaxPool3d(kernel_size=2, stride=2)          # output: 100x100x64
        # input: 100x100x64
        self.enc2a = nn.Conv3d(64, 128, kernel_size=3, padding=1, dtype=torch.float64)   # output:
        self.enc2b = nn.Conv3d(128, 128, kernel_size=3, padding=1, dtype=torch.float64)
        self.enc2c = nn.MaxPool3d(kernel_size=2, stride=2)          # output: 50x50x128
        # input: 50x50x128
        self.enc3a = nn.Conv3d(128, 256, kernel_size=3, padding=1, dtype=torch.float64)   # output:
        self.enc3b = nn.Conv3d(256, 256, kernel_size=3, padding=1, dtype=torch.float64)
        self.enc3c = nn.MaxPool3d(kernel_size=2, stride=2)          # output: 24x24x256
        # Bottleneck or Baseline
        # input: 24x24x256
        self.enc4a = nn.Conv3d(256, 516, kernel_size=3, padding=1, dtype=torch.float64)   # output:
        self.enc4b = nn.Conv3d(512, 512, kernel_size=3, padding=1, dtype=torch.float64)
        # Decoder
        # input: 24x24x256 (516 with the connected features)
        self.dec1a = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2)
        self.dec1b = nn.Conv3d(512, 256, kernel_size=3, padding=1, dtype=torch.float64)
        self.dec1c = nn.Conv3d(256, 256, kernel_size=3, padding=1, dtype=torch.float64)
        # input: 50x50x128 (256 with the connected features)
        self.dec2a = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2)
        self.dec2b = nn.Conv3d(256, 128, kernel_size=3, padding=1, dtype=torch.float64)
        self.dec2c = nn.Conv3d(128, 128, kernel_size=3, padding=1, dtype=torch.float64)
        # input: 100x100x64 (128 with the connected features)
        self.dec3a = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2)
        self.dec3b = nn.Conv3d(128, 64, kernel_size=3, padding=1, dtype=torch.float64)
        self.dec3c = nn.Conv3d(64, 64, kernel_size=3, padding=1, dtype=torch.float64) # output: 200x200x64
        # Output
        # input: 200x200x64
        self.outpa = nn.Conv3d(64, 1, kernel_size=3, padding=1, dtype=torch.float64)

    def forward(self, x):
        e1 = nn.ReLU()(self.enc1a(x))
        e1 = nn.ReLU()(self.enc1b(e1))
        p1 = self.enc1c(e1) 

        e2 = nn.ReLU()(self.enc2a(p1))
        e2 = nn.ReLU()(self.enc2b(e2))
        p2 = self.enc2c(e2)

        e3 = nn.ReLU()(self.enc3a(p2))
        e3 = nn.ReLU()(self.enc3b(e3))
        p3 = self.enc3c(e3)

        e4 = nn.ReLU()(self.enc4a(p3))
        e4 = nn.ReLU()(self.enc4b(e4))

        d1 = self.dec1a(e4)
        d1 = torch.cat((d1, p3), dim=1)
        d1 = nn.ReLU()(self.dec1b(d1))
        d1 = nn.ReLU()(self.dec1c(d1))

        d2 = self.dec2a(d1)
        d2 = torch.cat((d2, p2), dim=1)
        d2 = nn.ReLU()(self.dec2b(d2))
        d2 = nn.ReLU()(self.dec2c(d2))

        d3 = self.dec3a(d2)
        d3 = torch.cat((d3, p1), dim=1)
        d3 = nn.ReLU()(self.dec3b(d3))
        d3 = nn.ReLU()(self.dec3c(d3))

        out = self.outpa(d3)
        return out

    
    # def conv_block(in_channels, out_channels):
    #     # this block repeats both in the encoder and decoder
    #     return nn.Sequential(
    #         nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #     )
    #
    # def encoder_block(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         self.conv_block(in_channels, out_channels),
    #         nn.MaxPool3d(kernel_size=2, stride=2),
    #     )
    #
    # def decoder_block(self, in_channels, out_channels, skip_features):
    #     return nn.Sequential(
    #         nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2),
    #         torch.cat((, skip_features), dim=1),
    #         self.conv_block(in_channels, out_channels)
    #     )
