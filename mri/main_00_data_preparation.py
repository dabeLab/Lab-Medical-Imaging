import pandas as pd
from imaging.mri.utilities import *
from class_BrainRegisterer import BrainRegisterer

# Select patient ID to process
patient_id = 2225172

# Load data (brain atlas, database...)
atlas = sitk.ReadImage("E:/2021_local_data/2023_Gd_synthesis/atlas/Johnson et al 2019 Canine Atlas v2/Canine_population_template.nii.gz")
df = pd.read_csv("E:/2021_local_data/2023_Gd_synthesis/DICOM/database filtered.csv", sep=";", index_col=False)

# Register pre-contrast image
path = df[(df["patient id"] == patient_id) & (df["contrast dose"] == 0)]["series directory"].values[0]
mri_0t1w = read_dicom_series(path)
path = df[(df["patient id"] == patient_id) & (df["contrast dose"] == 1 / 2)]["series directory"].values[0]
mri_05t1w = read_dicom_series(path)
path = df[(df["patient id"] == patient_id) & (df["contrast dose"] == 1)]["series directory"].values[0]
mri_1t1w = read_dicom_series(path)

brain_registerer = BrainRegisterer(mri_0t1w, mri_05t1w, mri_1t1w, atlas, patient_id)
brain_registerer.execute()





