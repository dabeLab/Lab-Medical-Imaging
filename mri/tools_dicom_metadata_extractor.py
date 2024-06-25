"""
Generate patients database from DICOM file directories and subdirectories.
This code assumes that data are organized in main/patient/study/series.
Please note that all MR slices DICOM files include the same metadata. Metadata
may differ between series, therefore we want a database that groups data by patient, study, series.
"""

import os
import pydicom
import pandas as pd
import datetime
from imaging.mri.backup.utilities import cosines_to_patient

# Create an empty list -> to be converted into DataFrame
df = []
# Assuming main/patient/study/series/
# E:\2021_local_data\2023_Gd_synthesis\DICOM\DICOM1\IMediaExport\DICOM\PAT_0000\STD_0000\SER_0000\OBJ_0001
main = "E:/2021_local_data/2023_Gd_synthesis/DICOM"
# Philips MRI manager exports in groups
groups = [folder for folder in os.listdir(main) if folder.lower().startswith("dicom")]
for group in groups:
    # get list of patients
    patients = [folder for folder in os.listdir(os.path.join(main, group, "IMediaExport", "DICOM")) if folder.startswith("PAT")]
    for patient in patients:
        # get list of studies
        studies = [folder for folder in os.listdir(os.path.join(main, group, "IMediaExport", "DICOM", patient)) if folder.startswith("STD")]
        for study in studies:
            # get list of series
            series = [folder for folder in os.listdir(os.path.join(main, group, "IMediaExport", "DICOM", patient, study)) if folder.startswith("SER")]
            for serie in series:
                # get list of images
                imgs = [img for img in os.listdir(os.path.join(main, group, "IMediaExport", "DICOM", patient, study, serie, "OBJ_0001")) if img.startswith("IM")]
                # pick first image in series
                for n, img in enumerate(imgs):
                    if n > 0:
                        continue

                    metadata = pydicom.dcmread(os.path.join(main, group, "IMediaExport", "DICOM", patient, study, serie, "OBJ_0001", img), stop_before_pixels=True)
                    data = {"patient id": int(metadata.PatientID),
                            "patient date of birth": datetime.datetime.strptime(metadata.PatientBirthDate, "%Y%m%d"),
                            "patient age": metadata.PatientAge if "PatientAge" in metadata else None,
                            "patient gender": metadata.PatientSex,
                            "patient position": metadata.PatientPosition if "PatientPosition" in metadata else None,
                            "patient weight": metadata.PatientWeight if "PatientWeight" in metadata else None,

                            "modality": metadata.Modality,
                            "series description": metadata.SeriesDescription if "SeriesDescription" in metadata else None,
                            "study datetime": datetime.datetime.strftime(datetime.datetime.strptime(metadata.StudyDate + metadata.StudyTime, "%Y%m%d%H%M%S"), "%Y.%m.%d %H:%M:%S"),
                            "series datetime": datetime.datetime.strftime(datetime.datetime.strptime(metadata.SeriesDate + metadata.SeriesTime, "%Y%m%d%H%M%S.%f"), "%Y.%m.%d %H:%M:%S"),
                            "series directory": os.path.join(main, group, "IMediaExport", "DICOM", patient, study, serie, "OBJ_0001"),

                            "body part examined": metadata.BodyPartExamined if "BodyPartExamined" in metadata else None,

                            "pixel spacing row": float(metadata.PixelSpacing[0]) if "PixelSpacing" in metadata else None,
                            "pixel spacing column": float(metadata.PixelSpacing[1]) if "PixelSpacing" in metadata else None,
                            "slice spacing": float(metadata.SliceThickness) if "SliceThickness" in metadata else None,
                            "direction cosines": tuple([x for x in metadata.ImageOrientationPatient]) if "ImageOrientationPatient" in metadata else None,
                            "direction": cosines_to_patient([x for x in metadata.ImageOrientationPatient]) if "ImageOrientationPatient" in metadata else None,
                            "magnetic field strength": metadata.MagneticFieldStrength if "MagneticFieldStrength" in metadata else None,
                            "manufacturer model": (metadata.Manufacturer if "Manufacturer" in metadata else None, metadata.ManufacturerModelName if "ManufacturerModelName" in metadata else None),
                            }
                    data = pd.Series(data)
                    df.append(data)

df1 = pd.DataFrame(df)
df2 = pd.read_csv("E:/2021_local_data/2023_Gd_synthesis/DICOM/database species.csv", sep=";", header=0)
df2.rename(columns={'Clinical Number': 'patient id', "Animal Species": "patient species"}, inplace=True)
df2 = df2[["patient id", "patient species"]]
df = pd.merge(df1, df2, on="patient id", how='left')
# Move the last column to the third position
cols = list(df.columns)
cols = [cols[0]] + [cols[-1]] + cols[1:-1]
df = df[cols]
df.to_csv(os.path.join(main, "database.csv"), sep=";", index=False)
