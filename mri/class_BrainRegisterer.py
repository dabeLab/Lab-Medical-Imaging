import os.path

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from class_BrainAligner import BrainAligner
from utilities import *


class BrainRegisterer:
    """
    This class includes methods and attributes to register a brain atlas (moving image) to an MR image (fixed image).
    The registration it executed by running the 'execute()' method.
    """

    def __init__(self, mri0: sitk.Image, mri1: sitk.Image, mri2: sitk.Image, atlas: sitk.Image, patient_id: int, d: float = 10e-3):

        self.patient_id = patient_id

        # Initialize images
        self.mri0 = mri0  # T1w
        self.mri1 = mri1  # T1wC0.5
        self.mri2 = mri2  # T1wC1.0
        self.atlas0 = atlas  # Brain Atlas - Original copy
        self.atlas = atlas  # Brain Atlas - It includes deformations
        self.mri2_0 = None  # MRI subtraction: mri2 - mri0
        self.mri1_0 = None  # MRI subtraction: mri1 - mri0

        # Initialize the Region of Interest for data visualization
        # self.roi_mri0 = sitk.RegionOfInterestImageFilter()
        # self.roi_mri1 = sitk.RegionOfInterestImageFilter()
        # self.roi_mri2 = sitk.RegionOfInterestImageFilter()
        # self.roi_atlas = sitk.RegionOfInterestImageFilter()

        # Initialize masks for registration
        self.msk0 = None
        self.msk1 = None
        self.msk2 = None

        # Characteristics of common physical space
        self.origin = [0, 0, 0]
        self.spacing = [0.2, 0.2, 0.2]
        self.direction = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float).flatten()
        self.size = [512, 512, 128]

        # Attributes for Registration Boundaries
        self.mask = None
        self.mask_contour = None
        self.d = d  # the dilation radius to constrain the elastic registration (in mm)

        # Paths
        self.path = "E:/2021_local_data/2023_Gd_synthesis/dataset"

    def execute(self):
        """
        The registration consists in the following five steps:

        1. Rescale intensity of fixed and moving image in the range [0, 1], and resample the moving image to the fixed image (the largest MRI)
        with a 3D affine transformation. The resulting moving image has same spacings, origin and direction cosines as the fixed image,
        i.e. the fixed and moving image now share the same space.

        2. Match the intensity histogram of the moving image to the intensity histogram of the fixed image. This is
        necessary to make the image intensities comparable.

        3. Initialize the registration by (i) aligning the brain atlas center with the MR image brain center, and (ii) rescaling the brain atlas
        to match approximately the brain in the MR image.

        4. Registration. Register the brain atlas with a rigid and elastic transformation. A mask is used to limit the region available for
        registration. The mask is defined as the atlas brain mask dilated with a 3D ball structuring element of radius D (in mm).

        5. Calculate brain region in the MR image.

        6. Co-register the three MR images, using their brain masks to limit the registration to the actual brain volume.
        """

        # 1 --------------------------------------------------------------------
        output_size_mri0 = np.array(self.mri0.GetSize()) * np.array(self.mri0.GetSpacing() / np.array(self.spacing))
        output_size_mri1 = np.array(self.mri1.GetSize()) * np.array(self.mri1.GetSpacing() / np.array(self.spacing))
        output_size_mri2 = np.array(self.mri2.GetSize()) * np.array(self.mri2.GetSpacing() / np.array(self.spacing))
        output_size_atlas = np.array(self.atlas.GetSize()) * np.array(self.atlas.GetSpacing() / np.array(self.spacing))
        self.size = [int(max(x)) for x in zip(output_size_mri0, output_size_mri1, output_size_mri2, output_size_atlas)]

        self.mri0 = self.project_img_in_custom_space(self.mri0)
        self.mri1 = self.project_img_in_custom_space(self.mri1)
        self.mri2 = self.project_img_in_custom_space(self.mri2)
        self.atlas = self.project_img_in_custom_space(self.atlas)

        print(f"T1w - New Direction Cosines: {self.mri0.GetDirection()}\n"
              f"T1w - New Origin: {self.mri0.GetOrigin()}\n"
              f"T1w - New Size: {self.mri0.GetSize()}\n"
              f"T1w - New Spacing: {self.mri0.GetSpacing()}\n"
              f"T1wC0.5 - New Direction Cosines: {self.mri1.GetDirection()}\n"
              f"T1wC0.5 - New Origin: {self.mri1.GetOrigin()}\n"
              f"T1wC0.5 - New Size: {self.mri1.GetSize()}\n"
              f"T1wC0.5 - New Spacing: {self.mri1.GetSpacing()}\n"
              f"T1wC1.0 - New Direction Cosines: {self.mri2.GetDirection()}\n"
              f"T1wC1.0 - New Origin: {self.mri2.GetOrigin()}\n"
              f"T1wC1.0 - New Size: {self.mri2.GetSize()}\n"
              f"T1wC1.0 - New Spacing: {self.mri2.GetSpacing()}\n"
              f"atlas - New Direction Cosines: {self.atlas.GetDirection()}\n"
              f"atlas - New Origin: {self.atlas.GetOrigin()}\n"
              f"atlas - New Size: {self.atlas.GetSize()}\n"
              f"atlas - New Spacing: {self.atlas.GetSpacing()}\n"
              )

        # 2 --------------------------------------------------------------------
        print("Matching Histograms... ", end="")
        self.mri1 = sitk.HistogramMatching(image=self.mri1, referenceImage=self.mri0)
        self.mri2 = sitk.HistogramMatching(image=self.mri2, referenceImage=self.mri0)
        self.atlas = sitk.HistogramMatching(image=self.atlas, referenceImage=self.mri0)
        self.atlas0 = self.atlas
        print("Done.")

        # 3.0 to 5.0 -----------------------------------------------------------
        print("Starting Aligner...")
        brain_aligner = BrainAligner(self.mri0, self.atlas)
        brain_aligner.execute()
        self.atlas = sitk.Resample(self.atlas, brain_aligner.transform)

        print("Starting Registration...")
        self.generate_mask()
        self.register_atlas(self.mri0)
        check_registration(self.mri0, self.atlas, self.mask_contour, [brain_aligner.i, brain_aligner.j, brain_aligner.k], [10, 10, 5], 3)
        plt.savefig(os.path.join(self.path, f"{self.patient_id} T1wRC0.0 atlas registration.png"), dpi=600)
        # plt.show()
        plt.close()

        print(f"Generating Brain Mask for Patient {self.patient_id}")
        self.msk0 = sitk.BinaryThreshold(self.atlas, lowerThreshold=0.001, insideValue=1)
        self.atlas = self.atlas0

        # 3.1 to 5.1 ------------------------------------------------------------
        print("Starting Aligner...")
        brain_aligner = BrainAligner(self.mri1, self.atlas)
        brain_aligner.execute()
        self.atlas = sitk.Resample(self.atlas, brain_aligner.transform)

        print("Starting Registration...")
        self.generate_mask()
        self.register_atlas(self.mri1)
        check_registration(self.mri1, self.atlas, self.mask_contour, [brain_aligner.i, brain_aligner.j, brain_aligner.k], [10, 10, 5], 3)
        plt.savefig(os.path.join(self.path, f"{self.patient_id} T1wRC0.5 atlas registration.png"), dpi=600)
        # plt.show()
        plt.close()

        self.msk1 = sitk.BinaryThreshold(self.atlas, lowerThreshold=0.001, insideValue=1)
        self.atlas = self.atlas0

        # 3.2 to 5.2 ------------------------------------------------------------
        print("Starting Aligner...")
        brain_aligner = BrainAligner(self.mri2, self.atlas)
        brain_aligner.execute()
        self.atlas = sitk.Resample(self.atlas, brain_aligner.transform)

        print("Starting Registration...")
        self.generate_mask()
        self.register_atlas(self.mri2)
        check_registration(self.mri2, self.atlas, self.mask_contour, [brain_aligner.i, brain_aligner.j, brain_aligner.k], [10, 10, 5], 3)
        plt.savefig(os.path.join(self.path, f"{self.patient_id} T1wRC1.0 atlas registration.png"), dpi=600)
        # plt.show()
        plt.close()

        self.msk2 = sitk.BinaryThreshold(self.atlas, lowerThreshold=0.001, insideValue=1)
        self.atlas = self.atlas0

        # 6. --------------------------------------------------------------------
        self.register_mri()
        check_registration(self.mri0, self.mri1, None, [int(x // 2) for x in self.mri0.GetSize()], [10, 10, 5], 3)
        plt.savefig(os.path.join(self.path, f"{self.patient_id} T1wRC0.5 T1wRC0.0 registration.png"), dpi=600)
        plt.close()
        check_registration(self.mri0, self.mri2, None, [int(x // 2) for x in self.mri0.GetSize()], [10, 10, 5], 3)
        plt.savefig(os.path.join(self.path, f"{self.patient_id} T1wRC1.0 T1wRC0.0 registration.png"), dpi=600)
        plt.close()

        # 7. --------------------------------------------------------------------
        self.mri1_0 = sitk.Subtract(self.mri1, self.mri0)
        sitk.WriteImage(self.mri1_0, os.path.join(self.path, f"{self.patient_id} T1wRDC0.5.nii"))
        check_contrast(self.mri0, self.mri1_0, [int(x // 2) for x in self.mri1_0.GetSize()], [10, 10, 5], 3)
        plt.savefig(os.path.join(self.path, f"{self.patient_id} T1wRDC0.5.png"), dpi=600)
        plt.close()
        del self.mri1_0

        self.mri2_0 = sitk.Subtract(self.mri2, self.mri0)
        sitk.WriteImage(self.mri2_0, os.path.join(self.path, f"{self.patient_id} T1wRDC1.0.nii"))
        check_contrast(self.mri0, self.mri2_0, [int(x // 2) for x in self.mri2_0.GetSize()], [10, 10, 5], 3)
        plt.savefig(os.path.join(self.path, f"{self.patient_id} T1wRDC1.0.png"), dpi=600)
        plt.close()
        del self.mri2_0

        # 8. --------------------------------------------------------------------
        sitk.WriteImage(self.mri0, os.path.join(self.path, f"{self.patient_id} T1wRC0.0.nii"))
        sitk.WriteImage(self.msk0, os.path.join(self.path, f"{self.patient_id} T1wRC0.0.msk.nii"))
        sitk.WriteImage(self.mri1, os.path.join(self.path, f"{self.patient_id} T1wRC0.5.nii"))
        sitk.WriteImage(self.msk1, os.path.join(self.path, f"{self.patient_id} T1wRC0.5.msk.nii"))
        sitk.WriteImage(self.mri2, os.path.join(self.path, f"{self.patient_id} T1wRC1.0.nii"))
        sitk.WriteImage(self.msk2, os.path.join(self.path, f"{self.patient_id} T1wRC1.0.msk.nii"))



    def project_img_in_custom_space(self, img):
        """This method rescale the intensity in the range [0, 1], and then project the passed image in the physical space
        defined by the object attributes. By default, the image is resampled in order to have its physical origin at (0,0,0),
        spacing of [0.2, 0.2, 0.8] mm, and direction cosines [(1, 0, 0), (0, 1, 0), (0, 0, 1)]."""

        img = sitk.RescaleIntensity(img, 0, 1)

        # Define the Transform (translation and rotation) necessary to resample the image after setting a new origin and new direction cosines
        transform = sitk.AffineTransform(3)
        transform.SetTranslation(-(self.origin - np.array(img.GetOrigin())))
        output_direction = np.array(self.direction).reshape((3, 3))
        input_direction = np.array(img.GetDirection()).reshape((3, 3))
        rotation_matrix = np.dot(input_direction, np.linalg.inv(output_direction))
        transform.SetMatrix(rotation_matrix.flatten())

        # Define the Resampler Filter
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetSize(self.size)
        resampler.SetOutputOrigin(self.origin)
        resampler.SetOutputDirection(self.direction.flatten())
        resampler.SetOutputSpacing(self.spacing)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        img = resampler.Execute(img)

        return img

    def match_intensity_histograms(self):
        """Match intensity histogram of moving image to intensity histogram of fixed image"""
        self.atlas = sitk.HistogramMatching(image=self.atlas, referenceImage=self.mri0)

    def generate_mask(self):
        self.mask = sitk.BinaryThreshold(self.atlas, lowerThreshold=0.001, insideValue=1)
        r = np.array([self.d * 1e3 / self.mask.GetSpacing()[0],
                      self.d * 1e3 / self.mask.GetSpacing()[1],
                      self.d * 1e3 / self.mask.GetSpacing()[2]],
                     dtype=int)
        self.mask = sitk.BinaryDilate(self.mask, [int(x) for x in r])
        self.mask_contour = sitk.BinaryDilate(sitk.BinaryContour(self.mask), [2, int(2 * r[1] / r[0]), int(2 * r[2] / r[0])])

    def register_atlas(self, img):
        """This method register the brain atlas to the passed mri image, using a user defined mask that
        prevents the brain atlas to exceed the actual brain volume."""
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(img)
        elastixImageFilter.SetMovingImage(self.atlas)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
        elastixImageFilter.SetFixedMask(self.mask)
        elastixImageFilter.Execute()
        self.atlas = elastixImageFilter.GetResultImage()

        elastixImageFilter.SetFixedImage(img)
        elastixImageFilter.SetMovingImage(self.atlas)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("bspline"))
        elastixImageFilter.SetFixedMask(self.mask)
        elastixImageFilter.Execute()
        self.atlas = elastixImageFilter.GetResultImage()

    def register_mri(self):
        """This method register the mri1 and mri2 to mri0, using their masks to limit the registration to the
        actual brain volume. To initialize the registration, the center of mass of mr1 and mr2 image are aligned
        to the center of mass of mri0."""

        # Get brain masks' center of mass
        cm0 = get_center_of_mass(self.msk0)
        cm1 = get_center_of_mass(self.msk1)
        cm2 = get_center_of_mass(self.msk2)
        # Calculate transformation offset
        offset10 = cm1 - cm0
        offset20 = cm2 - cm0
        # Align (transform) masks and MRI images
        self.mri1 = transform_translate_rigid(self.mri1, offset10.tolist())
        self.msk1 = transform_translate_rigid(self.msk1, offset10.tolist())
        self.mri2 = transform_translate_rigid(self.mri2, offset20.tolist())
        self.msk2 = transform_translate_rigid(self.msk2, offset20.tolist())

        # Register MRI1 to MRI0
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(self.mri0)
        elastixImageFilter.SetMovingImage(self.mri1)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
        elastixImageFilter.SetFixedMask(self.msk0)
        elastixImageFilter.SetMovingMask(self.msk1)
        elastixImageFilter.Execute()
        self.mri1 = elastixImageFilter.GetResultImage()

        # Transform msk1 to match the position of registered MRI1
        parameters = elastixImageFilter.GetTransformParameterMap()
        self.msk1 = sitk.Transformix(self.msk1, parameters)

        # Register MRI2 to MRI0
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(self.mri0)
        elastixImageFilter.SetMovingImage(self.mri2)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
        elastixImageFilter.SetFixedMask(self.msk0)
        elastixImageFilter.SetMovingMask(self.msk2)
        elastixImageFilter.Execute()
        self.mri2 = elastixImageFilter.GetResultImage()

        # Transform msk1 to match the position of registered MRI1
        parameters = elastixImageFilter.GetTransformParameterMap()
        self.msk2 = sitk.Transformix(self.msk2, parameters)

