import numpy as np
import SimpleITK as sitk
from skimage.segmentation import felzenszwalb
import matplotlib.pyplot as plt


def cosines_to_patient(direction_cosines):
    # Convert the direction cosines to a 3x2 matrix
    matrix = np.array(direction_cosines).reshape((3, 2))
    # Determine orientation labels
    orientation_labels = []

    # determines the sign of the angle between the image first row and the right-to-left patient direction
    if matrix[0, 0] > 0:
        orientation_labels.append('R')
    elif matrix[0, 0] < 0:
        orientation_labels.append('L')

    # determines the sign of the angle between the image first column and the anterior-to-posterior patient direction
    if matrix[1, 1] > 0:
        orientation_labels.append('A')
    elif matrix[1, 1] < 0:
        orientation_labels.append('P')

    # determines the sign of the angle between the image first row and the head(S)-to-feet(I) patient direction
    if matrix[2, 0] > 0:
        orientation_labels.append('S')
    elif matrix[2, 0] < 0:
        orientation_labels.append('I')

    # Join orientation labels to get the final orientation
    orientation = ''.join(orientation_labels)

    return orientation


def read_dicom_series(path_dicom_series):
    """Read a DICOM series and convert it to 3D nifti image"""
    # Load the DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(path_dicom_series)
    reader.SetFileNames(dicom_series)
    image = reader.Execute()
    # Convert the SimpleITK image to NIfTI format in memory
    # nifti_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
    # nifti_image.CopyInformation(image)
    # Convert the SimpleITK image to NIfTI format
    # sitk.WriteImage(image, path_nifti)
    return image


def register(fix_img: sitk.Image, mov_img: sitk.Image, registration="rigid"):
    """ This function takes a moving image (mov_img) and a fixed image (image).
    It resample the moving image to the fixed image, and then register
    the moving image first rigidly, then elastically. To facilitate the registration,
    the function:
    1. Pre-orient the mov_img so that its direction cosines align with those of the fixed image.
    2. Histogram matching of the atlas intensity range to the subject’s image intensity range.
    3. Initialization of the registration by preliminary placing the moving image around the brain in the MR image.
    To this purpose, the MR image is segmented with the Felzenszwalb algorithm, and the region corresponding to
    the brain selected as the one whose centroid is closest to a reference point, assigned to each anatomical plane
    (sagittal, dorsal and transverse), and representing the average brain’s position. This reference point is
    selected by the user by left-click on the sagittal, dorsal and transverse section of the MR image.
    4. Registration - Rigid. To reduce computational time and help the registration procedure to focus on the brain region, we
    applied a mask to the fixed target image. The mask is chosen to correspond to the atlas brain mask,
    dilated with a 3D ball structuring element of radius 10 pixels.
    5 Registration - Elastic."""

    slice2plot = 256
    extract_axial_section(mov_img)
    extract_coronal_section(mov_img)

    # Cast the moving image to the pixel type of the fixed image
    mov_img = sitk.Cast(mov_img, fix_img.GetPixelID())

    # Create a 3D rigid transformation object
    transform = sitk.Euler3DTransform()
    transform = sitk.AffineTransform(3)
    # Set the center of rotation to the center of the fixed image
    transform.SetCenter(fix_img.TransformContinuousIndexToPhysicalPoint([index / 2.0 for index in fix_img.GetSize()]))
    # Set the rotation matrix
    fix_img_direction_cosines = np.array(fix_img.GetDirection()).reshape((3, 3))
    mov_img_direction_cosines = np.array(mov_img.GetDirection()).reshape((3, 3))
    rotation_matrix = np.dot(np.linalg.inv(fix_img_direction_cosines), mov_img_direction_cosines)
    transform.SetMatrix(rotation_matrix.flatten())
    # Set the translation - difference between origins
    transform.SetTranslation(np.array(fix_img.GetOrigin()) - np.array(mov_img.GetOrigin()))
    sitk.Resample(image1=mov_img, referenceImage=fix_img, interpolator=sitk.sitkLinear, transform=transform)

    extract_axial_section(mov_img)
    extract_coronal_section(mov_img)

    # 2. ALIGN DIRECTION COSINES
    # Get the direction matrices from the fixed and moving images
    direction_cosines_fix = np.array(fix_img.GetDirection()).reshape((3, 3))
    direction_cosines_mov = np.array(mov_img.GetDirection()).reshape((3, 3))
    # Calculate the rotation matrix to align the moving image with the fixed image
    rotation_matrix = np.dot(np.linalg.inv(direction_cosines_fix), direction_cosines_mov)
    # Apply the rotation matrix to the direction cosines of the moving image, and update direction cosines
    direction_cosines_mov = np.dot(rotation_matrix, direction_cosines_mov)
    # Convert the new direction cosines to a 1D list for SimpleITK
    direction_cosines_mov_list = direction_cosines_mov.flatten().tolist()
    # Set the new direction cosines for the moving image
    mov_img.SetDirection(direction_cosines_mov_list)
    # Now, 'moving_image' has been oriented to align with the direction cosines of 'fixed_image'

    extract_axial_section(mov_img)
    extract_coronal_section(mov_img)


    # 3. ALIGN ORIGIN
    # Get the origins of the fixed and moving images
    fix_origin = fix_img.GetOrigin()
    mov_origin = mov_img.GetOrigin()
    # Calculate the differences in the x, y, and z coordinates
    dx = fix_origin[0] - mov_origin[0]
    dy = fix_origin[1] - mov_origin[1]
    dz = fix_origin[2] - mov_origin[2]
    # Update the origin of the moving image
    mov_origin = (mov_origin[0] + dx, mov_origin[1] + dy, mov_origin[2] + dz)
    mov_img.SetOrigin(mov_origin)
    # Now, 'moving_image' has been translated to align its origin with the fixed image
    extract_axial_section(mov_img)
    extract_coronal_section(mov_img)


    # 1. RESAMPLE MOVING IMAGE
    # resample atlas, so that its dimension is in the same order of magnitude of the mri brain
    mov_img = sitk.Resample(image1=mov_img,  # image to resample
                            referenceImage=fix_img,  # reference image
                            transform=sitk.Transform(),
                            interpolator=sitk.sitkLinear, )  # type of interpolation

    extract_axial_section(mov_img)
    extract_coronal_section(mov_img)

    plt.show()

    # 4. MATCH INTENSITY HISTOGRAMS
    mov_img = sitk.HistogramMatching(image=mov_img, referenceImage=fix_img)

    # 5. INITIALIZE REGISTRATION - PRE-POSITIONING
    # segment the MR image by graph-method - Felzenswalb. This method works on 2D images.
    felzenszwalb(sitk.GetArrayFromImage(mov_img), scale=3.0, sigma=0.5, min_size=5)

    # 6. REGISTRATION
    parameterMap = sitk.GetDefaultParameterMap(registration)
    #sitk.PrintParameterMap(parameterMap)

    # create an elastic object
    elastixImageFilter = sitk.ElastixImageFilter()
    # set fixed and moving images, and mapping parameters
    elastixImageFilter.SetFixedImage(fix_img)
    elastixImageFilter.SetMovingImage(mov_img)
    elastixImageFilter.SetParameterMap(parameterMap)
    # execute registration
    elastixImageFilter.Execute()
    # get resulting image
    resultImage = elastixImageFilter.GetResultImage()
    # get transformation parameters
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    return resultImage


def extract_sagittal_section(img: sitk.Image):
    """It assumes a 3D image"""
    size = img.GetSize()
    spacing = img.GetSpacing()
    n = int(size[0]/2)
    img_slice = sitk.Extract(img, [0, size[1], size[2]], [n, 0, 0])
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(img_slice), cmap='gray', aspect=spacing[2] / spacing[1])
    plt.axis("off")
    plt.title("sagittal")


def extract_coronal_section(img: sitk.Image):
    """It assumes a 3D image"""
    size = img.GetSize()
    spacing = img.GetSpacing()
    n = int(size[1]/2)
    img_slice = sitk.Extract(img, [size[0], 0, size[2]], [0, n, 0])
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(img_slice), cmap='gray', aspect=spacing[2] / spacing[0])
    plt.axis("off")
    plt.title("coronal")


def extract_axial_section(img: sitk.Image):
    """It assumes a 3D image"""
    size = img.GetSize()
    spacing = img.GetSpacing()
    n = int(size[2] / 2)
    img_slice = sitk.Extract(img, [size[0], size[1], 0], [0, 0, n])
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(img_slice), cmap='gray', aspect=spacing[1] / spacing[0])
    plt.axis("off")
    plt.title("axial")


def check_registration(fix_img: sitk.Image, mov_img: sitk.Image):
    """It assumes a 3D image"""
    fix_img_size = fix_img.GetSize()
    mov_img_size = mov_img.GetSize()
    fix_img_spacing = fix_img.GetSpacing()
    mov_img_spacing = mov_img.GetSpacing()
    n = int(size[2] / 2)
    img_slice = sitk.Extract(img, [size[0], size[1], 0], [0, 0, n])
    plt.figure()
    plt.imshow(sitk.GetArrayFromImage(img_slice), cmap='gray', aspect=spacing[1] / spacing[0])
    plt.axis("off")
    plt.title("axial")
