import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


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

    """Read a DICOM series and save it in memory as sitk.Image"""
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(path_dicom_series)
    reader.SetFileNames(dicom_series)
    image = reader.Execute()

    return image


def extract_axial_section(ax: plt.Axes, img: sitk.Image, n, cmap):
    """Extract slice from sitk.Image in the xy plane."""
    img_slice = sitk.Extract(img, [img.GetSize()[0], img.GetSize()[1], 0], [0, 0, n])
    img_extent = (0,
                  (0 + img_slice.GetSize()[0]) * img_slice.GetSpacing()[0],
                  (0 - img_slice.GetSize()[1]) * img_slice.GetSpacing()[1],
                  0)
    ax.imshow(sitk.GetArrayFromImage(img_slice), cmap=cmap, extent=img_extent)
    ax.set_axis_off()


def extract_coronal_section(ax: plt.Axes, img: sitk.Image, n, cmap):
    """Extract slice from sitk.Image in the xz plane."""
    img_slice = sitk.Extract(img, [img.GetSize()[0], 0, img.GetSize()[2]], [0, n, 0])
    img_extent = (0,
                  (0 + img_slice.GetSize()[0]) * img_slice.GetSpacing()[0],
                  (0 - img_slice.GetSize()[1]) * img_slice.GetSpacing()[1],
                  0)
    ax.imshow(sitk.GetArrayFromImage(img_slice), cmap=cmap, extent=img_extent)
    ax.set_axis_off()


def extract_sagittal_section(ax: plt.Axes, img: sitk.Image, n, cmap):
    """Extract slice from sitk.Image in the yz plane."""
    img_slice = sitk.Extract(img, [0, img.GetSize()[1], img.GetSize()[2]], [n, 0, 0])
    img_extent = (0,
                  (0 + img_slice.GetSize()[0]) * img_slice.GetSpacing()[0],
                  (0 - img_slice.GetSize()[1]) * img_slice.GetSpacing()[1],
                  0)
    ax.imshow(sitk.GetArrayFromImage(img_slice), cmap=cmap, extent=img_extent)
    ax.set_axis_off()


def custom_colormap(vmin=0, vmax=1):
    # Create a custom colormap based on "jet"
    cmap_jet = plt.colormaps.get_cmap("afmhot")
    n = 256  # Number of values in the colormap
    jet_colors = cmap_jet(np.linspace(vmin, vmax, n))
    # Set alpha channel to 0 where the value is 0
    jet_colors[:, 3] = np.where(np.linspace(vmin, vmax, n) == 0, 0, 0.5)
    return LinearSegmentedColormap.from_list("custom_jet", jet_colors, n)


def custom_colormap_for_contrast(vmin=-1, vmax=1):
    # Create a custom colormap based on "bwr"
    cmap_jet = plt.colormaps.get_cmap("bwr")
    n = 256  # Number of values in the colormap
    jet_colors = cmap_jet(np.linspace(vmin, vmax, n))
    # Set alpha channel to 0 where the value is 0
    jet_colors[:, 3] = np.where(np.linspace(vmin, vmax, n) == 0, 0, 0.5)
    return LinearSegmentedColormap.from_list("custom_jet", jet_colors, n)


def custom_colormap_for_mask():
    # Define the colors and corresponding values
    colors = np.array([(0, 0, 0, 0), (0.8, 0, 0, 0.5)])  # (R, G, B, Alpha)
    # Create a ListedColormap
    custom_cmap = ListedColormap(colors)
    return custom_cmap


def check_registration(fix_img: sitk.Image, mov_img: sitk.Image, mask: sitk.Image or None, slice, delta_slice, n_slice):
    """
    Plot xy, xz and yz slices of fixed and moving images overlay. The selected slices are determined
    by the list of n tuples [(i, j, k)_1, (i, j, k)_2 ... (i, j, k)_n], where the i, j, and k represent
    the voxel indices in the image coordinate system. Since the same tuples are used to extract slices of
    both images, and the tuples are selected on the basis of the fixed image properties,
    the plot is ideal to display the quality of the registration.
    """

    # Calculate initial slice
    i0 = slice[0] - (delta_slice[0] * n_slice) // 2
    j0 = slice[1] - (delta_slice[1] * n_slice) // 2
    k0 = slice[2] - (delta_slice[2] * n_slice) // 2

    fig, ax = plt.subplots(n_slice, 3)  # in cm
    ax = ax.flatten()

    for t, idx in enumerate(np.linspace(0, n_slice + 3, 3, dtype=int)):

        i = int(i0 + delta_slice[0] * idx)
        j = int(j0 + delta_slice[0] * idx)
        k = int(k0 + delta_slice[0] * idx)

        ax[idx + 0].set_title("xy - axial") if idx == 0 else None
        extract_axial_section(ax[idx + 0], fix_img, k, "gray")
        extract_axial_section(ax[idx + 0], mov_img, k, custom_colormap())
        extract_axial_section(ax[idx + 0], mask, k, custom_colormap_for_mask()) if isinstance(mask, sitk.Image) else None

        ax[idx + 1].set_title("xz - coronal") if idx == 0 else None
        extract_coronal_section(ax[idx + 1], fix_img, j, "gray")
        extract_coronal_section(ax[idx + 1], mov_img, j, custom_colormap())
        extract_coronal_section(ax[idx + 1], mask, j, custom_colormap_for_mask()) if isinstance(mask, sitk.Image) else None

        ax[idx + 2].set_title("yz - sagittal") if idx == 0 else None
        extract_sagittal_section(ax[idx + 2], fix_img, i, "gray")
        extract_sagittal_section(ax[idx + 2], mov_img, i, custom_colormap())
        extract_sagittal_section(ax[idx + 2], mask, i, custom_colormap_for_mask()) if isinstance(mask, sitk.Image) else None

    plt.tight_layout()
    return fig, ax


def check_contrast(img: sitk.Image, cnt: sitk.Image, slice, delta_slice, n_slice):

    # Calculate initial slice
    i0 = slice[0] - (delta_slice[0] * n_slice) // 2
    j0 = slice[1] - (delta_slice[1] * n_slice) // 2
    k0 = slice[2] - (delta_slice[2] * n_slice) // 2

    fig, ax = plt.subplots(n_slice, 3)  # in cm
    ax = ax.flatten()

    for t, idx in enumerate(np.linspace(0, n_slice + 3, 3, dtype=int)):
        i = int(i0 + delta_slice[0] * idx)
        j = int(j0 + delta_slice[0] * idx)
        k = int(k0 + delta_slice[0] * idx)

        ax[idx + 0].set_title("xy - axial") if idx == 0 else None
        extract_axial_section(ax[idx + 0], img, k, "gray")
        extract_axial_section(ax[idx + 0], cnt, k, custom_colormap_for_contrast())

        ax[idx + 1].set_title("xz - coronal") if idx == 0 else None
        extract_coronal_section(ax[idx + 1], img, j, "gray")
        extract_coronal_section(ax[idx + 1], cnt, j, custom_colormap_for_contrast())

        ax[idx + 2].set_title("yz - sagittal") if idx == 0 else None
        extract_sagittal_section(ax[idx + 2], img, i, "gray")
        extract_sagittal_section(ax[idx + 2], cnt, i, custom_colormap_for_contrast())

    plt.tight_layout()
    return fig, ax


def rescale_intensity_zscore(img: sitk.Image or list[sitk.Image]):
    """Rescale image intensity according to the z-score metric in the interval [0, 1]."""

    for idx, val in enumerate(img) if isinstance(img, list) else enumerate([img]):

        stats = sitk.StatisticsImageFilter()
        stats.Execute(val)
        sigma = stats.GetSigma()
        mean = stats.GetMean()
        val = sitk.Add(val, -mean)
        val = sitk.Divide(val, sigma)

        img[idx] = val

    return img


def get_center_of_mass(img: sitk.Image):
    """Calculate the center of mass of the passed image, in the physical space."""
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(img, img)
    cm = stats.GetCenterOfGravity(1)
    return np.array(cm)


def transform_translate_rigid(img: sitk.Image, offset):
    """Rigidly translate an image and return a resampled version of the image."""
    transform = sitk.TranslationTransform(3)
    transform.SetOffset(offset)
    img = sitk.Resample(img,
                        transform=transform,
                        interpolator=sitk.sitkLinear,
                        defaultPixelValue=0,
                        outputPixelType=img.GetPixelIDValue())
    return img


def closest_divisible_by_power_of_two(target: int, n: int):
    """
    Return the closest number to target divisible by 2 n times.
    """
    closest_number = None
    min_difference = np.inf
    num = target + 1  # Start searching from the number next to the target
    while True:
        if num % (2 ** n) == 0:
            difference = num - target
            if difference < min_difference:
                min_difference = difference
                closest_number = num
                break  # Found the closest number, exit the loop
        num += 1

    return closest_number