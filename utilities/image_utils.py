import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy import ndimage as ndi
from numpy import ones, zeros, min, max, mean, std, histogram, linspace, digitize, unique, quantile, inf, log10, floor, logspace, sqrt, zeros_like, random, dstack, uint8, any, logical_and
from scipy.interpolate import make_interp_spline
from skimage.filters import threshold_mean, threshold_minimum, threshold_otsu, threshold_local, threshold_multiotsu
from skimage.feature import canny, peak_local_max
from skimage.exposure import equalize_hist, rescale_intensity
from skimage.segmentation import clear_border, watershed
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square, dilation, erosion
from skimage.color import label2rgb
from skimage.util import invert
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
matplotlib.rcParams.update({'font.size': 10})
fig_size = (30/2.54, 10/2.54)

def pixel_stats(image_rgb=None, image_gray=None, n_bins_rgb=50, n_bins_grayscale=50, plot=True, save=(False, None)):
    if image_rgb is not None:
        y_r, bin_edges_r = histogram(image_rgb[:, :, 0].flatten(), bins=n_bins_rgb, range=(0, 255))
        y_g, bin_edges_g = histogram(image_rgb[:, :, 1].flatten(), bins=n_bins_rgb, range=(0, 255))
        y_b, bin_edges_b = histogram(image_rgb[:, :, 2].flatten(), bins=n_bins_rgb, range=(0, 255))
        bincenters_r = 0.5 * (bin_edges_r[1:] + bin_edges_r[:-1])
        bincenters_g = 0.5 * (bin_edges_g[1:] + bin_edges_g[:-1])
        bincenters_b = 0.5 * (bin_edges_b[1:] + bin_edges_b[:-1])
        spl_r = make_interp_spline(bincenters_r, y_r, k=3)  # type: BSpline
        spl_g = make_interp_spline(bincenters_g, y_g, k=3)  # type: BSpline
        spl_b = make_interp_spline(bincenters_b, y_b, k=3)  # type: BSpline
        x_rgb = linspace(0, 255, 300)
        power_smooth_r = spl_r(x_rgb)
        power_smooth_g = spl_g(x_rgb)
        power_smooth_b = spl_b(x_rgb)
    if image_gray is not None:
        x_gray = linspace(0, 1, 300)
        y, bin_edges = histogram(image_gray.flatten(), bins=n_bins_grayscale, range=(0, 1))
        bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        spl = make_interp_spline(bincenters, y, k=3)  # type: BSpline
        power_smooth = spl(x_gray)
    if plot is True:
        fig, ax = plt.subplots(2, 2, figsize=(20/2.54, 15/2.54))
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(image_rgb)
        ax[0, 0].axis("off")
        ax[0, 1].set_title("Original RGB Hist.")
        ax[0, 1].hist(image_rgb[:, :, 0].flatten(), bins=n_bins_rgb, log=False, alpha=0.25, color="red")
        ax[0, 1].hist(image_rgb[:, :, 1].flatten(), bins=n_bins_rgb, log=False, alpha=0.25, color="green")
        ax[0, 1].hist(image_rgb[:, :, 2].flatten(), bins=n_bins_rgb, log=False, alpha=0.25, color="blue")
        ax[0, 1].plot(x_rgb, power_smooth_r, c="red")
        ax[0, 1].plot(x_rgb, power_smooth_g, c="green")
        ax[0, 1].plot(x_rgb, power_smooth_b, c="blue")
        ax[1, 0].set_title("Grayscale")
        ax[1, 0].imshow(image_gray, cmap=plt.cm.gray)
        ax[1, 0].axis("off")
        ax[1, 1].set_title("Greyscale Hist.")
        ax[1, 1].hist(image_gray.flatten(), bins=n_bins_grayscale, log=False, alpha=0.25)
        ax[1, 1].plot(x_gray, power_smooth, c="black")
        fig.tight_layout()
        if save[0] is True:
            fig.savefig(rf"{os.getcwd()}\data processed\photos\{save[1]} - statistics - rgb grayscale.jpg", dpi=1200)
    return {"x_rgb": x_rgb, "y_rgb":[power_smooth_r, power_smooth_g, power_smooth_b], "x_gray": x_gray, "y_gray": power_smooth_g}

def segmentation_threshold_mean_steps(image, structuring_element_dilation=None, structuring_element_erosion=None, n_dilation=1, n_erosion=1, area_thresh=None):
    """Uses mean value of pixel intensity to threshold the image. Brighter pixels are considered background. Perform a
    morphological closing operation to fill the dark holes, generates labels and remove the labels toching the image border. Finally filter
    out all labels with a pixel area outside the passe range."""
    img = invert(image)
    if structuring_element_dilation is not None:
        for idx in range(n_dilation):
            img = dilation(img, structuring_element_dilation)
    if structuring_element_erosion is not None:
        for idx in range(n_erosion):
            img = erosion(img, structuring_element_erosion)
    threshold = threshold_mean(img)
    mask = img > threshold
    labels, num = label(label_image=mask, background=0, return_num=True, connectivity=1)
    labels = clear_border(labels=labels, bgval=0)
    if area_thresh is not None and isinstance(area_thresh, tuple):
        for region in regionprops(labels):
            if region.area < area_thresh[0] or region.area > area_thresh[1]:    # if area outside passed range
                labels[labels == region.label] = 0                              # set the label equal to background (zero)
    image_label_overlay = label2rgb(labels, image, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
    return mask, labels, image_label_overlay

def segmentation_threshold_value_steps(image, threshold, structuring_element_dilation=square(2), structuring_element_erosion=square(2), n_dilation=1, n_erosion=1, area_thresh=None):
    """Uses the passed 'threshold' value as threshold for the pixel intensities. Brighter pixels are considered background. Perform a
    morphological closing operation to fill the dark holes, generates labels and remove the labels toching the image border. Finally filter
    out all labels with a pixel area outside the passe range."""
    img = rescale_intensity(image=image, in_range=(0, threshold))
    img = invert(img)
    for idx in range(n_dilation):
        img = dilation(img, structuring_element_dilation)
    for idx in range(n_erosion):
        img = erosion(img, structuring_element_erosion)
    mask = img > 0
    labels, num = label(label_image=mask, background=0, return_num=True, connectivity=1)
    labels = clear_border(labels=labels, bgval=0)
    if area_thresh is not None and isinstance(area_thresh, tuple):
        for region in regionprops(labels):
            if region.area < area_thresh[0] or region.area > area_thresh[1]:    # if area outside passed range
                labels[labels == region.label] = 0                              # set the label equal to background (zero)
    image_label_overlay = label2rgb(labels, image, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
    return mask, labels, image_label_overlay

def segmentation_threshold_minimum_steps(image, structuring_element_dilation=square(2), structuring_element_erosion=square(2), n_dilation=1, n_erosion=1, area_thresh=None):
    """Uses mean minimum variance alogorithm to threshold the image. Brighter pixels are considered background. Perform a
    morphological closing operation to fill the dark holes, generates labels and remove the labels toching the image border. Finally filter
    out all labels with a pixel area outside the passe range."""
    img = invert(image)
    for idx in range(n_dilation):
        img = dilation(img, structuring_element_dilation)
    for idx in range(n_erosion):
        img = erosion(img, structuring_element_erosion)
    threshold = threshold_minimum(img)
    mask = img > threshold
    labels, num = label(label_image=mask, background=0, return_num=True, connectivity=1)
    labels = clear_border(labels=labels, bgval=0)
    if area_thresh is not None and isinstance(area_thresh, tuple):
        for region in regionprops(labels):
            if region.area < area_thresh[0] or region.area > area_thresh[1]:    # if area outside passed range
                labels[labels == region.label] = 0                              # set the label equal to background (zero)
    image_label_overlay = label2rgb(labels, image, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
    return mask, labels, image_label_overlay

def segmentation_threshold_otsu_steps(image, structuring_element_dilation=square(2), structuring_element_erosion=square(2), n_dilation=1, n_erosion=1, area_thresh=None):
    """Otsu’s method calculates an “optimal” threshold (marked by a red line in the histogram below)
    by maximizing the variance between two tools of pixels, which are separated by the threshold.
    Equivalently, this threshold minimizes the intra-class variance. Brighter pixels are considered background. Perform a
    morphological closing operation to fill the dark holes, generates labels and remove the labels toching the image border. Finally filter
    out all labels with a pixel area outside the passe range."""
    img = invert(image)
    for idx in range(n_dilation):
        img = dilation(img, structuring_element_dilation)
    for idx in range(n_erosion):
        img = erosion(img, structuring_element_erosion)
    threshold = threshold_otsu(image)
    mask = img > threshold
    labels, num = label(label_image=mask, background=0, return_num=True, connectivity=1)
    labels = clear_border(labels=labels, bgval=0)
    if area_thresh is not None and isinstance(area_thresh, tuple):
        for region in regionprops(labels):
            if region.area < area_thresh[0] or region.area > area_thresh[1]:    # if area outside passed range
                labels[labels == region.label] = 0                              # set the label equal to background (zero)
    image_label_overlay = label2rgb(labels, image, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
    return mask, labels, image_label_overlay

def segmentation_threshold_local_steps(image, structuring_element_dilation, structuring_element_erosion, n_dilation, n_erosion, area_thresh=None, block_size=35, offset=0):
    """Compute a threshold mask image based on local pixel neighborhood. Also known as adaptive or dynamic thresholding.
    The threshold value is the weighted mean for the local neighborhood of a pixel subtracted by a constant.
    Alternatively the threshold can be determined dynamically by a given function, using the ‘generic’ method."""
    img = invert(image)
    for idx in range(n_dilation):
        img = dilation(img, structuring_element_dilation)
    for idx in range(n_erosion):
        img = erosion(img, structuring_element_erosion)
    threshold = threshold_local(image, block_size, method='gaussian', offset=offset)
    mask = img > threshold
    labels, num = label(label_image=mask, background=0, return_num=True, connectivity=1)
    labels = clear_border(labels=labels, bgval=0)
    if area_thresh is not None and isinstance(area_thresh, tuple):
        for region in regionprops(labels):
            if region.area < area_thresh[0] or region.area > area_thresh[1]:    # if area outside passed range
                labels[labels == region.label] = 0                              # set the label equal to background (zero)
    image_label_overlay = label2rgb(labels, image, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
    return mask, labels, image_label_overlay

def segmentation_edge_canny(image, threshold, sigma, structuring_element_dilation, structuring_element_erosion, n_dilation, n_erosion, area_thresh=None):
    """Segmentation by edge-recognition via Canny filter. In sequence,
    (1) define a mask with the passed threshold value (everything above threshold is considered background) and rescale the pixel intensity
    (2) apply Canny filter to the masked image
    (3) close features
    (4) label regions,
    (5) discard small regions."""
    if threshold is not None:
        image = rescale_intensity(image=image, in_range=(0, threshold))
    edge_map = canny(image, sigma=sigma, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False, mode='constant', cval=0.0)
    for idx in range(n_dilation):
        edge_map = dilation(edge_map, structuring_element_dilation)
    for idx in range(n_erosion):
        edge_map = erosion(edge_map, structuring_element_erosion)
    edge_map = ndi.binary_fill_holes(edge_map)
    labels, num = label(label_image=edge_map, background=0, return_num=True, connectivity=1)
    labels = clear_border(labels=labels, bgval=0)
    if area_thresh is not None and isinstance(area_thresh, tuple):
        for region in regionprops(labels):
            if region.area < area_thresh[0] or region.area > area_thresh[1]:    # if area outside passed range
                labels[labels == region.label] = 0                              # set the label equal to background (zero)
    image_label_overlay = label2rgb(labels, image, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
    return edge_map, labels, image_label_overlay

def segmentation_edge_watershed(image, threshold, n_dilation, n_erosion, structuring_element_dilation=square(2), structuring_element_erosion=square(2), area_thresh=None,
                                peaks_min_distance=10, peaks_rel_thresh=0.01, peaks_footprint=square(10)):
    """Segmentation by edge-recognition via watershed algorithm. For the algorithm to work, it needs that the user defines
    a region that is 'for sure' background, one that is 'for sure' foregroun', and an ambiguous region. In sequence,
    (1) define a mask with the passed threshold value (everything above threshold is considered background) and rescale the pixel intensity
    (2) Dilate n times, then erode n times to'isolate' features
    (3) calculate distance matrix
    (4) apply watershed algorithm to the distance matrix, where 0 is the background
    (5) get labels and discard small regions."""
    if threshold is not None:
        image = rescale_intensity(image=image, in_range=threshold)
    image = invert(image)
    for idx in range(n_dilation):
        image = dilation(image, structuring_element_dilation)
    for idx in range(n_erosion):
        image = erosion(image, structuring_element_erosion)
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, min_distance=peaks_min_distance, threshold_rel=peaks_rel_thresh, footprint=peaks_footprint)
    mask = zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, n = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image, watershed_line=True)
    labels = clear_border(labels=labels, bgval=0)
    if area_thresh is not None and isinstance(area_thresh, tuple):
        for region in regionprops(labels):
            if region.area < area_thresh[0] or region.area > area_thresh[1]:    # if area outside passed range
                labels[labels == region.label] = 0
    image_label_overlay = label2rgb(labels, image, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
    return labels, image_label_overlay

def segmentation_threshold_value(image, threshold=0, eq=0, footprint=square(1), area_thresh=None):
    """Uses the passed 'threshold' value as threshold for the pixel intensities. Brighter pixels are considered background. Perform a
    morphological closing operation to fill the dark holes, generates labels and remove the labels toching the image border. Finally filter
    out all labels with a pixel area outside the passe range."""
    # rescale_intensity(image=image, in_range=, out_range=
    # mask = closing(image < threshold, footprint)
    if eq == 0:
        image = equalize_hist(image=image, mask=image < threshold)
    elif eq == 1:
        image = rescale_intensity(image=image)
    mask = closing(image, footprint)
    labels, num = label(label_image=mask, background=0, return_num=True, connectivity=1)
    labels = clear_border(labels=labels, bgval=0)
    if area_thresh is not None and isinstance(area_thresh, tuple):
        for region in regionprops(labels):
            if region.area < area_thresh[0] or region.area > area_thresh[1]:    # if area outside passed range
                labels[labels == region.label] = 0                              # set the label equal to background (zero)
    image_label_overlay = label2rgb(labels, image, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
    return threshold, mask, labels, image_label_overlay

def plot_segmentation_steps(filename, image, mask, labels, image_label_overlay):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=fig_size)
    ax[0].set_title("Original")
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[1].set_title(f"Thresholded")
    ax[1].imshow(mask, cmap=plt.cm.gray)
    ax[1].axis("off")
    ax[2].set_title(f"Segmented n: {len(unique(labels))}")
    ax[2].imshow(image_label_overlay)
    ax[2].axis("off")
    fig.tight_layout()
    fig.savefig(rf"{filename}", dpi=1200)
    plt.close(fig)
    # if plot_regions is True:
    #     for region in regionprops(labels):
    #         # take regions with large enough areas
    #         if region.area <= area_thresh:
    #             # draw rectangle around segmented coins
    #             minr, minc, maxr, maxc = region.bbox
    #             rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
    #             ax[2].add_patch(rect)

def plot_segmentation(filename, image, mask, labels, image_label_overlay):
    x, y = image.shape
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10/2.54, y/x*15/2.54))
    ax.set_title(f"Segmented n: {len(unique(labels))}")
    ax.imshow(image_label_overlay)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(rf"{os.getcwd()}\data processed\photos\{filename}", dpi=1200)
    plt.close(fig)
    # if plot_regions is True:
    #     for region in regionprops(labels):
    #         # take regions with large enough areas
    #         if region.area <= area_thresh:
    #             # draw rectangle around segmented coins
    #             minr, minc, maxr, maxc = region.bbox
    #             rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
    #             ax[2].add_patch(rect)

def plot_pixel_stats_grayscale(filename, image_gray, thresh, n_bins_grayscale=50):
    x_gray = linspace(0, 1, 300)
    y, bin_edges = histogram(image_gray.flatten(), bins=n_bins_grayscale, range=(0, 1))
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    spl = make_interp_spline(bincenters, y, k=3)  # type: BSpline
    power_smooth = spl(x_gray)
    fig, ax = plt.subplots(1, 1, figsize=(20/2.54, 15/2.54))
    ax.hist(image_gray.flatten(), bins=n_bins_grayscale, log=False, alpha=0.5, color="gray")
    ax.plot(x_gray, power_smooth, c="black", linewidth=2)
    ax.axvline(thresh, color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(rf"{os.getcwd()}\data processed\photos\{filename}", dpi=1200)
    plt.close(fig)

def area_from_regionprops(labels):
    regions = regionprops(labels)
    data_label = zeros(len(regions))
    data_area = zeros_like(data_label)
    data_centroid = zeros_like(data_label)
    for idx, region in enumerate(regions):
        data_label[idx] = region.label
        data_area[idx] = region.area
        data_centroid[idx] = region.centroid
    return data_label, data_area, data_centroid

def plot_area_stats(filename, data):
    data = log10(data)
    n = int(sqrt(len(data)))
    fig, ax = plt.subplots(1, 1, figsize=(20/2.54, 15/2.54))
    x = linspace(1, floor(abs(max(data))+1), n)
    ax.hist(data, bins=x, log=False, color="gray", edgecolor='black', linewidth=1.2)
    ax.set_xlabel("Log10(area) [pixel^2]")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(rf"{os.getcwd()}\data processed\photos\{filename}", dpi=1200)
    plt.close(fig)

def customized_box_plot(n_box, percentiles, axes, *args, **kwargs):
    """
    Generates a customized boxplot based on the given percentile values
    """
    box_plot = axes.boxplot([[-9, -4, 2, 4, 9],]*n_box, *args, **kwargs)
    # Creates len(percentiles) no of box plots
    min_y, max_y = inf, -inf
    for box_no, (q1_start, q2_start, q3_start, q4_start, q4_end, fliers_xy) in enumerate(percentiles):
        box_plot['caps'][2*box_no].set_ydata([q1_start, q1_start])
        box_plot['whiskers'][2*box_no].set_ydata([q1_start, q2_start])
        box_plot['caps'][2*box_no + 1].set_ydata([q4_end, q4_end])
        box_plot['whiskers'][2*box_no + 1].set_ydata([q4_start, q4_end])
        box_plot['boxes'][box_no].set_ydata([q2_start, q2_start, q4_start, q4_start, q2_start])
        box_plot['medians'][box_no].set_ydata([q3_start, q3_start])
        if fliers_xy is not None and len(fliers_xy[0]) != 0:  # If outliers exist
            box_plot['fliers'][box_no].set(xdata = fliers_xy[0], ydata = fliers_xy[1])
            min_y = min(q1_start, min_y, fliers_xy[1].min())
            max_y = max(q4_end, max_y, fliers_xy[1].max())
        else:
            min_y = min(q1_start, min_y)
            max_y = max(q4_end, max_y)
        axes.set_ylim([min_y*1.1, max_y*1.1])
    return box_plot

def append_dataframes(path, output_filename):
    files = [x for x in os.listdir(path) if x.endswith(".csv")]
    for idx, file in enumerate(files):
        if idx == 0:
            df = pd.read_csv(rf"{path}\{file}", index_col=False)
        else:
            df = pd.concat((df, pd.read_csv(rf"{path}\{file}", index_col=False)))
    df.to_csv(rf"{path}\{output_filename}", sep=",", index=False)

def stats(labels, df=True):
    data = regionprops(labels)
    rows = []
    for i in range(len(data)):
        row = {"label": data[i].label, "area": data[i].area, "centroid": data[i]. centroid}
        rows.append(row)
    return pd.DataFrame.from_dict(rows)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = ones((m.shape[0], m.shape[1], 3))
        color_mask = random.random(3).tolist()
        for i in range(3):
            img[:,:,i] = color_mask[i]
        dstack((img, m*0.35))
        ax.imshow(dstack((img, m*0.35)))

def labels_from_sam_masks(image, masks):
    if len(masks) == 0:
        return
    labels = zeros((image.shape[0], image.shape[1]), dtype=uint8)
    """Sorting is fundamental because some labels overlap. Therefore the user will want
    to label first the largest regions, and only then the smaller ones."""
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    for idx, mask in enumerate(sorted_masks):
        m = mask["segmentation"]
        labels[m] = idx
    return labels

def check_overalpping_masks(image, masks):
    labels = zeros((image.shape[0], image.shape[1]), dtype=uint8)
    # sorted_labels = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    for idx, mask in enumerate(masks):
        m = mask["segmentation"]
        if any(logical_and(m, labels)):
            print(f"mask {idx} is superimposed")
        labels[m] = idx

def segmentation_sam(image, model, points_per_side, crop_n_layers, downscale_factor, clear_border_labels=False, area_thresh=None):
    mask_generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side, # default is 32 and maximum value is 32*4=128
        points_per_batch=128,  # the higher the more GPU memory required, default is 64
        # pred_iou_thresh=0.88,
        # stability_score_thresh=0.95,
        # stability_score_offset= 1.0,
        # box_nms_thresh: float = 0.7,
        crop_n_layers=crop_n_layers, # If >0, mask prediction will be run again on crops of the image.
        # crop_nms_thresh: float = 0.7,
        # crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor=downscale_factor,
        # point_grids = ,
        # min_mask_region_area: int = 20,  # Requires open-cv to run post-processing
        # output_mode: str = "binary_mask"
    )
    masks = mask_generator.generate(image)
    labels = labels_from_sam_masks(image, masks)
    if clear_border_labels:
        labels = clear_border(labels=labels, bgval=0)
    if area_thresh is not None and isinstance(area_thresh, tuple):
        for region in regionprops(labels):
            if region.area < area_thresh[0] or region.area > area_thresh[1]:
                labels[labels == region.label] = 0
    return labels

def segmentation_sam_with_cropper(dx, dy, image, model, points_per_side, crop_n_layers, downscale_factor, clear_border_labels=False, area_thresh=None):
    try:
        (image.shape[0] / dx).is_integer() and (image.shape[1] / dy).is_integer() and (image.shape[0]/dx == image.shape[1]/dy)
    except:
        exit("Crop size is not an integer number of image. Terminate.")

    labels = zeros(image.shape[0:2])
    counter = int(image.shape[0] / dx)
    for idx_y in range(counter):
        top_left_y = idx_y * dy
        for idx_x in range(counter):
            top_left_x = idx_x * dx
            print(top_left_x, top_left_y)
            crop = image[top_left_x:top_left_x+dx, top_left_y:top_left_y+dy]
            crop_labels = segmentation_sam(crop, model, points_per_side, crop_n_layers, downscale_factor, False, area_thresh)

            crop_labels[crop_labels>0] = crop_labels + max(labels)
            labels[top_left_x:top_left_x+dx, top_left_y:top_left_y+dy] = crop_labels
    return labels

def calculate_slice_bboxes(image_height: int,
                           image_width: int,
                           slice_height: int = 512,
                           slice_width: int = 512,
                           overlap_height_ratio: float = 0.2,
                           overlap_width_ratio: float = 0.2
                           ) -> list[list[int]]:
    """Given the height and width of an image, calculates how to divide the image into
    overlapping slices according to the height and width provided. These slices are returned
    as bounding boxes in xyxy format.
    :param image_height: Height of the original image.
    :param image_width: Width of the original image.
    :param slice_height: Height of each slice
    :param slice_width: Width of each slice
    :param overlap_height_ratio: Fractional overlap in height of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :param overlap_width_ratio: Fractional overlap in width of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :return: a list of bounding boxes in xyxy format
    """
    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min((image_width, x_max))
                ymax = min((image_height, y_max))
                xmin = max((0, xmax - slice_width))
                ymin = max((0, ymax - slice_height))
                slice_bboxes.append([(xmin, ymin), (xmax, ymax)])
            else:
                slice_bboxes.append([(x_min, y_min), (x_max, y_max)])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes

def plot_slice(image, slice_box, figure_index):
    plt.figure(figure_index)
    xmin, ymin = slice_box[0]
    xmax, ymax = slice_box[1]
    plt.imshow(image[xmin:xmax, ymin: ymax])

def plot_bbox_on_image(slice_box, figure_index, fill=False):
    plt.figure(figure_index)
    ax = plt.gca()
    xmin, ymin = slice_box[0]
    xmax, ymax = slice_box[1]
    rect = mpatches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=fill, edgecolor='red', linewidth=2)
    ax.add_patch(rect)


def merge_labels_after_stitching(labels):
    """check if neighbor elements are != 0 and != from element. Merge the labels """
    rmin, rmax, cmin, cmax = 0, labels.shape[0], 0, labels.shape[1]
    # run through all label matrix elements
    for r in range(rmax):
        for c in range(cmax):
            # check all neighbour of element r, c with connectivity = 2
            if 0 < r < rmax - 1 and 0 < c < cmax-1 and labels[r, c] > 0:
                for subr in [r-1, r, r+1]:
                    for subc in [c-1, c, c+1]:
                        if labels[subr, subc] != 0 and labels[subr, subc] != labels[r, c]:
                            target = labels[subr, subc]
                            labels[labels == target] = labels[r, c]
    return labels