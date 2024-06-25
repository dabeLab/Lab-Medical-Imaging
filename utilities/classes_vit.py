import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pickle
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage import io, color
from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2gray
from skimage.segmentation import find_boundaries, clear_border
from skimage.draw import polygon, polygon2mask
from skimage.feature import canny
from skimage.transform import resize
import torch

class SegmentationWithSAM:

    def __init__(self):
        self.path_image = None
        self.path_checkpoint = r"/content/models" # path where models are stored
        self.path_output = r"/content/data" # path where to save data output
        self.dict_checkpoint = {"vit_h": "sam_vit_h_4b8939.pth", "vit_b": "sam_vit_b_01ec64.pth", "vit_l": "sam_vit_l_0b3195.pth"}
        self.image_name = None
        """Image"""
        self.image = None
        self.image_lx = None
        self.image_ly = None
        self.image_dx = None
        self.image_dy = None
        self.image_aspect_ratio = None
        self.mask:np.array = None
        """SAM attributes"""
        self.model_type = "vit_h" # model type in string
        self.points_per_side = 128 # maximum value is 128
        self.crop_n_layers = 0
        self.downscale_factor = 1
        self.points_per_batch = 128 # the larger the faster SAM is, but also the larger RAM uses
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """Parameters for cropping purposes"""
        self.characteristic_dimension = 25 # characteristic dimension of the feature to segment
        self.threshold = 1 # everything above threshold (0-1) is set to white (255, 255, 255)
        self.slice_bboxes: list[tuple, tuple] = None # list of slice bboxes coordinates in the format [(xmin, ymin), (xmax, ymax)]
        self.overlap = 20  # Number of pixels removed from each image crop border (must be > 0 as the outest labels are not touching the picture borders)
        self.connectivity = 1
        """Parameters for filtering purposes"""
        self.thresh_area = (500, 5000) # segments with area outisde the boundaries are discarded
        self.thresh_eccentricity = 0.5 # # segments with eccentricity smaller than tresh_eccentricitiy are discarded
        self.box_safe_margin: int = 5 # For 'filter_box_walls' method: number of pixel to extend the approximate polygon representing the sample box
        """Output"""
        self.labels = None
        self.segments = None

    """ Methods: *** LOAD IMAGE *** """

    def load_image(self, filename):
        self.path_image = filename
        self.image_name = self.path_image.split("/")[-1]
        self.image = io.imread(self.path_image)
        self.update_attributes()

    '''Methods: *** SEGMENTATION WITH SAM ***'''

    # execute SAM on the whole image
    def run_sam(self, mask: bool = False):
        """Note: the method overwrite the attribute labels"""
        mask_generator = SamAutomaticMaskGenerator(
            model = self.model_type,
            points_per_side = self.points_per_side,
            points_per_batch = self.points_per_batch,
            # pred_iou_thresh=0.88,
            # stability_score_thresh=0.95,
            # stability_score_offset= 1.0,
            # box_nms_thresh: float = 0.7,
            crop_n_layers = self.crop_n_layers,
            # crop_nms_thresh: float = 0.7,
            # crop_overlap_ratio: float = 512 / 1500,
            crop_n_points_downscale_factor = self.downscale_factor)
        if mask:
            masks = mask_generator.generate(self.apply_mask_to_image(copy=True))
        if not mask:
            masks = mask_generator.generate(self.image)
        self.labels = self.labels_from_sam_masks(self.image, masks)
        self.segments = self.count_segments()
    # execute SAM on image slices. Requires to define the slices first
    def run_sam_on_sliced_image(self, mask: bool = False, flag_print: bool = False, flag_plot: bool = False, flag_bin: bool = False):
        """Run SAM on all image crops separately. Switch the flag on to print SAM status,
        and to save to disc plots and/or binary files. Note: the method overwrite the attribute labels, and
        the data of each image crop are appended to the object attribute 'labels'."""
        if self.slice_bboxes is None:
            print("'slice_bboxes' is empty. Please slice the image with 'slice_image' method.")
            return
        sam = sam_model_registry[self.model_type](checkpoint = f"{self.path_checkpoint}/{self.dict_checkpoint[self.model_type]}")
        sam.to(device = self.device)
        for idx, slice_box in enumerate(self.slice_bboxes):
            print(f'Running SAM on slice {idx}... ', end='')
            xmin, ymin = slice_box[0]
            xmax, ymax = slice_box[1]
            if mask:
                image_crop = self.apply_mask_to_image(copy=True)[xmin:xmax, ymin:ymax]
            if not mask:
                image_crop = self.image[xmin:xmax, ymin:ymax]

            if flag_print is True: # print segmentation status
                print(f"{idx+1} out of {len(self.slice_bboxes)}, {self.image_name}, size {self.image.shape}, crop {slice_box}, thresholding {self.threshold}, model {self.model_type}, points per side {self.points_per_side}, crop layers {self.crop_n_layers}, downscale factor {self.downscale_factor}... ", end="")

            # run segmentation on image crop
            mask_generator = SamAutomaticMaskGenerator(
                model = sam,
                points_per_side = self.points_per_side,
                points_per_batch = self.points_per_batch,
                crop_n_layers = self.crop_n_layers,
                crop_n_points_downscale_factor = self.downscale_factor)
            masks = mask_generator.generate(image_crop)
            labels_crop = self.labels_from_sam_masks(image_crop, masks)

            # define a filename for the image crop for saving purposes
            filename = f"{self.path_output}/{self.image_name}, size {self.image.shape}, crop {slice_box}, thresholding {self.threshold}, model {self.model_type}, points per side {self.points_per_side}, crop layers {self.crop_n_layers}, downscale factor {self.downscale_factor}, segments {len(np.unique(labels_crop))}"

            if flag_print is True: # print number of segments in image crop
                print(f"segments {len(np.unique(labels_crop))}")

            data = {"image": self.image_name,
                    "crop": slice_box,
                    "segments": len(np.unique(labels_crop)),
                    "labels": labels_crop}
            if idx == 0:
                self.labels = [data]
            else:
                self.labels.append(data)

            if flag_bin is True: # save binary copy to disc
                with open(f"{filename}.dat", "wb") as writer:
                    pickle.dump(data, writer)

            if flag_plot is True:
                plt.figure() # generate and save segmentation figure to disc
                image_label_overlay = label2rgb(labels_crop, image_crop, alpha=0.3, bg_label=0, bg_color=None, kind="overlay", saturation=0.6)
                plt.imshow(image_label_overlay)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(f"{filename}.png", bbox_inches="tight")

                ax = plt.gca()  # generate and save segmentation with bboxes figure to disc
                regions = regionprops(labels_crop)
                for region in regions:
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1)
                    ax.add_patch(rect)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(f"{filename}, bboxes.png", bbox_inches="tight")
                plt.close()
            print("Done.")
        self.labels = self.stitch_labels(flag_print=flag_print)
        self.segments = self.count_segments()

    '''Methods: *** LABELS FILTERING ***'''

    def filter_labels_by_shape(self):
        regions = regionprops(self.labels)
        for region in regions:
            if not self.thresh_area[0] <= region.area <= self.thresh_area[1]:
                self.labels[self.labels == region.label] = 0
            if region.eccentricity < self.thresh_eccentricity:
                self.labels[self.labels == region.label] = 0
        self.labels = label(self.labels, connectivity=1)  # re-label label matrix
        self.segments = self.count_segments()

    def filter_labels_on_the_border(self):
        """Note: the method overwrite the attribute labels"""
        self.labels = clear_border(self.labels)
        self.segments = self.count_segments()

    def filter_labels_outside_box_by_canny(self):
        """This method is meant to be used to remove all labels on the box walls that result from
        non-homogeneous illumination. Note: the method overwrite the label matrix with a new label, where
        all segments on the box walls are discarded (or set to background value)"""
        gra = rgb2gray(self.image)
        edge_map = canny(gra, sigma=1, low_threshold=0, high_threshold=1, mask=None)
        idxs = np.argwhere(edge_map == 1)
        box_xmin = min(idxs[:, 0]) - self.box_safe_margin
        box_xmax = max(idxs[:, 0]) + self.box_safe_margin
        box_ymin = min(idxs[:, 1]) - self.box_safe_margin
        box_ymax = max(idxs[:, 1]) + self.box_safe_margin
        box_x = np.array([box_xmin, box_xmin, box_xmax, box_xmax])
        box_y = np.array([box_ymin, box_ymax, box_ymax, box_ymin])
        polygon(box_x, box_y)
        box = polygon2mask(gra.shape, list(zip(box_x, box_y)))
        self.labels[box == 0] = 0

    '''Methods: *** MASK TO REMOVE BACKGROUND - BEFORE SAM ***'''

    def make_mask_to_remove_background_by_hist(self, epsilon=5):
        """The method finds the most recurring pixel color, and mask out everything else."""
        hist_r = np.histogram(self.image[:, :, 0].flatten(), 100)
        hist_g = np.histogram(self.image[:, :, 1].flatten(), 100)
        hist_b = np.histogram(self.image[:, :, 2].flatten(), 100)
        max_hist_r = hist_r[1][np.argmax(hist_r[0])]
        max_hist_g = hist_g[1][np.argmax(hist_g[0])]
        max_hist_b = hist_b[1][np.argmax(hist_b[0])]
        range_r = (max_hist_r * (1-epsilon/100), max_hist_r * (1+epsilon/100))
        range_g = (max_hist_g * (1-epsilon/100), max_hist_g * (1+epsilon/100))
        range_b = (max_hist_b * (1-epsilon/100), max_hist_b * (1+epsilon/100))
        mask = (self.image[:, :, 0] <= range_r[1]) * (self.image[:, :, 0] >= range_r[0]) * \
               (self.image[:, :, 1] <= range_g[1]) * (self.image[:, :, 1] >= range_g[0]) * \
               (self.image[:, :, 2] <= range_b[1]) * (self.image[:, :, 2] >= range_b[0])
        return mask

    def make_mask_to_remove_background_by_polygon_input(self):
        """The method asks the user to select the vertex of a polygon that would contain all
        features to segment. Then, it convert the vertex into a polygon and the polygon into a mask,
        which is stored in the object attribute."""
        # Display the image using imshow
        plt.imshow(self.image)
        plt.title('Click to Define Polygon (Right-click to Delete, Middle-click to Finish)')
        # Ask the user to click on the image to define polygon vertices
        print("Click on the image to define the polygon vertices. Right-click to finish.")
        vertices = plt.ginput(n=-1, timeout=0, show_clicks=True)
        # Close the image plot
        plt.close()
        # Display the extracted polygon coordinates
        print("Polygon vertices:")
        for vertex in vertices:
            print(f"X: {vertex[0]}, Y: {vertex[1]}")
        # Create a mask using polygon2mask
        mask = polygon2mask(self.image.shape[0:2], np.flip(vertices, axis=1))
        self.mask = np.logical_not(mask)
        #self.image[np.logical_not(mask)] = 0, 0, 0
        #plt.imshow(self.image)

    def apply_mask_to_image(self, copy:bool=True):
        """Retunr a copy of the image where everything outside the mask is set to black"""
        if copy:
            masked_image = self.image
            masked_image[np.logical_not(self.mask)] = 0, 0, 0
            return masked_image
        if not copy:
            self.image[np.logical_not(self.mask)] = 0, 0, 0

    '''Methods: *** IMAGE INFO & STATISTICS ***'''

    def get_image_info(self):
        print(f"Image path: {self.path_image}\n"
              f"Aspect ratio {self.image_aspect_ratio:.1f}\n"
              f"Lenght X: {self.image_lx} px, Delta X: {self.image_dx:.4f} px\n"
              f"Length Y: {self.image_ly} px, Delta Y: {self.image_dy:.4f} px\n"
              f"Characteristic dimension: {self.characteristic_dimension} px")

    def pixel_stats_rgb(self, n_bins_rgb=100, plot:bool=False, save:bool=False):
        plt.figure()
        r = plt.hist(self.image[:, :, 0].flatten(), range=(0, 255), bins=n_bins_rgb, log=False, alpha=0.25, color="red")
        g = plt.hist(self.image[:, :, 1].flatten(), range=(0, 255), bins=n_bins_rgb, log=False, alpha=0.25, color="green")
        b = plt.hist(self.image[:, :, 2].flatten(), range=(0, 255), bins=n_bins_rgb, log=False, alpha=0.25, color="blue")
        if save:
            plt.savefig(f"{os.getcwd()}/{self.image_name}, pixel rgb histogram.png", dpi=1200)
        if plot:
            plt.show()
        plt.close()
        return r, g, b

    def pixel_stats_grayscale(self, n_bins_grayscale=100, plot:bool=False, save:bool=False):
        g = plt.hist(rgb2gray(self.image).flatten(), range=(0, 1), bins=n_bins_grayscale, log=False)
        if save:
            plt.savefig(f"{os.getcwd()}/{self.image_name}, pixel grayscale histogram.png", dpi=1200)
        if plot:
            plt.show()
        plt.close()
        return g

    '''Methods: *** LABELS STATISTICS ***'''

    def hist_labels_geometry(self, plot:bool=True):
        """Generate histrograms data and plots for labels area and eccentricity"""
        regions = regionprops(self.labels)
        area = np.zeros(len(regions))
        ecce = np.zeros(len(regions))
        for idx, region in enumerate(regions):
            area[idx] = region.area
            ecce[idx] = region.eccentricity
        if plot is True:
            plt.figure()
            plt.hist(x=area, bins=100)
            plt.figure()
            plt.hist(x=ecce, bins=100)
        return area, ecce

    '''Methods: *** UTILITIES ***'''

    # slice images several image crops, depending on self.characteristic_dimension
    def slice_image(self):
        """Features are nicely segmented when their characheristic length is about
        max 1/10th of the image size. It comes therefore necessary to run SAM on
        image crops to ensure that the condition on the ratio is met."""
        slice_lx = 10 * self.characteristic_dimension
        slice_ly = 10 * self.characteristic_dimension
        slice_bboxes = self.calculate_slice_bboxes(self.image_ly, self.image_lx, slice_ly, slice_lx, 0.2, 0.2)
        print(f"Number of slices: {int(len(slice_bboxes))}")
        self.slice_bboxes = slice_bboxes
    # stich labels together while running self.sam_on_sliced_image
    def stitch_labels(self, flag_print: bool = False):
        if len(self.labels) <= 1:
            print("There is nothing to stitch.")
            return

        # create empty matrix for labeling
        labels = np.zeros((self.image_lx, self.image_ly))

        for n, val in enumerate(self.labels):

            if flag_print:
                print(f"Stitching {n+1} out of {len(self.slice_bboxes)}, crop {val['crop']}... ", end="")

            (xmin, ymin), (xmax, ymax) = val["crop"] # Get crop coordinates. Data format is [(x_min, y_min), (x_max, y_max)]
            labels_crop = val["labels"]  # Get image crop labels
            # Create a background (=0) border to separate touching labels. This is necessary because of the stitching operation
            # Otherwise, adjacent labels would be merged together and relabeled as one.
            boundary_matrix = find_boundaries(labels_crop, connectivity=1, mode="inner", background=0)
            labels_crop[boundary_matrix > 0] = 0

            if xmin == ymin == 0:
                labels[xmin:xmax, ymin:ymax] = labels_crop
            if xmin > 0 and ymin == 0:
                labels[xmin+self.overlap:xmax, ymin:ymax] = labels_crop[self.overlap:, :]
            if xmin == 0 and ymin > 0:
                labels[xmin:xmax, ymin+self.overlap:ymax] = labels_crop[:, self.overlap:]
            if xmin > 0 and ymin > 0:
                labels[xmin+self.overlap:xmax, ymin+self.overlap:ymax] = labels_crop[self.overlap:, self.overlap:]

            labels = label(labels, connectivity=self.connectivity)  # re-label label matrix
            labels = self.merge_labels_after_stitching(labels)
            labels = label(labels, connectivity=self.connectivity)  # re-label label matrix

            if flag_print:
                print(f"Done.")

        return labels
    # conts the number of segments.
    def count_segments(self):
        return len(np.unique(self.labels))
    # removes the alpha channel from png images
    def remove_trasparency_from_png_images(self):
        self.image = self.image[:, :, 0:3]
    # resize the image to y = 1600 px, preserving the aspect ratio
    def resize_image(self, y:int=1600):
        f = y / self.image_ly
        self.image = resize(self.image, (self.image_lx*f, self.image_ly*f), anti_aliasing=True, preserve_range=True).astype('uint8')
        self.update_attributes()
    # Update image attributes. Might be useful if: e.g. the image is rescaled.
    def update_attributes(self):
        self.image_lx = self.image.shape[0]
        self.image_ly = self.image.shape[1]
        self.image_dx = (1-0)/self.image.shape[0]
        self.image_dy = (1-0)/self.image.shape[1]
        self.image_aspect_ratio = self.image.shape[0] / self.image.shape[1]

    @staticmethod
    def calculate_slice_bboxes(image_height: int, image_width: int, slice_height: int = 512, slice_width: int = 512, overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2):
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

    @staticmethod
    def labels_from_sam_masks(image, masks):
        if len(masks) == 0:
            return
        labels = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        """Sorting is fundamental because some labels overlap. Therefore the user will want
        to label first the largest regions, and only then the smaller ones."""
        sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        for idx, mask in enumerate(sorted_masks):
            m = mask["segmentation"]
            labels[m] = idx
        return labels

    @staticmethod
    def merge_labels_after_stitching(labels):
        """Check if neighbor elements are != 0 and != from (r, c) element. If so, then merge the labels"""
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

    def threshold(self):
        """Note: this method overwrites the image originally stored in the object
        with a thresholded copy of the image, where all pixels with grayscale value
        larger than threshold are set to white."""
        image_gs = color.rgb2gray(self.image)
        mask = image_gs > self.threshold
        self.image[mask] = [255, 255, 255]

    '''Methods: *** PLOT ***'''

    # display segmented image with boundary boxes
    def plot_figure_label_overlay(self, bboxes:bool = True):
        plt.figure()
        image_label_overlay = label2rgb(self.labels, self.image, alpha=0.3, bg_label=0, bg_color=None, kind="overlay", saturation=0.6)
        plt.imshow(image_label_overlay)
        plt.axis("off")
        if bboxes is True:
            ax = plt.gca()
            regions = regionprops(self.labels)
            for region in regions:
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)
        plt.show()

    '''Methods: *** LOAD and SAVE ***'''

    # Save only the data attributes to a binary file
    def save_attributes_to_disc(self):
        with open(f"{self.path_output}/{self.image_name}, {self.model_type}, {self.points_per_side}, labels.dat", "wb") as file:
            pickle.dump(self.__dict__, file)

    # Load the data attributes from the binary file
    def load_attributes_from_disc(self, filename):
        with open(filename, "rb") as file:
            self.__dict__ = pickle.load(file)

    def save_figure_label_overlay_to_disc(self):
        # save segmented image without boundary boxes to disc
        plt.figure()
        image_label_overlay = label2rgb(self.labels, self.image, alpha=0.3, bg_label=0, bg_color=None, kind="overlay", saturation=0.6)
        plt.imshow(image_label_overlay)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{self.path_output}/{self.image_name}, {self.model_type}, {self.points_per_side}, segmentation.png", bbox_inches="tight", dpi=1200)
        # save segmented image with boundary boxes to disc
        ax = plt.gca()
        regions = regionprops(self.labels)
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.2)
            ax.add_patch(rect)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{self.path_output}/{self.image_name}, {self.model_type}, {self.points_per_side}, segmentation with bboxes.png", bbox_inches="tight", dpi=1200)
        plt.close()
