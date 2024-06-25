import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.patches as mpatches
import pickle
import re
import os
from numpy import inf, linspace, sort, unique, zeros, max, min, argwhere, zeros_like, ones_like, where, nan, array, concatenate
from scipy.interpolate import make_interp_spline
from skimage import io
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from image_utils import merge_labels_after_stitching

DATA_PATH = r"C:\tierspital\data processed\photos\segmentation sam"
OUTPUT_PATH = DATA_PATH
IMAGE_PATH = r"C:\tierspital\data raw\photos"

IMAGE = linspace(1, 14, 14, dtype=int)
MODEL = ["vit_b", "vit_l", "vit_h"]
CROP_LAYERS = 0
DOWNSCALE_FACTOR = 1
AREA_THRESH = (20, 1000)
POINTS_PER_SIDE = [32, 64, 128]
THRESHOLD = None
CONNECTIVITY = 1
OVERLAP = 20  # Number of pixels removed from each image crop border (must be > 0 as the outest labels are not touching the picture borders)


process_each_picture = False
process_all_pictures_by_model_complexity = True

for c, model in enumerate(MODEL):
    for r, point_per_side in enumerate(POINTS_PER_SIDE):
        for k, image in enumerate(IMAGE):
            print(f"{image}, {model}, {point_per_side}")

            img = io.imread(rf"{IMAGE_PATH}\{image:02d}.jpg")
            with open(rf"{OUTPUT_PATH}\{image:02d}.jpg, {model}, {point_per_side}, labels.dat", "rb") as reader:
                labels = pickle.load(reader)

            print(f"Total number of segments: {int(len(unique(labels)))}")
            image_label_overlay = label2rgb(labels, img, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
            plt.figure()  # Plot figure with segmentation
            plt.imshow(image_label_overlay)
            plt.axis("off")
            plt.title(f"Number of segments: {int(len(unique(labels)))}")
            plt.tight_layout()
            plt.savefig(rf"{OUTPUT_PATH}\segments,{image:02d}.jpg,{model},{point_per_side}.png", dpi=1200, bbox_inches='tight')

            ax = plt.gca()
            regions = regionprops(labels)
            for region in regions:
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.2)
                ax.add_patch(rect)
            plt.title(f"Number of segments: {int(len(unique(labels)))}")
            plt.tight_layout()
            plt.savefig(rf"{OUTPUT_PATH}\segments with box,{image:02d}.jpg,{model},{point_per_side}.png", dpi=1200, bbox_inches='tight')

            plt.ylim(400, 601)
            plt.savefig(rf"{OUTPUT_PATH}\segments with box crop,{image:02d}.jpg,{model},{point_per_side}.png", dpi=1200, bbox_inches='tight')
            plt.close()

