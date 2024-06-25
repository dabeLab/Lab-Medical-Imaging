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

process_each_picture = True
process_all_pictures_by_model_complexity = True

for c, model in enumerate(MODEL):
    for r, point_per_side in enumerate(POINTS_PER_SIDE):
        for k, image in enumerate(IMAGE):
            print(f"{image}, {model}, {point_per_side}")
            with open(f"{OUTPUT_PATH}/{image:02d}.jpg, {model}, {point_per_side}, labels.dat", "rb") as reader:
                labels = pickle.load(reader)

            regions = regionprops(labels)
            area = zeros(len(regions))
            area[:] = nan
            ecce = zeros(len(regions))
            ecce[:] = nan
            for idx, region in enumerate(regions):
                area[idx] = region.area
                ecce[idx] = region.eccentricity

            if k == 0:
                areas = area
                ecces = ecce
            else:
                areas = concatenate((areas, area))
                ecces = concatenate((ecces, ecce))

            if process_each_picture is False:
                continue

            plt.figure()  # distribution area vs eccentricity
            plt.hist2d(area, ecce, 101, range=[[0, 500], [0, 1]], cmap=matplotlib.cm.viridis)
            plt.title(f"{model}, {point_per_side}")
            plt.xlabel(r"Area ($px^2$)")
            plt.ylabel(r"Eccentricity")
            plt.tight_layout()
            plt.savefig(rf"{OUTPUT_PATH}\distribution area vs eccentricity, {image:02d}.jpg, {model}, {point_per_side}.png", dpi=1200, bbox_inches='tight')
            plt.close()

            plt.figure()  # distribution area vs eccentricity zoom
            plt.hist2d(area, ecce, 101, range=[[20, 200], [0.9, 1]], cmap=matplotlib.cm.viridis)
            plt.title(f"{model}, {point_per_side}")
            plt.xlabel(r"Area ($px^2$)")
            plt.ylabel(r"Eccentricity")
            plt.tight_layout()
            plt.savefig(rf"{OUTPUT_PATH}\distribution area vs eccentricity zoom, {image:02d}.jpg, {model}, {point_per_side}.png", dpi=1200, bbox_inches='tight')
            plt.close()

            plt.figure()  # distribution area
            plt.hist(area,100,range=(0, 500), alpha=0.5, edgecolor = "black")[0:2]
            plt.title(f"{model}, {point_per_side}")
            plt.xlabel(r"Area ($px^2$)")
            plt.ylabel(r"Counts")
            plt.tight_layout()
            plt.savefig(rf"{OUTPUT_PATH}\distribution area, {image:02d}.jpg, {model}, {point_per_side}.png", dpi=1200, bbox_inches='tight')

            plt.xlim(20, 200)
            plt.title(f"{model}, {point_per_side}")
            plt.xlabel(r"Area ($px^2$)")
            plt.ylabel(r"Counts")
            plt.tight_layout()
            plt.savefig(rf"{OUTPUT_PATH}\distribution area zoom, {image:02d}.jpg, {model}, {point_per_side}.png", dpi=1200, bbox_inches='tight')
            plt.close()

            plt.figure()  # distribution eccentricity
            plt.hist(ecce, 100, range=(0, 1), alpha=0.5, edgecolor = "black")[0:2]
            plt.title(f"{model}, {point_per_side}")
            plt.xlabel(r"Eccentricity")
            plt.ylabel(r"Counts")
            plt.tight_layout()
            plt.savefig(rf"{OUTPUT_PATH}\distribution eccentricity, {image:02d}.jpg, {model}, {point_per_side}.png", dpi=1200, bbox_inches='tight')

            plt.xlim(0.9, 1)
            plt.title(f"{model}, {point_per_side}")
            plt.xlabel(r"Eccentricity")
            plt.ylabel(r"Counts")
            plt.tight_layout()
            plt.savefig(rf"{OUTPUT_PATH}\distribution eccentricity zoom, {image:02d}.jpg, {model}, {point_per_side}.png", dpi=1200, bbox_inches='tight')
            plt.close()

        if process_all_pictures_by_model_complexity is False:
            continue

        plt.figure()  # distribution area vs eccentricity
        plt.hist2d(areas, ecces, 100, range=[[0, 500], [0, 1]], cmap=matplotlib.cm.viridis)
        plt.title(f"{model}, {point_per_side}")
        plt.xlabel(r"Area ($px^2$)")
        plt.ylabel(r"Eccentricity")
        plt.tight_layout()
        plt.savefig(rf"{OUTPUT_PATH}\distribution area vs eccentricity, all, {model}, {point_per_side}.png", dpi=1200, bbox_inches='tight')
        plt.close()

        plt.figure()  # distribution area vs eccentricity zoom
        plt.hist2d(areas, ecces, 100, range=[[20, 200], [0.9, 1]], cmap=matplotlib.cm.viridis)
        plt.title(f"{model}, {point_per_side}")
        plt.xlabel(r"Area ($px^2$)")
        plt.ylabel(r"Eccentricity")
        plt.tight_layout()
        plt.savefig(rf"{OUTPUT_PATH}\distribution area vs eccentricity zoom, all, {model}, {point_per_side}.png", dpi=1200, bbox_inches='tight')
        plt.close()

        plt.figure()  # distribution area
        plt.hist(area,101,range=(0, 500), alpha=0.5, edgecolor = "black")[0:2]
        plt.title(f"{model}, {point_per_side}")
        plt.xlabel(r"Area ($px^2$)")
        plt.ylabel(r"Counts")
        plt.tight_layout()
        plt.savefig(rf"{OUTPUT_PATH}\distribution area, all, {model}, {point_per_side}.png", dpi=1200, bbox_inches='tight')

        plt.xlim(20, 200)
        plt.title(f"{model}, {point_per_side}")
        plt.xlabel(r"Area ($px^2$)")
        plt.ylabel(r"Counts")
        plt.tight_layout()
        plt.savefig(rf"{OUTPUT_PATH}\distribution area zoom, all, {model}, {point_per_side}.png", dpi=1200, bbox_inches='tight')
        plt.close()

        plt.figure()  # distribution eccentricity
        plt.hist(ecce, 101, range=(0, 1), alpha=0.5, edgecolor = "black")[0:2]
        plt.title(f"{model}, {point_per_side}")
        plt.xlabel(r"Eccentricity")
        plt.ylabel(r"Counts")
        plt.tight_layout()
        plt.savefig(rf"{OUTPUT_PATH}\distribution eccentricity, all, {model}, {point_per_side}.png", dpi=1200, bbox_inches='tight')

        plt.xlim(0.9, 1)
        plt.title(f"{model}, {point_per_side}")
        plt.xlabel(r"Eccentricity")
        plt.ylabel(r"Counts")
        plt.tight_layout()
        plt.savefig(rf"{OUTPUT_PATH}\distribution eccentricity zoom, all, {model}, {point_per_side}.png", dpi=1200, bbox_inches='tight')
        plt.close()
