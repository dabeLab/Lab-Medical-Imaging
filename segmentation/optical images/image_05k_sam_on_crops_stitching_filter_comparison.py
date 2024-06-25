import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.patches as mpatches
import pickle
from numpy import inf, linspace, sort, unique, zeros, max, min, argwhere, zeros_like, ones_like, where, nan, array, concatenate, uint8
from skimage import io
from skimage.color import rgb2gray
from skimage.color import label2rgb
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
from skimage.feature import canny
from skimage.draw import polygon, polygon_perimeter, polygon2mask
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity

DATA_PATH = r"T:\data processed\optical images\segmentation sam\batch00"
OUTPUT_PATH = DATA_PATH
IMAGE_PATH = r"T:\data raw\optical images\batch00"

IMAGE = linspace(1, 14, 14, dtype=int)
MODEL = ["vit_b", "vit_l", "vit_h"]
CROP_LAYERS = 0
DOWNSCALE_FACTOR = 1
AREA_THRESH = (20, 1000)
POINTS_PER_SIDE = [32, 64, 128]
THRESHOLD = 0.3
CONNECTIVITY = 1
OVERLAP = 20  # Number of pixels removed from each image crop border (must be > 0 as the outest labels are not touching the picture borders)
BOX_SAFE_MARGIN = 5  # Number of pixel to extend the approximate polygon representing the sample box

only_box = True
norm = matplotlib.colors.Normalize(vmin=32, vmax=128)
sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.coolwarm, norm=norm)
linestyle_dict = {"vit_b": "-", "vit_l": "--", "vit_h": "-."}

plt.xlabel("Sample number")
plt.ylabel("Number of segments")

for k, image in enumerate(IMAGE):
    img = io.imread(rf"{IMAGE_PATH}\{image:02d}.jpg")

    for c, model in enumerate(MODEL):
        for r, point_per_side in enumerate(POINTS_PER_SIDE):
            print(f"{image}, {model}, {point_per_side}")
            segments = zeros(len(IMAGE))

            # Load labels
            with open(rf"{OUTPUT_PATH}\{image:02d}.jpg, {model}, {point_per_side}, labels.dat", "rb") as reader:
                labels = pickle.load(reader)
            with open(rf"{OUTPUT_PATH}\{image:02d}.jpg, {model}, {point_per_side}, labels filtered.dat", "rb") as reader:
                labels_filtered = pickle.load(reader)
            plt.figure()
            image_label_overlay = label2rgb(labels_filtered, img, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
            plt.imshow(image_label_overlay)
            plt.axis("off")
            plt.title(f"Number of segments: {int(len(unique(labels_filtered)))}")
            plt.tight_layout()
            regions = regionprops(labels_filtered)
            for region in regions:
                if not 20 <= region.area <= inf:
                    labels_filtered[labels_filtered == region.label] = 0
                if region.eccentricity < 0.7 and region.area > 100:
                    labels_filtered[labels_filtered == region.label] = 0
            labels_filtered = label(labels_filtered, connectivity=1)  # re-label label matrix

            ax = plt.gca()
            regions = regionprops(labels_filtered)
            for region in regions:
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.2)
                ax.add_patch(rect)


            plt.savefig(rf"{OUTPUT_PATH}\{image:02d}.jpg,{model},{point_per_side}.png", dpi=1200, bbox_inches='tight')
            plt.close()
