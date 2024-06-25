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

DATA_PATH = r"C:\tierspital\data processed\photos\segmentation sam"
OUTPUT_PATH = DATA_PATH
IMAGE_PATH = r"C:\tierspital\data raw\photos"

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


plot_segments = True
norm = matplotlib.colors.Normalize(vmin=32, vmax=128)
sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.coolwarm, norm=norm)
linestyle_dict = {"vit_b": "-", "vit_l": "--", "vit_h": "-."}


plt.xlabel("Sample number")
plt.ylabel("Number of segments")

for k, image in enumerate(IMAGE):

    # Define box area
    img = io.imread(rf"{IMAGE_PATH}\{image:02d}.jpg")
    gra = rgb2gray(img)
    edge_map = canny(gra, sigma=1, low_threshold=0, high_threshold=1, mask=None)
    idxs = argwhere(edge_map == 1)
    box_xmin = min(idxs[:, 0]) - BOX_SAFE_MARGIN
    box_xmax = max(idxs[:, 0]) + BOX_SAFE_MARGIN
    box_ymin = min(idxs[:, 1]) - BOX_SAFE_MARGIN
    box_ymax = max(idxs[:, 1]) + BOX_SAFE_MARGIN
    box_x = array([box_xmin, box_xmin, box_xmax, box_xmax])
    box_y = array([box_ymin, box_ymax, box_ymax, box_ymin])
    rr, cc = polygon(box_x, box_y)
    box = polygon2mask(gra.shape, list(zip(box_x, box_y)))

    for c, model in enumerate(MODEL):
        for r, point_per_side in enumerate(POINTS_PER_SIDE):
            print(f"{image}, {model}, {point_per_side}")
            segments = zeros(len(IMAGE))

            # Load labels
            with open(rf"{OUTPUT_PATH}\{image:02d}.jpg, {model}, {point_per_side}, labels.dat", "rb") as reader:
                labels = pickle.load(reader)

            # and remove everythig outside box and mistaken features.
            labels[box == 0] = 0
            regions = regionprops(labels)
            for region in regions:
                if region.area < 50:
                    region.label = 0
                if region.eccentricity < 0.7 and region.area < 100:
                    region.label = 0

            labels = label(labels)

            with open(rf"{OUTPUT_PATH}\{image:02d}.jpg, {model}, {point_per_side}, labels filtered.dat", "wb") as writer:
                pickle.dump(labels, writer)

            if plot_segments:

                image_label_overlay = label2rgb(labels, img, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
                plt.figure()  # Plot figure with segmentation
                plt.imshow(image_label_overlay)
                plt.axis("off")
                plt.title(f"Number of segments: {int(len(unique(labels)))}")
                plt.tight_layout()
                plt.savefig(rf"{OUTPUT_PATH}\segments filtered,{image:02d}.jpg,{model},{point_per_side}.png", dpi=1200, bbox_inches='tight')

                ax = plt.gca()
                regions = regionprops(labels)
                for region in regions:
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.2)
                    ax.add_patch(rect)
                plt.tight_layout()
                plt.savefig(rf"{OUTPUT_PATH}\segments filtered with box,{image:02d}.jpg,{model},{point_per_side}.png", dpi=1200, bbox_inches='tight')

