import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.patches as mpatches
import pickle
import re
from numpy import inf, linspace, sort, unique, zeros, max, min, argwhere, zeros_like, ones_like, where
from scipy.interpolate import make_interp_spline
from skimage import io
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from image_utils import merge_labels_after_stitching

DATA_PATH = r"D:\My Drive\data processed\photos\segmentation sam"
OUTPUT_PATH = DATA_PATH
IMAGE_PATH = r"D:\My Drive\data raw\photos"

IMAGE = linspace(1, 14, 14, dtype=int)
MODEL = ["vit_b"]
CROP_LAYERS = 0
DOWNSCALE_FACTOR = 1
AREA_THRESH = (20, 1000)
POINTS_PER_SIDE = [32, 64, 128]
THRESHOLD = None
CONNECTIVITY = 1
OVERLAP = 20  # Number of pixels removed from each image crop border (must be > 0 as the outest labels are not touching the picture borders)

# image = io.imread(rf"{IMAGE_PATH}\01.jpg")
with open(f"{OUTPUT_PATH}/data.dat", "rb") as reader:
    df = pickle.load(reader)
df.astype({"crop": str})
df = df.loc[(df["image"].isin([f"{x:02d}.jpg" for x in IMAGE])) &
            (df["thresholding"].isnull()) &
            (df["crop layers"] == CROP_LAYERS) &
            (df["downscale factor"] == DOWNSCALE_FACTOR) &
            (df["model"].isin(MODEL)) &
            (df["points per side"].isin(POINTS_PER_SIDE))]

for key1, grp1 in df.groupby(["image"]):
    labels = zeros((grp1.lx.unique()[0], grp1.ly.unique()[0]), dtype=int)
    area_all = []

    # norm = matplotlib.colors.Normalize(vmin=32, vmax=128)
    for key2, grp2 in grp1.groupby(["model", "points per side"]):
        print(f"Processing {key1} {key2}")

        for key3, grp3 in grp2.groupby(["crop"]):

            xmin, ymin, xmax, ymax = [int(re.sub(r"[^0-9]", "", x)) for x in key3[0].split(",")]  # Get xmin, ymin, xmax, ymax of i-th crop
            labels_crop = grp3.labels.values[0]  # Get current crop labels
            print(f"{key1} - {key2} - {key3} - {len(unique(labels_crop))}")

            # Create a background (=0) border to separate touching labels. This is necessary because of the stitching operation
            # Otherwise, adjacent labels would be merged together and relabeled as one.
            boundary_matrix = find_boundaries(labels_crop, connectivity=1, mode="inner", background=0)
            labels_crop[boundary_matrix > 0] = 0

            if xmin == ymin == 0:
                labels[xmin:xmax, ymin:ymax] = labels_crop
            if xmin > 0 and ymin == 0:
                labels[xmin+OVERLAP:xmax, ymin:ymax] = labels_crop[OVERLAP:, :]
            if xmin == 0 and ymin > 0:
                labels[xmin:xmax, ymin+OVERLAP:ymax] = labels_crop[:, OVERLAP:]
            if xmin > 0 and ymin > 0:
                labels[xmin+OVERLAP:xmax, ymin+OVERLAP:ymax] = labels_crop[OVERLAP:, OVERLAP:]

        labels = label(labels, connectivity=CONNECTIVITY)  # re-label label matrix
        labels = merge_labels_after_stitching(labels)
        labels = label(labels, connectivity=CONNECTIVITY)  # re-label label matrix

        area = []
        regions = regionprops(labels)
        for region in regions:
            if region.area < AREA_THRESH[0] or region.area > AREA_THRESH[1]:    # if area outside passed range
                labels[labels == region.label] = 0
            else:
                area.append(region.area)
                area_all.append(region.area)

        labels = label(labels, connectivity=1)  # re-label label matrix

        # print(f"Total number of segments: {int(len(unique(labels)))}")
        # image_label_overlay = label2rgb(labels, image, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
        # plt.figure(1)  # Plot figure with segmentation
        # plt.imshow(image_label_overlay)
        # plt.axis("off")
        # plt.title(f"Number of segments: {int(len(unique(labels)))}")
        # plt.tight_layout()
        # plt.savefig(rf"{OUTPUT_PATH}\{key1}, {key2[0]}, {key2[1]}, segments.png", dpi=1200, bbox_inches='tight')
        #
        # plt.figure(2)  # Plot figure with segmentation and boundary boxes
        # plt.imshow(image_label_overlay)
        # regions = regionprops(labels)
        # plt.axis("off")
        # ax = plt.gca()
        # for region in regions:
        #     minr, minc, maxr, maxc = region.bbox
        #     rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.2)
        #     ax.add_patch(rect)
        # plt.title(f"Number of segments: {int(len(unique(labels)))}")
        # plt.tight_layout()
        # plt.savefig(rf"{OUTPUT_PATH}\{key1}, {key2[0]}, {key2[1]}, segments with boxes.png", dpi=1200, bbox_inches='tight')
        #
        # plt.figure(3)  # Plot distribution of area
        # c = matplotlib.cm.coolwarm(norm(key2[1]))
        # n, bin_edges = plt.hist(area, alpha=0.1, bins=50, range=(0, 1000), edgecolor='black', linewidth=1.2, color=c)[0:2]
        # bincenters = 1/2 * (bin_edges[1:] + bin_edges[:-1])
        # x = linspace(min(bincenters), max(bincenters) , 100)
        # spl = make_interp_spline(bincenters, n, k=3)
        # power_smooth = spl(x)
        # plt.plot(x, power_smooth, linewidth=2, alpha=0.8, color=c, label=key2[1])

        with open(f"{OUTPUT_PATH}/{key1[0]}, {key2[0]}, {key2[1]}, labels.dat", "wb") as writer:
            pickle.dump(labels, writer)

# plt.figure(3)
# plt.xlabel(r"Area ($px^2$)")
# plt.ylabel("Counts")
# plt.legend(title="Points per side")
# plt.savefig(rf"{OUTPUT_PATH}\{key1}, {key2} - segments distribution.png", dpi=1200, bbox_inches='tight')

# plt.figure(4)  # Plot distribution of area
# c = matplotlib.cm.coolwarm(norm(key2[1]))
# n, bin_edges = plt.hist(area_all, alpha=0.1, bins=50, range=(0, 1000), edgecolor='black', linewidth=1.2, color=c)[0:2]
# bincenters = 1/2 * (bin_edges[1:] + bin_edges[:-1])
# x = linspace(min(bincenters), max(bincenters) , 100)
# spl = make_interp_spline(bincenters, n, k=3)
# power_smooth = spl(x)
# plt.plot(x, power_smooth, linewidth=2, alpha=0.8, color=c)
# plt.show()
# print("Done.")