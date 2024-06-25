import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.cm
import matplotlib.patches as mpatches
import pickle
from numpy import inf, linspace, sort, unique, zeros, max, min, argwhere, zeros_like, ones_like, where, nan, array, concatenate
from skimage import io
from skimage.measure import label, regionprops

DATA_PATH = r"C:\tierspital\data processed\photos\segmentation sam"
OUTPUT_PATH = DATA_PATH
IMAGE_PATH = r"C:\tierspital\data raw\photos"

IMAGE = linspace(1, 14, 14, dtype=int)
MODEL = ["vit_h"]#b", "vit_l", "vit_h"]
CROP_LAYERS = 0
DOWNSCALE_FACTOR = 1
AREA_THRESH = (20, 1000)
POINTS_PER_SIDE = [32, 64, 128]
THRESHOLD = None
CONNECTIVITY = 1
OVERLAP = 20  # Number of pixels removed from each image crop border (must be > 0 as the outest labels are not touching the picture borders)


process_each_picture = False
process_all_pictures_by_model_complexity = True
norm = matplotlib.colors.Normalize(vmin=32, vmax=128)
sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.coolwarm, norm=norm)
linestyle_dict = {"vit_b": "-", "vit_l": "--", "vit_h": "-."}
marker_dict = {0: "o", 1: "s"}

plt.figure()  # Plot figure with segmentation
plt.xlabel("Sample number")
plt.ylabel("Number of segments")

for c, model in enumerate(MODEL):
    for r, point_per_side in enumerate(POINTS_PER_SIDE):
        segments = zeros(len(IMAGE))
        segments_filt = zeros(len(IMAGE))
        for k, image in enumerate(IMAGE):
            print(f"{image}, {model}, {point_per_side}")
            img = io.imread(rf"{IMAGE_PATH}\{image:02d}.jpg")
            with open(rf"{OUTPUT_PATH}\{image:02d}.jpg, {model}, {point_per_side}, labels.dat", "rb") as reader:
                labels = pickle.load(reader)
            segments[k] = int(len(unique(labels)))
            with open(rf"{OUTPUT_PATH}\{image:02d}.jpg, {model}, {point_per_side}, labels filtered.dat", "rb") as reader:
                labels = pickle.load(reader)
            segments_filt[k] = int(len(unique(labels)))

        plt.plot(IMAGE, segments, linewidth=0, marker=marker_dict[0],
                 color=matplotlib.cm.coolwarm(norm(point_per_side)),
                 label=f"{model} - {point_per_side} points")
        plt.plot(IMAGE, segments_filt, linewidth=0, marker=marker_dict[1],
                 color=matplotlib.cm.coolwarm(norm(point_per_side)),
                 label=f"{model} - {point_per_side} points - filt")

plt.legend()
plt.tight_layout()
plt.savefig(rf"{OUTPUT_PATH}\segments vs model complexity vs filtering.png", dpi=1200, bbox_inches='tight')
