import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
import os
import pickle
from skimage.measure import label, regionprops, regionprops_table

DATA_PATH = r"C:\tierspital\data processed\photos\segmentation sam"
IMAGE_PATH = r"C:\tierspital\data raw\photos"
OUTPUT_PATH = r"C:\tierspital\data processed\photos\segmentation sam"
IMAGE = "01.jpg"
CROP = "s"
MODEL = "vit_b"

with open(f"{OUTPUT_PATH}/statistics a.dat", "rb") as reader:
    df = pickle.load(reader)
norm = matplotlib.colors.Normalize(vmin=32, vmax=128)
sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.coolwarm, norm=norm)
df = df[(df["crop"]=="s") & (df["image"]=="01.jpg") & (df["model"]=="vit_b") & (df["crop layers"] == 0) & (df["downscale factor"] == 1)]
grps = df.groupby(["points per side"], dropna=False)
for idx, (key, grp) in enumerate(grps):
    plt.figure(idx)
    #plt.title(f"crop: {CROP}, model: {MODEL}")
    ax = plt.gca()
    plt.axis("off")
    plt.tight_layout()
    fname = f"{IMAGE}"
    fname = [x for x in os.listdir(IMAGE_PATH) if x.startswith(fname) and x.endswith(".jpg")][0]
    img = matplotlib.pyplot.imread(rf"{IMAGE_PATH}\{fname}", format=None)
    plt.imshow(img[400:601, 400:601])
    regions = regionprops(grp["labels"].values[0])
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    plt.savefig(rf"{OUTPUT_PATH}\{IMAGE}, {CROP}, size (1001, 1001, 3), model {MODEL}, points per side {key[0]}, bboxes.png", bbox_inches="tight", dpi=1200)



