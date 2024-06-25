import matplotlib.pyplot as plt
import matplotlib.colors
import os
import pickle
from numpy import inf, unique, zeros, array, linspace
import itertools
from scipy.interpolate import make_interp_spline
from skimage import io
from skimage.segmentation import clear_border, watershed
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
from image_utils import show_anns, labels_from_sam_masks, check_overalpping_masks

DATA_PATH = r"C:\tierspital\data processed\photos\segmentation sam"
OUTPUT_PATH = r"C:\tierspital\data processed\photos\segmentation sam"

with open(f"{OUTPUT_PATH}/statistics a.dat", "rb") as reader:
    df = pickle.load(reader)
norm = matplotlib.colors.Normalize(vmin=32, vmax=128)
sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.coolwarm, norm=norm)
df = df[(df["crop layers"] == 0) & (df["downscale factor"] == 1)]
grps = df.groupby(["crop", "model"], dropna=False)
for idx, (key, grp) in enumerate(grps):
    plt.figure(idx)
    plt.title(f"crop: {key[0]}, model: {key[1]}")
    plt.xlabel(r"Area ($px^2$)")
    plt.ylabel("Counts")
    sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.coolwarm, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label="Points per side")
    subgrps = grp.groupby(["points per side"])
    for subidx, (subkey, subgrp) in enumerate(subgrps):
        print(f"Group: {key}, SubGroup: {subkey}")
        key_points_per_side = subkey[0]
        areas = []
        labels_list = subgrp["labels"].values  # return a list of matrixes
        for labels in labels_list:
            area = [x.area for x in regionprops(labels)]
            areas.append(area)
        areas = list(itertools.chain(*areas))
        n, bin_edges = plt.hist(areas, alpha=0.1, bins=50, range=(0, 1000), edgecolor='black', linewidth=1.2, color=matplotlib.cm.coolwarm(norm(key_points_per_side)))[0:2]
        bincenters = 1/2 * (bin_edges[1:] + bin_edges[:-1])
        x = linspace(min(bincenters), max(bincenters) , 100)
        spl = make_interp_spline(bincenters, n, k=3)
        power_smooth = spl(x)
        plt.plot(x, power_smooth, linewidth=2, alpha=0.8, color=matplotlib.cm.coolwarm(norm(key_points_per_side)))
    plt.savefig(rf"{OUTPUT_PATH}\statistics area, crop {key[0]}, model {key[1]}.png", bbox_inches="tight", dpi=1200)