import os
import pickle
import matplotlib.pyplot as plt
from skimage import color, io
from image_utils import *
import pandas as pd
from skimage.morphology import closing, square
from numpy import unique, roll
import matplotlib.colors

"""SEGMENT IMAGES BY THRESHOLD vs THRESHOLD  VALUE vs STRUCTURING ELEMENT SIZE.
Segment images by threshold value and plot the results as a function of 
threshold value and strucuturing element size. Plot the feature size distribution on a 2D images."""

filter_area = (20, inf)

os.chdir(r"C:\tierspital\data processed\photos\segmentation threshold global value vs value")  # set current working directory
files = [x for x in os.listdir(os.getcwd()) if (x.endswith(".dat"))]
rows = []
for idx, file in enumerate(files):
    with open(file, "rb") as reader:
        data = pickle.load(reader)
        row = {"image": data["image"],
               "threshold": data["threshold"],
               "structuring element dilation": data["structuring element dilation"],
               "structuring element erosion": data["structuring element erosion"],
               "segments": len(unique(data["data"]["label"])),
               "area threshold": data["area thresh"],
               "label area mean": mean(data["data"]["area"]),
               "label area std": std(data["data"]["area"])}
        rows.append(row)
df = pd.DataFrame.from_dict(rows)

norm = matplotlib.colors.Normalize(vmin=0.2, vmax=0.6)
sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.coolwarm, norm=norm)
sm.set_array([])
plt.figure(1)
plt.title("Segments vs Structuring Element")
plt.xlabel("Structuring Element")
plt.ylabel("Segments")
plt.colorbar(sm, label="Threshold")
plt.figure(2)
plt.title("Label Area vs Structuring Element")
plt.xlabel("Structuring Element")
plt.ylabel(r"Label Area ($px^2$)")
plt.colorbar(sm, label="Threshold")

grps = df[df["area threshold"]==filter_area].groupby(["image"])
for i, (key, val) in enumerate(grps):
    subgrps = val.groupby(["threshold"])
    for j, (subkey, subval) in enumerate(subgrps):
        subval = subval.sort_values("structuring element dilation")
        xticks = [str(x) for x in subval["structuring element dilation"].values]
        xticks = roll(xticks, 1)
        x = linspace(0, len(subval["structuring element dilation"]), len(subval["structuring element dilation"]))
        y1 = subval["segments"].values
        y1 = roll(y1, 1)
        y2 = subval["label area mean"].values
        y2 = roll(y2, 1)

        plt.figure(1)
        plt.plot(x, y1, label=f"image: {key} - thresh: {subkey}", linewidth=2, alpha=1, color=matplotlib.cm.coolwarm(norm(subkey[0])))
        plt.xticks(ticks=x, labels=xticks, rotation=45, ha='right')
        plt.tight_layout()

        plt.figure(2)
        plt.plot(x, y2, label=f"image: {key} - thresh: {subkey}", linewidth=2, alpha=1, color=matplotlib.cm.coolwarm(norm(subkey[0])))
        plt.xticks(ticks=x, labels=xticks, rotation=45, ha='right')
        plt.tight_layout()

norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.coolwarm, norm=norm)
sm.set_array([])
plt.figure(3)
plt.title("Segments vs Threshold")
plt.xlabel("Structuring Element")
plt.ylabel("Segments")
plt.colorbar(sm, label="Structuring Element")
plt.figure(4)
plt.title("Label Area vs Threshold")
plt.xlabel("Structuring Element")
plt.ylabel(r"Label Area ($px^2$)")
plt.colorbar(sm, label="Structuring Element")

grps = df[df["area threshold"]==filter_area].groupby(["image"])
for i, (key, val) in enumerate(grps):
    subgrps = val.groupby(["structuring element dilation"])
    for j, (subkey, subval) in enumerate(subgrps):

        x = subval["threshold"].values
        y1 = subval["segments"].values
        y2 = subval["label area mean"].values

        plt.figure(3)
        plt.plot(x, y1, label=f"image: {key} - thresh: {subkey}", linewidth=2, alpha=1, color=matplotlib.cm.coolwarm(norm(j)))
        plt.tight_layout()

        plt.figure(4)
        plt.plot(x, y2, label=f"image: {key} - thresh: {subkey}", linewidth=2, alpha=1, color=matplotlib.cm.coolwarm(norm(j)))
        plt.tight_layout()

plt.figure(1)
plt.savefig("segments vs structuring element.jpg", dpi=1200)
plt.figure(2)
plt.savefig("label area vs structuring element.jpg", dpi=1200)
plt.figure(3)
plt.savefig("segments vs threshold.jpg", dpi=1200)
plt.figure(4)
plt.savefig("label area vs threshold.jpg", dpi=1200)
plt.show()
