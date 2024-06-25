from skimage import color, io
from image_utils import *
import pandas as pd
from skimage.morphology import closing, square
from numpy import unique
import matplotlib.colors

"""SEGMENT IMAGES BY THRESHOLD vs THRESHOLD  VALUE vs STRUCTURING ELEMENT SIZE.
Segment images by threshold value and plot the results as a function of 
threshold value and strucuturing element size. Plot the feature size distribution on a 2D images."""

os.chdir(r"C:\tierspital")  # set current working directory
df = pd.read_csv(rf"{os.getcwd()}\data processed\photos\all - statistics - segmentation vs threshold value vs structuring element vs filter area.csv")
#grps = df.groupby(["structuring element", "area"])
#grps = df[df["image"]=="01.jpg"].groupby(["threshold"])
fig1, ax1 = plt.subplots(1, 1, figsize=(15/2.54, 10/2.54))
ax1.set_title(f"Segments vs Structuring element")
#norm = matplotlib.colors.Normalize(vmin=df["threshold"].min(), vmax=len(grps))
norm = matplotlib.colors.Normalize(vmin=df["threshold"].min(), vmax=df["threshold"].max())
grps = df.groupby(["image"])
for i, (key, val) in enumerate(grps):
    subgrps = val.groupby(["threshold"])
    for j, (subkey, subval) in enumerate(subgrps):
        print(key)
        #ax1.plot(val["threshold"].values, val["segments"].values, label=key, linewidth=2, alpha=0.7, color=matplotlib.cm.viridis(norm(idx)))
        ax1.plot(subval["structuring element"].values, subval["segments"].values, label=key, linewidth=2, alpha=0.5, color=matplotlib.cm.viridis(norm(subkey[0])))
        ax1.set_xlabel("structuring element")
        ax1.set_ylabel("segments")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.viridis, norm=norm)
sm.set_array([])
#plt.colorbar(sm, ax=ax1, label="Structuring element Square(n)")
plt.colorbar(sm, ax=ax1, label="Threshold")
fig1.savefig(rf"{os.getcwd()}\data processed\photos\all - statistics - segmented vs contrast vs structuring element.jpg", dpi=1200)
plt.show()
