from image_utils import *
import pandas as pd
import pickle
import scipy.stats as stats

plot = True
os.chdir(r"C:\tierspital")  # set current working directory
matplotlib.rcParams.update({'font.size': 10})
os.chdir(r"C:\tierspital\data processed\photos\segmentation edge canny")  # set current working directory
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
           "sigma": data["sigma"],
           "data": data["data"]}
    rows.append(row)

norm = matplotlib.colors.Normalize(vmin=0.5, vmax=2)
sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.coolwarm, norm=norm)
sm.set_array([])
plt.colorbar(sm, label="Sigma")
df = pd.DataFrame.from_dict(rows)
for row in rows:
    if row["image"] == "01.jpg" and row["threshold"] == 0.4 and row["structuring element dilation"] == (2,2):
        n, bin_edges = plt.hist(log10(row["data"]["area"]), alpha=0.1, bins=50, range=(1, 5), color=matplotlib.cm.coolwarm(norm(row["sigma"])))[0:2]
        bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        x = linspace(min(bincenters), max(bincenters) , 300)
        spl = make_interp_spline(bincenters, n, k=3)
        power_smooth = spl(x)
        plt.plot(x, power_smooth, linewidth=2, alpha=0.8, color=matplotlib.cm.coolwarm(norm(row["sigma"])))
        plt.xlabel(r"$Log_{10}$(Area) ($px^2$)")
        plt.ylabel("Counts")

plt.tight_layout()
plt.savefig("area vs sigma.jpg", dpi=1200)
plt.show()