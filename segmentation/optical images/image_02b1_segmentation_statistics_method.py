from image_utils import *
import pandas as pd

plot = True
os.chdir(r"C:\tierspital")  # set current working directory
matplotlib.rcParams.update({'font.size': 10})
path = rf"{os.getcwd()}\data processed\photos\segmentation threshold local gaussian"
df = pd.read_csv(rf"{path}\data.csv")
grps = df.groupby("image")
maxn = 0
for key, grp in grps:
    print(key)
    mu = grp["area"].mean()
    sigma = grp["area"].std()
    n, bin_edges = plt.hist(log10(grp["area"]), alpha=0.2, bins=50, range=(1, 5), ec='black')[0:2]
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    x = linspace(min(bincenters), max(bincenters) , 300)
    spl = make_interp_spline(bincenters, n, k=3)
    power_smooth = spl(x)
    plt.plot(x, power_smooth, c="black", linewidth=1)
    if max(n) > maxn:
        maxn = max(n)

plt.vlines(x=log10(50), ymin=0, ymax=maxn, linestyles="--", color="red")
plt.vlines(x=log10(500), ymin=0, ymax=maxn, linestyles="--", color="red")

plt.xlabel(r"$Log_{10}(Area)$ $(Log_{10}(px^2))$")
plt.ylabel("Counts")
plt.tight_layout()
plt.savefig(rf"{path}\area histogram.jpg", dpi=1200)
plt.show()