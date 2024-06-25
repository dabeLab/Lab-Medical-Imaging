import os

from image_utils import *
import pandas as pd

plot = True
os.chdir(r"C:\tierspital")  # set current working directory
matplotlib.rcParams.update({'font.size': 10})
algorithms = ["global mean", "global value", "global minimum", "global otsu", "local gaussian"]
maxn = 0
for algorithm in algorithms:
    path = rf"{os.getcwd()}\data processed\photos\segmentation threshold {algorithm}"
    df = pd.read_csv(rf"{path}\data.csv")
    mu = df["area"].mean()
    sigma = df["area"].std()
    n, bin_edges = plt.hist(log10(df["area"]), alpha=0.3, bins=50, range=(1, 5), ec='black', label=algorithm)[0:2]
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    x = linspace(min(bincenters), max(bincenters) , 300)
    spl = make_interp_spline(bincenters, n, k=3)
    power_smooth = spl(x)
    plt.plot(x, power_smooth, c="black", linewidth=1)
    if max(n) > maxn:
        maxn = max(n)

plt.vlines(x=log10(50), ymin=0, ymax=maxn, linestyles="--", color="red")
plt.vlines(x=log10(500), ymin=0, ymax=maxn, linestyles="--", color="red")
plt.legend()

plt.xlabel(r"$Log_{10}(Area)$ $(Log_{10}(px^2))$")
plt.ylabel("Counts")
plt.tight_layout()
plt.show()