from image_utils import *
import pandas as pd
import scipy.stats as stats

plot = True
os.chdir(r"C:\tierspital\data processed\photos\segmentation watershed")  # set current working directory

matplotlib.rcParams.update({'font.size': 10})
df = pd.read_csv(rf"{os.getcwd()}\data.csv")
grps = df[df["footprint"]=="(10, 10)"].groupby(["algorithm", "contrast", "dilation", "erosion", "distance"])
for key, grp in grps:
    print(key)
    mu = grp["area"].mean()
    sigma = grp["area"].std()
    n, bin_edges = plt.hist(log10(grp["area"]), alpha=0.3, bins=100, label=key[-1])[0:2]

    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    x = linspace(min(bincenters), max(bincenters) , 300)
    spl = make_interp_spline(bincenters, n, k=3)
    power_smooth = spl(x)
    plt.plot(x, power_smooth, c="black", linewidth=1)
    plt.legend()

    plt.xlabel("Log10(Area) (pixel2)")
    plt.ylabel("Counts")

plt.savefig(rf"{os.getcwd()}\statistics distance", dpi=1200)
plt.show()