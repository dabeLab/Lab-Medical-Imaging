import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import csv


path_main = "E:/gd_synthesis"
path_scores = sorted(glob.glob(os.path.join(path_main, "model", "*scores.csv")))
path_losses = sorted(glob.glob(os.path.join(path_main, "model", "*losses.csv")))
path_attrib = sorted(glob.glob(os.path.join(path_main, "model", "*attributes.csv")))
paths = [{"losses": path1, "scores": path2, "attributes": path3} for path1, path2, path3 in zip(path_losses, path_scores, path_attrib)]


for path in paths:
    loss = np.loadtxt(path["losses"], delimiter=",", skiprows=1)
    scor = np.loadtxt(path["scores"], delimiter=",", skiprows=1)
    attr = {}
    with open(path["attributes"], 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the header
        data = next(reader)    # Read the first row of data
        for i in range(len(header)):
            attr[header[i]] = data[i]

    plt.figure(0)
    plt.plot(loss[:, 0], loss[:, 1], marker="o", linewidth=0, markersize=6, alpha=0.7, label=attr["channels"])
    plt.figure(1)
    scor = scor[scor[:, 1] != 0]
    plt.plot(scor[:, 0], scor[:, 1], marker="o", linewidth=0, markersize=6, alpha=0.7, label=attr["channels"])

plt.figure(0)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.semilogy()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(path_main, "results", "loss vs epoch.png"), dpi=600)

plt.figure(1)
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.semilogy()
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(path_main, "results", "score vs epoch.png"), dpi=600)

