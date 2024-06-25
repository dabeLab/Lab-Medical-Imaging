import os
import csv
from skimage import color, io
from image_utils import *
from numpy import inf
import pandas as pd
import pickle

os.chdir(r"C:\tierspital")  # set current working directory
files = os.listdir(rf"{os.getcwd()}\data raw\photos")
threshold = 0.4
n_bins=50
area_thresh = (20, inf)
n_iteration_dilation = 1
n_iteration_erosion = 1
structuring_element_dilation = square(2)
structuring_element_erosion = square(2)
columns = ["image", "label", "area", "centroid"]

"""SEGMENTATION BY THRESHOLD. All gray pictures show histograms with a singluar feature around 0.4.
Let's try to mask out everything with greay value > 0.4 and try all thresholding algorithm provided by the library, which include:"""
# region ----- THRESHOLD GLOBAL MEAN -----
algorithm = "threshold global mean"
df = pd.DataFrame(data=None, index=None, columns=columns)
path = rf"{os.getcwd()}\data processed\photos\segmentation threshold global mean"
for idx, file in enumerate(files):
    print(f"figure {file} segmentation - {algorithm} - ...", end="")
    image = io.imread(rf"{os.getcwd()}\data raw\photos\{file}")
    image_gray = color.rgb2gray(image)
    mask, labels, image_overlay = segmentation_threshold_mean_steps(image_gray, structuring_element_dilation, structuring_element_erosion, n_iteration_erosion, n_iteration_dilation, area_thresh)
    df_file = stats(file, labels)
    df = pd.concat((df, df_file), ignore_index=True)
    plot_segmentation_steps(rf"{path}\{file[:-4]} - segmentation.jpg", image, mask, labels, image_overlay)
    print("Done.")
df.to_csv(rf"{path}\data.csv", sep=",", index=False)
data = {"info": {"contrast enhancement": None,
                 "structuring_element_dilation": structuring_element_dilation,
                 "structuring_element_erosion": structuring_element_erosion,
                 "n_iteration_dilation": n_iteration_dilation,
                 "n_iteration_erosion": n_iteration_erosion,
                 "area_thresh": area_thresh},
        "data": df}
with open(rf"{path}\data.dat", "wb") as fp:
    pickle.dump(data, fp)
#endregion

#region ----- THRESHOLD GLOBAL VALUE -----
algorithm = "threshold global value"
df = pd.DataFrame(data=None, index=None, columns=columns)
path = rf"{os.getcwd()}\data processed\photos\segmentation threshold global value"
for idx, file in enumerate(files):
    print(f"figure {file} segmentation - {algorithm} {threshold} - ...", end="")
    image = io.imread(rf"{os.getcwd()}\data raw\photos\{file}")
    image_gray = color.rgb2gray(image)
    mask, labels, image_overlay = segmentation_threshold_value_steps(image_gray, threshold, structuring_element_dilation, structuring_element_erosion, n_iteration_dilation, n_iteration_erosion, area_thresh)
    df_file = stats(file, labels)
    df = pd.concat((df, df_file), ignore_index=True)
    plot_segmentation_steps(rf"{path}\{file[:-4]} - segmentation - threshold {threshold}.jpg", image, mask, labels, image_overlay)
    print("Done.")
df.to_csv(rf"{path}\data.csv", sep=",", index=False)
data = {"info": {"contrast enhancement": (0, threshold),
                 "structuring_element_dilation": structuring_element_dilation,
                 "structuring_element_erosion": structuring_element_erosion,
                 "n_iteration_dilation": n_iteration_dilation,
                 "n_iteration_erosion": n_iteration_erosion,
                 "area_thresh": area_thresh},
        "data": df}
with open(rf"{path}\data.dat", "wb") as fp:
    pickle.dump(data, fp)
#endregion

# region ----- THRESHOLD GLOBAL MINIMUM -----
algorithm = "threshold global minimum"
df = pd.DataFrame(data=None, index=None, columns=columns)
path = rf"{os.getcwd()}\data processed\photos\segmentation threshold global minimum"
for idx, file in enumerate(files):
    print(f"figure {file} segmentation - {algorithm} - ...", end="")
    image = io.imread(rf"{os.getcwd()}\data raw\photos\{file}")
    image_gray = color.rgb2gray(image)
    mask, labels, image_overlay = segmentation_threshold_minimum_steps(image_gray, structuring_element_dilation, structuring_element_erosion, n_iteration_dilation, n_iteration_erosion, area_thresh)
    df_file = stats(file, labels)
    df = pd.concat((df, df_file), ignore_index=True)
    plot_segmentation_steps(rf"{path}\{file[:-4]} - segmentation - minimum.jpg", image, mask, labels, image_overlay)
    print("Done.")
df.to_csv(rf"{path}\data.csv", sep=",", index=False)
data = {"info": {"contrast enhancement": None,
                 "structuring_element_dilation": structuring_element_dilation,
                 "structuring_element_erosion": structuring_element_erosion,
                 "n_iteration_dilation": n_iteration_dilation,
                 "n_iteration_erosion": n_iteration_erosion,
                 "area_thresh": area_thresh},
        "data": df}
with open(rf"{path}\data.dat", "wb") as fp:
    pickle.dump(data, fp)
# endregion

# region ----- THRESHOLD GLOBAL OTSU -----
algorithm = "threshold global otsu"
df = pd.DataFrame(data=None, index=None, columns=columns)
path = rf"{os.getcwd()}\data processed\photos\segmentation threshold global otsu"
for idx, file in enumerate(files):
    print(f"figure {file} segmentation - {algorithm} - ...", end="")
    image = io.imread(rf"{os.getcwd()}\data raw\photos\{file}")
    image_gray = color.rgb2gray(image)
    mask, labels, image_overlay = segmentation_threshold_otsu_steps(image_gray, structuring_element_dilation, structuring_element_erosion, n_iteration_dilation, n_iteration_erosion, area_thresh)
    df_file = stats(file, labels)
    df = pd.concat((df, df_file), ignore_index=True)
    plot_segmentation_steps(rf"{path}\{file[:-4]} - segmentation - otsu.jpg", image, mask, labels, image_overlay)
    print("Done.")
df.to_csv(rf"{path}\data.csv", sep=",", index=False)
data = {"info": {"contrast enhancement": None,
                 "structuring_element_dilation": structuring_element_dilation,
                 "structuring_element_erosion": structuring_element_erosion,
                 "n_iteration_dilation": n_iteration_dilation,
                 "n_iteration_erosion": n_iteration_erosion,
                 "area_thresh": area_thresh},
        "data": df}
with open(rf"{path}\data.dat", "wb") as fp:
    pickle.dump(data, fp)
# endregion

# region ----- THRESHOLD LOCAL GAUSSIAN -----
algorithm = "local threshold gaussian"
df = pd.DataFrame(data=None, index=None, columns=columns)
path = rf"{os.getcwd()}\data processed\photos\segmentation threshold local gaussian"
for idx, file in enumerate(files):
    print(f"gigure {file} segmentation - {algorithm} - ...", end="")
    image = io.imread(rf"{os.getcwd()}\data raw\photos\{file}")
    image_gray = color.rgb2gray(image)
    mask, labels, image_overlay = segmentation_threshold_local_steps(image_gray, structuring_element_dilation, structuring_element_erosion, n_iteration_dilation, n_iteration_erosion, area_thresh, 35, 0)
    df_file = stats(file, labels)
    df = pd.concat((df, df_file), ignore_index=True)
    plot_segmentation_steps(rf"{path}\{file[:-4]} - segmentation - local gaussian.jpg", image, mask, labels, image_overlay)
    print("Done.")
df.to_csv(rf"{path}\data.csv", sep=",", index=False)
data = {"info": {"contrast enhancement": None,
                 "structuring_element_dilation": structuring_element_dilation,
                 "structuring_element_erosion": structuring_element_erosion,
                 "n_iteration_dilation": n_iteration_dilation,
                 "n_iteration_erosion": n_iteration_erosion,
                 "area_thresh": area_thresh,
                 "block size": 35,
                 "offset": 0},
        "data": df}
with open(rf"{path}\data.dat", "wb") as fp:
    pickle.dump(data, fp)
# endregion
