import numpy
import pickle
from skimage import color, io
from image_utils import *
from numpy import inf, array
import pandas as pd
from skimage.morphology import closing, square
import itertools
import matplotlib.colors

"""SEGMENT IMAGES BY THRESHOLD vs THRESHOLD  VALUE vs STRUCTURING ELEMENT SIZE.
Segment images by threshold value and plot the results as a function of 
threshold value and strucuturing element size. Plot the feature size distribution on a 2D images."""

os.chdir(r"C:\tierspital")  # set current working directory
files = os.listdir(rf"{os.getcwd()}\data raw\photos")
path = rf"{os.getcwd()}\data processed\photos\segmentation threshold global value vs value"
thresholds = linspace(0.2, 0.6, 5)
structuring_elements = [square(int(x)) for x in linspace(2, 10, 10)]
structuring_elements = list(itertools.chain([None], structuring_elements))
filter_areas = [(x, inf) for x in [0, 10, 20, 50, 100]]
columns = ["label", "area", "centroid"]
df = pd.DataFrame(data=None, index=None, columns=columns)
matplotlib.rcParams.update({'font.size': 12})
data = numpy.ones((len(thresholds), len(structuring_elements), len(filter_areas)))

"""SEGMENTATION BY THRESHOLD. All gray pictures show histograms with a singluar feature around 0.4.
Let's try to mask out everything with greay value > 0.4 and try all thresholding algorithm provided by the library, which include:"""
for idx, file in enumerate(files[2:]):
    image = io.imread(rf"{os.getcwd()}\data raw\photos\{file}")
    image_gray = color.rgb2gray(image)
    for threshold in thresholds:
        for structuring_element in structuring_elements:
            for filter_area in filter_areas:
                print(f"{file} - segmentation - threshold value: {threshold:0.2f} - structuring element: {None if structuring_element is None else structuring_element.shape} - filter area {filter_area}... ", end="")
                mask, labels, image_overlay = segmentation_threshold_value_steps(image_gray, threshold, structuring_element, structuring_element, 1, 1, filter_area)
                df = stats(labels)
                plot_segmentation_steps(rf"{path}\{file[:-4]} - threshold {threshold} - structuring element {None if structuring_element is None else structuring_element.shape} - filter area {filter_area}.jpg", image, mask, labels, image_overlay)
                df.to_csv(rf"{path}\{file[:-4]} - threshold {threshold} - structuring element {None if structuring_element is None else structuring_element.shape} - filter area {filter_area}.csv", sep=",", index=False)
                data = {"image": file,
                        "threshold": threshold,
                        "contrast enhancement": (0, threshold),
                        "structuring element dilation": None if structuring_element is None else structuring_element.shape,
                        "structuring element erosion": None if structuring_element is None else structuring_element.shape,
                        "n iteration dilation": 1,
                        "n iteration erosion": 1,
                        "area thresh": filter_area,
                        "data": df}
                with open(rf"{path}\{file[:-4]} - threshold {threshold} - structuring element {None if structuring_element is None else structuring_element.shape} - filter area {filter_area}.dat", "wb") as fp:
                    pickle.dump(data, fp)
                print("Done.")
