from skimage import color, io
from image_utils import *
import pandas as pd
import itertools
import pickle

plot = False
os.chdir(r"C:\tierspital")  # set current working directory
files = os.listdir(rf"{os.getcwd()}\data raw\photos")
path = rf"{os.getcwd()}\data processed\photos\segmentation edge canny"
matplotlib.rcParams.update({'font.size': 10})
columns = ["image", "algorithm", "threshold", "contrast", "structuring element", "label", "area"]
df = pd.DataFrame(data=None, index=None, columns=columns)
thresholds = linspace(0.2, 0.6, 5)
structuring_elements = list(itertools.chain([None], [square(int(x)) for x in [2]]))
n_dilation = 1
n_erosion = 1
filter_areas = [(20, inf)]
sigmas = linspace(0.5, 2, 7)

"""SEGMENTATION VIA EDGE-BASED ALGORITHM"""
image = io.imread(rf"{os.getcwd()}\data raw\photos\01.jpg")
image_gray = color.rgb2gray(image)
for idx, file in enumerate(files[0:1]):
    for threshold in thresholds:
        for structuring_element in structuring_elements:
            for filter_area in filter_areas:
                for sigma in sigmas:
                    print(f"{file} - segmentation - edge canny - contrast: rescale intensity (0, {threshold:0.2f}) - structuring element: {None if structuring_element is None else structuring_element.shape} - filter area {filter_area} - sigma {sigma}... ", end="")
                    edge_map, labels, image_label_overlay = segmentation_edge_canny(image_gray, threshold, sigma, structuring_element, structuring_element, n_dilation, n_erosion, area_thresh=filter_area)
                    df = stats(labels)
                    plot_segmentation_steps(rf"{path}\{file[:-4]} - threshold {threshold} - structuring element {None if structuring_element is None else structuring_element.shape} - filter area {filter_area} - sigma {sigma}.jpg", image, edge_map, labels, image_label_overlay)
                    df.to_csv(rf"{path}\{file[:-4]} - threshold {threshold} - structuring element {None if structuring_element is None else structuring_element.shape} - filter area {filter_area} - sigma {sigma}.csv", sep=",", index=False)
                    data = {"image": file,
                            "threshold": threshold,
                            "contrast enhancement": (0, threshold),
                            "structuring element dilation": None if structuring_element is None else structuring_element.shape,
                            "structuring element erosion": None if structuring_element is None else structuring_element.shape,
                            "n iteration dilation": 1,
                            "n iteration erosion": 1,
                            "area thresh": filter_area,
                            "sigma": sigma,
                            "data": df}
                    with open(rf"{path}\{file[:-4]} - threshold {threshold} - structuring element {None if structuring_element is None else structuring_element.shape} - filter area {filter_area} - sigma {sigma}.dat", "wb") as fp:
                        pickle.dump(data, fp)
                    if plot is True:
                        fig, ax = plt.subplots(1, 3, figsize=(30/2.54, 10/2.54))
                        ax[0].set_title(f"Original")
                        ax[0].axis("off")
                        ax[0].imshow(image_gray, cmap=plt.cm.gray)
                        ax[1].set_title(f"Edge map")
                        ax[1].axis("off")
                        ax[1].imshow(edge_map, cmap=plt.cm.gray)
                        ax[2].set_title(f"Segmented n: {len(unique(labels))}")
                        ax[2].axis("off")
                        ax[2].imshow(image_label_overlay)
                        fig.savefig(rf"{path}\{file[0:-4]} - segmentation - edge canny - threshold {threshold} - structuring element {None if structuring_element is None else structuring_element.shape} - filter area {filter_area} - sigma {sigma}.jpg", dpi=1200)
                        plt.close(fig)
                    print("Done.")