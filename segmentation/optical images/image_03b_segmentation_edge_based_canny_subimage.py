from skimage import color, io
from image_utils import *
import pandas as pd
import itertools
from numpy import mean, std

plot = True
os.chdir(r"C:\tierspital")  # set current working directory
files = os.listdir(rf"{os.getcwd()}\data raw\photos")
matplotlib.rcParams.update({'font.size': 10})
#columns = ["image", "algorithm", "threshold", "contrast", "structuring element", "area mean", "area std", "area min", "area q25", "area q50", "area q75", "area max"]
columns = ["image", "algorithm", "threshold", "contrast", "structuring element", "label", "area"]
df = pd.DataFrame(data=None, index=None, columns=columns)
thresholds = linspace(0.4, 0.4, 1)
structuring_elements = list(itertools.chain([None], [square(int(x)) for x in linspace(1, 1, 1)]))
filter_areas = [(20, inf)]
sigmas = linspace(0.5, 2, 7)

"""The segmentation cannot be done directly and solely via thresholding and/or from the histogram of gray values, 
because the background shares enough gray levels with the worms. Since every picture will have a different exposure 
gamma correction would be necessary to make two images comparable and therefore make possible to apply the same code  
A first idea is to take advantage of the local contrast, that is, to use the gradients rather than the gray values.
SEGMENTATION VIA EDGE-BASED ALGORITHM"""
image = io.imread(rf"{os.getcwd()}\data raw\photos\01.jpg")
image_gray = color.rgb2gray(image[400:701, 400:701])
for idx, file in enumerate(files[0:1]):
    for threshold in thresholds:
        for structuring_element in structuring_elements:
            for filter_area in filter_areas:
                for sigma in sigmas:
                    print(f"{file} - segmentation - edge canny - contrast: rescale intensity (0, {threshold:0.2f}) - structuring element: {None if structuring_element is None else structuring_element.shape} - filter area {filter_area} - sigma {sigma}... ", end="")
                    edge_map, labels, image_label_overlay = segmentation_edge_canny(image_gray, threshold=threshold, sigma=sigma, structuring_element=structuring_element, area_thresh=filter_area)
                    labels, areas = area_from_regionprops(labels)
                    for idx in range(len(labels)):
                        values = [file, "canny", threshold, "rescale intensity", {None if structuring_element is None else structuring_element.shape}, labels[idx], areas[idx]]
                        df = pd.concat([df, pd.Series(dict(zip(columns, values))).to_frame().T], ignore_index=True)
                    # stats = mean(label_area), std(label_area), min(label_area), quantile(label_area, 0.25), quantile(label_area, 0.5), quantile(label_area, 0.75), max(label_area)
                    # values = list(itertools.chain(*[[file, "canny", threshold, "rescale intensity", {None if structuring_element is None else structuring_element.shape}, len(unique(labels))], stats]))
                    # df = pd.concat([df, pd.Series(dict(zip(columns, values))).to_frame().T], ignore_index=True)
                    if plot is True:
                        fig, ax = plt.subplots(1, 1)
                        ax.set_title(f"Segmented n: {len(unique(labels))}")
                        ax.axis("off")
                        ax.imshow(image_label_overlay)
                        fig.savefig(rf"{os.getcwd()}\data processed\photos\{file[0:-4]}_sub - segmentation - edge canny - contrast rescale intensity (0, {threshold:0.2f}) - structuring element {None if structuring_element is None else structuring_element.shape} - filter area {filter_area} - sigma {sigma}.jpg", dpi=1200)
                        plt.close(fig)
                    print("Done.")

df.to_csv(rf"{os.getcwd()}\data processed\photos\all sub - statistics - segmentation canny.csv", sep=",", index=False)