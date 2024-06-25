import itertools

import numpy
from skimage import color, io
from image_utils import *
from numpy import inf, array
import pandas as pd
from skimage.morphology import closing, square
from skimage.exposure import equalize_hist, rescale_intensity, histogram
import itertools
import matplotlib.colors

"""SEGMENT IMAGES BY THRESHOLD vs THRESHOLD  VALUE vs STRUCTURING ELEMENT SIZE.
Segment images by threshold value and plot the results as a function of 
threshold value and strucuturing element size. Plot the feature size distribution on a 2D images."""

os.chdir(r"C:\tierspital")  # set current working directory
files = os.listdir(rf"{os.getcwd()}\data raw\photos")
thresholds = linspace(0.2, 0.6, 5)
structuring_elements = list(itertools.chain([None], [square(int(x)) for x in linspace(1, 10, 10)]))
filter_areas = [(20, inf)]
df = pd.DataFrame(data=None, index=None, columns=["image", "algorithm", "threshold", "contrast", "structuring element", "area", "segments"])
matplotlib.rcParams.update({'font.size': 12})
data = numpy.ones((len(thresholds), len(structuring_elements), len(filter_areas)))

"""SEGMENTATION BY THRESHOLD. All gray pictures show histograms with a singluar feature around 0.4.
Let's try to mask out everything with greay value > 0.4 and try all thresholding algorithm provided by the library, which include:"""
for idx, file in enumerate(files):
    for threshold in thresholds:
        for structuring_element in structuring_elements:
            for filter_area in filter_areas:
                image = io.imread(rf"{os.getcwd()}\data raw\photos\{file}")
                image = color.rgb2gray(image)
                fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(25/2.54, 15/2.54))

                print(f"{file} - segmentation - threshold value: {threshold:0.2f} - contrast: None - structuring element: {None if structuring_element is None else structuring_element.shape} - filter area {filter_area}... ", end="")
                hist, bin_centers = histogram(image)
                mask = closing(image < threshold, structuring_element)
                labels, num = label(label_image=mask, background=0, return_num=True, connectivity=1)
                labels = clear_border(labels=labels, bgval=0)
                if filter_area is not None and isinstance(filter_area, tuple):
                    for region in regionprops(labels):
                        if region.area < filter_area[0] or region.area > filter_area[1]:    # if area outside passed range
                            labels[labels == region.label] = 0                              # set the label equal to background (zero)
                image_label_overlay = label2rgb(labels, image, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
                df = pd.concat([df, pd.Series({"image": file, "algorithm": "global threshold value", "threshold": threshold, "contrast": None, "structuring element": {None if structuring_element is None else structuring_element.shape}, "area": filter_area, "segments":len(unique(labels))}).to_frame().T], ignore_index=True)
                ax[0, 0].set_title(f"Segmented n: {len(unique(labels))}")
                ax[0, 0].imshow(image_label_overlay)
                ax[0, 0].axis("off")
                ax[1, 0].set_title("Hist.")
                ax[1, 0].plot(bin_centers, hist)
                print("Done.")

                print(f"{file} - segmentation - threshold value: {threshold:0.2f} - contrast: rescale intensity - structuring element: {None if structuring_element is None else structuring_element.shape} - filter area {filter_area}... ", end="")
                #image = equalize_hist(image=image, mask=image < threshold)
                #image = equalize_hist(image=image)
                image = rescale_intensity(image=image, in_range=(0, threshold))
                hist, bin_centers = histogram(image)
                mask = closing(image < 1, structuring_element)
                labels, num = label(label_image=mask, background=0, return_num=True, connectivity=1)
                labels = clear_border(labels=labels, bgval=0)
                if filter_area is not None and isinstance(filter_area, tuple):
                    for region in regionprops(labels):
                        if region.area < filter_area[0] or region.area > filter_area[1]:    # if area outside passed range
                            labels[labels == region.label] = 0                              # set the label equal to background (zero)
                image_label_overlay = label2rgb(labels, image, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
                df = pd.concat([df, pd.Series({"image": file, "algorithm": "global threshold value", "threshold": threshold, "contrast": "rescale intensity", "structuring element": {None if structuring_element is None else structuring_element.shape}, "area": filter_area, "segments":len(unique(labels))}).to_frame().T], ignore_index=True)
                ax[0, 1].set_title(f"Segmented n: {len(unique(labels))}")
                ax[0, 1].imshow(image_label_overlay)
                ax[0, 1].axis("off")
                ax[1, 1].set_title("Eq. Hist.")
                ax[1, 1].plot(bin_centers, hist)
                print("Done.")

                print(f"{file} - segmentation - threshold value: {threshold:0.2f} - contrast: eq. hist. - structuring element: {None if structuring_element is None else structuring_element.shape} - filter area {filter_area}... ", end="")
                image = equalize_hist(image=image, mask=image < threshold)
                #image = equalize_hist(image=image)
                # image = rescale_intensity(image=image, in_range=(0, 0.4))
                hist, bin_centers = histogram(image)
                mask = closing(image<1, structuring_element)
                labels, num = label(label_image=mask, background=0, return_num=True, connectivity=1)
                labels = clear_border(labels=labels, bgval=0)
                if filter_area is not None and isinstance(filter_area, tuple):
                    for region in regionprops(labels):
                        if region.area < filter_area[0] or region.area > filter_area[1]:    # if area outside passed range
                            labels[labels == region.label] = 0                              # set the label equal to background (zero)
                image_label_overlay = label2rgb(labels, image, alpha=0.5, bg_label=0, bg_color=None, kind="overlay")
                df = pd.concat([df, pd.Series({"image": file, "algorithm": "global threshold value", "threshold": threshold, "contrast": "rescale intensity", "structuring element": {None if structuring_element is None else structuring_element.shape}, "area": filter_area, "segments":len(unique(labels))}).to_frame().T], ignore_index=True)
                ax[0, 2].set_title(f"Segmented n: {len(unique(labels))}")
                ax[0, 2].imshow(image_label_overlay)
                ax[0, 2].axis("off")
                ax[1, 2].set_title("Eq. Hist.")
                ax[1, 2].plot(bin_centers, hist)
                print("Done.")

                fig.savefig(rf"{os.getcwd()}\data processed\photos\{file[0:-4]} - segmentation - threshold value {threshold:0.2f} - contrast all - structuring element {None if structuring_element is None else structuring_element.shape} - filter area {filter_area}.jpg", dpi=1200)
                plt.close(fig)

df.to_csv(rf"{os.getcwd()}\data processed\photos\all - statistics - segments vs contrast enchancement.csv", sep=",", index=False)
