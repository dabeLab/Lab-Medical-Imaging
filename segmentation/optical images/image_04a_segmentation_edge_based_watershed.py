from skimage import color, io
from image_utils import *
from numpy import array, savetxt
import pandas as pd

plot = False
os.chdir(r"C:\tierspital")  # set current working directory
files = os.listdir(rf"{os.getcwd()}\data raw\photos")
matplotlib.rcParams.update({'font.size': 8})

"""SEGMENTATION VIA EDGE-BASED RECOGNITION - THE WATERSHED ALGORITHM"""
image = io.imread(rf"{os.getcwd()}\data raw\photos\01.jpg")
image_gray = color.rgb2gray(image)
n_iteration_dilation = 2
n_iteration_erosion = n_iteration_dilation
structuring_element_dilation = square(2)
structuring_element_erosion = square(2)
threshold = (0, 0.4)
filter_area = [(20, inf)]
distances = [10, 20, 50, 100]
footprints = [square(x) for x in [10, 25, 50, 100]]
for idx, file in enumerate(files):
    columns = ["image", "algorithm", "contrast", "dilation", "erosion", "distance", "footprint", "label", "area"]
    df = pd.DataFrame(data=None, index=None, columns=columns)
    for idx2, distance in enumerate(distances):
        for idx3, footprint in enumerate(footprints):
            print(f"{file} - segmentation - edge watershed - filter area {filter_area} - distance {distance} - footprint {footprint.shape}... ", end="")
            labels, image_label_overlay = segmentation_edge_watershed(image=image_gray,
                                                                      threshold=threshold,
                                                                      n_dilation=n_iteration_dilation,
                                                                      n_erosion=n_iteration_erosion,
                                                                      structuring_element_dilation=structuring_element_dilation,
                                                                      structuring_element_erosion=structuring_element_erosion,
                                                                      area_thresh=filter_area,
                                                                      peaks_min_distance=distance,
                                                                      peaks_rel_thresh=0.1,
                                                                      peaks_footprint=footprint)
            print("Done.")
            plt.figure(idx)
            plt.imshow(image_label_overlay)
            plt.axis("off")
            plt.title(f"Segments {len(unique(labels))}")
            plt.savefig(rf"{os.getcwd()}\data processed\photos\segmentation watershed\{file[0:-4]} - contrast {threshold} - dilation {(structuring_element_dilation.shape, n_iteration_dilation)} - erosion {(structuring_element_erosion.shape, n_iteration_erosion)} - filter area {filter_area} - distance {distance} - footprint {footprint.shape}.jpg", dpi=1200)

            data = regionprops(labels)
            rows = []
            for i in range(len(data)):
                row = {"image": file, "algorithm": "watershed", "contrast": threshold,
                       "dilation": (structuring_element_dilation.shape, n_iteration_dilation),
                       "erosion": (structuring_element_erosion.shape, n_iteration_erosion),
                       "distance": distance, "footprint": footprint.shape,
                       "label": data[i].label, "area": data[i].area, "centroid": data[i].centroid}
                rows.append(row)
            df_temp = pd.DataFrame.from_dict(rows)
            df = pd.concat((df, df_temp), ignore_index=True)

    df.to_csv(rf"{os.getcwd()}\data processed\photos\segmentation watershed\{file[0:-4]} - statistics - segmentation watershed.csv", sep=",", index=False)