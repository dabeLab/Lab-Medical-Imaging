import os
from imaging.segmentation.mealworm_optical_image.tools.classes import SegmentationWithSAM

IMAGE_PATH = "T:/data raw/optical images/batch01"
FILES = [rf"{IMAGE_PATH}/{x}" for x in os.listdir(IMAGE_PATH) if x.endswith(".jpg") or x.endswith(".png")]

for FILE in FILES:
    print(f'Segmenting {FILE}')
    sam = SegmentationWithSAM()
    sam.load_image(FILE)
    sam.model_type = 'vit_h'
    sam.points_per_batch = 128
    sam.remove_trasparency_from_png_images()
    sam.resize_image()
    sam.characteristic_dimension = 40
    sam.path_checkpoint = r"D:/mealworm/sam models"
    sam.path_output = r"D:/mealworm/data processed/optical images/segmentation sam/batch02"
    sam.slice_image()
    sam.run_sam_on_sliced_image()
    sam.thresh_area = (40, 10000)
    sam.filter_labels_by_shape()
    sam.save_attributes_to_disc()
    sam.save_figure_label_overlay_to_disc()
    print("Done\n\n")