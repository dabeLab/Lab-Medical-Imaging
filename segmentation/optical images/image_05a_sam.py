import torch
import matplotlib.pyplot as plt
import os
from numpy import inf, unique, array_equal, ceil, floor, uint8, array
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage.exposure import equalize_hist, rescale_intensity
from skimage import io
import pickle
from skimage.morphology import dilation, erosion, square
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from skimage.color import label2rgb
from image_utils import labels_from_sam_masks, show_anns
import time

DATA_PATH = r"C:\tierspital\data raw\photos"
OUTPUT_PATH = r"C:\tierspital\data processed\photos\segmentation sam"
CHECKPOINT_PATH = r"C:\tierspital\sam models"
DICT_CHECKPOINT = {"vit_h": "sam_vit_h_4b8939.pth", "vit_b": "sam_vit_b_01ec64.pth", "vit_l": "sam_vit_l_0b3195.pth",  }
files = [x for x in os.listdir(DATA_PATH) if x.endswith(".jpg")]
IMAGE_PATH = rf"{DATA_PATH}\{files[0]}"
n_dilation = 0
n_erosion = 0
structuring_element_dilation = square(1)
structuring_element_erosion = square(1)
threshold = 0.4
area_thresh = (20, inf)

image = io.imread(IMAGE_PATH)
full_image = image
crop_small = image[400:601, 400:601]
crop_medium = image[200:801, 200:801]
crop_large = image[0:1001, 0:1001]

"""The SAM model can be loaded with 3 different encoders: ViT-B, ViT-L, and ViT-H. 
ViT-H improves substantially over ViT-B but has only marginal gains over ViT-L. 
These encoders have different parameter counts, with ViT-B having 91M, ViT-L having 308M, 
and ViT-H having 636M parameters. This difference in size also influences the speed of inference."""

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for image in [crop_small, crop_medium, crop_large, full_image]:
for image in [crop_small]:
    if array_equal(image, crop_small):
        image_name = "crop small"
    elif array_equal(image, crop_medium):
        image_name = "crop medium"
    elif array_equal(image, crop_large):
        image_name = "crop large"
    elif array_equal(image, full_image):
        image_name = "full image"
    """Features are nicely segmented when their characheristic length is about 
    1/10th of image size. It comes therefore necessary to run SAM on image crops 
    that ensure that the condition on the ratio is met."""
    image_lx = image.shape[0]
    image_ly = image.shape[1]
    image_dx = (1-0)/image_lx
    image_dy = (1-0)/image_ly
    aspect_ratio = image_ly / image_lx
    print(f"Aspect ratio {aspect_ratio:.1f}, "
          f"Lenght X: {image_lx} px, Delta X: {image_dx:.4f} px, "
          f"Length Y: {image_ly} px, Delta Y: {image_dy:.4f} px, ", end="")
    feature_to_image_ratio_x = 25 / image_lx
    feature_to_image_ratio_y = 25 / image_ly
    n_crops = int(ceil(max(1/feature_to_image_ratio_x, 1/feature_to_image_ratio_y)))
    n_crops = 0
    print(f"N. crops: {n_crops}")

    for model in [x for x in DICT_CHECKPOINT.keys()]:
        MODEL_TYPE = model
        sam = sam_model_registry[MODEL_TYPE](checkpoint=f"{CHECKPOINT_PATH}/{DICT_CHECKPOINT[MODEL_TYPE]}")
        sam.to(device=DEVICE)
    #for model in ["vit_b"]:
        for m in 32*array([1, 2, 4], dtype=uint8):
        #for m in 32*array([1], dtype=uint8):
            for n in [1, 2, 4]:
                print(f"{image_name}, size {image.shape}, model {MODEL_TYPE}, points per side {m}, crop lavers {n_crops}, downscale factor {n}, ", end="")
                #MODEL_TYPE = "vit_b"
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=m, # default is 32
                    points_per_batch=128,  # the higher the more GPU memory required, default is 64
                    # pred_iou_thresh=0.88,
                    # stability_score_thresh=0.95,
                    # stability_score_offset= 1.0,
                    # box_nms_thresh: float = 0.7,
                    crop_n_layers=n_crops, # If >0, mask prediction will be run again on crops of the image.
                    # crop_nms_thresh: float = 0.7,
                    # crop_overlap_ratio: float = 512 / 1500,
                    crop_n_points_downscale_factor=n,
                    # point_grids = ,
                    # min_mask_region_area: int = 20,  # Requires open-cv to run post-processing
                    # output_mode: str = "binary_mask"
                )
                masks = mask_generator.generate(image)
                labels = labels_from_sam_masks(image, masks)
                labels = clear_border(labels=labels, bgval=0)
                if area_thresh is not None and isinstance(area_thresh, tuple):
                    for region in regionprops(labels):
                        if region.area < area_thresh[0] or region.area > area_thresh[1]:
                            labels[labels == region.label] = 0
                print(f"segments {len(unique(labels))}")
                FILENAME = f"{OUTPUT_PATH}/{image_name}, model {MODEL_TYPE}, points per side {m}, crop lavers {n_crops}, downscale factor {n}, segments {len(unique(labels))}"
                # print(f"{image_name}, size {image.shape}, model {MODEL_TYPE}, points per side {m}, crop lavers {n_crops}, downscale factor {n}, segments {len(unique(labels))}")

                plt.figure()
                image_label_overlay = label2rgb(labels, image, alpha=0.3, bg_label=0, bg_color=None, kind="overlay", saturation=0.6)
                plt.imshow(image_label_overlay)
                plt.axis("off")
                plt.tight_layout()

                plt.savefig(f"{FILENAME}.png", bbox_inches="tight")
                plt.close()
                with open(f"{FILENAME}.dat", "wb") as writer:
                    pickle.dump(labels, writer)