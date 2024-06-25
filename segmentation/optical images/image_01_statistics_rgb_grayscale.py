from skimage import color, io
from image_utils import *
import random

os.chdir(r"C:\tierspital")  # set current working directory
files = os.listdir(rf"{os.getcwd()}\data raw\photos")
n_bins_rgb = 50
n_bins_greyscale = n_bins_rgb
n = random.randint(1, 14)  # select a random picture to display and process

"""IMAGE PIXELS STATISTICS. Plot RGB and grayscale images and pixel distributions of (i) a random image
and (ii) of the whole image set."""
fig1, ax1 = plt.subplots(1, 1, figsize=(15/2.54, 10/2.54))
ax1.set_title(f"Grayscale pixel statistics - n. images: {len(files)}")
for idx, file in enumerate(files):
    print(f"Processing {file}... ", end="")
    image = io.imread(rf"{os.getcwd()}\data raw\photos\{file}")
    image_gray = color.rgb2gray(image)
    stats = pixel_stats(image, image_gray, n_bins_rgb, n_bins_greyscale, plot=True, save=(True, file[:-4]))
    ax1.fill_between(stats["x_gray"], stats["y_gray"], y2=0, alpha=0.25, color="gray")
    ax1.plot(stats["x_gray"], stats["y_gray"], color="black")
    print("Done.")
fig1.savefig(rf"{os.getcwd()}\data processed\photos\all - statistics - rgb grayscale.jpg", dpi=1200)