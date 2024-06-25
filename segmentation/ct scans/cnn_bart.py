from datetime import datetime
import glob
import os
import re

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    CropForegroundd,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Invertd,
    Orientationd,
    RandFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RandCropByLabelClassesd,
    RandScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SaveImaged,
    RandRotate90d,
)

from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
)

import torch


# Set parameters, create and organise different datasets
params = {'experiment_name': {},
          'patch_size': (128, 128, 128),
          'bz_size': 1,
          'n_classes': 2,
          'class_ratios': [1, 10],
          'learning_rate': 1e-3,
          'weight_decay': 1e-4,
          'epoch_num': 1000,
          'val_interval': 5,
          'intensity_min': -1000,
          'intensity_max': 1000,
          'seed': 0,
          'train_set': {},
          'val_set': {},
          'test_set': {},
          'train_transforms': {},
          'val_transforms': {},
          'loss_function': DiceCELoss(to_onehot_y=True, sigmoid=True)
          }

now = datetime.now().strftime("%Y%m%d_%H%M")[2:]

params['experiment_name'] = "worms_{}_bz{}_patch{}_lr{:.0e}_e{}".format(
    now,
    params['bz_size'],
    params["patch_size"][0],
    params["learning_rate"],
    params["epoch_num"]
)

root_dir = os.getcwd()
data_dir = os.path.join(r"D:\Meelworms\data\worms")

files = {}
image_sets = ['train', 'validation', 'test']
files[image_sets[0]] = {}
files[image_sets[1]] = {}
files[image_sets[2]] = {}

set_files = {}

for nn_set in image_sets:
    files[nn_set]['image'] = sorted(glob.glob(os.path.join(data_dir, nn_set, "images", "*.nii.gz")))[:]
    files[nn_set]['label'] = sorted(glob.glob(os.path.join(data_dir, nn_set, "labels", "*.nii.gz")))[:]

    set_files[nn_set] = [{"image": image_name, "label": label_name} for image_name, label_name in
                         zip(files[nn_set]['image'], files[nn_set]['label'])]

index = '1'
while True:
    try:
        os.makedirs(os.path.join(root_dir, 'experiments', '{}-{}'.format(params['experiment_name'], index)))
        params['experiment_name'] = '{}-{}'.format(params['experiment_name'], index)
        break
    except FileExistsError:
        if index:
            index = str(int(index) + 1)
        else:
            index = '1'
        pass

if not os.path.exists(os.path.join(root_dir, 'experiments', '{}'.format(params['experiment_name']), 'predictions')):
    os.mkdir(os.path.join(root_dir, 'experiments', '{}'.format(params['experiment_name']), 'predictions'))

params['train_set'] = [re.search(r'images\\crop_(.+?)', x) for x in files['train']['image']]
params['val_set'] = [re.search(r'images\\crop_(.+?)', x) for x in files['validation']['image']]
params['test_set'] = [re.search(r'images\\crop_(.+?)', x) for x in files['test']['image']]

# Define Monai transforms and store in experiment folder

train_transformations = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(0.3, 0.3, 0.4), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=params['intensity_min'], a_max=params['intensity_max'], b_min=0.0, b_max=1.0, clip=True),
    RandCropByLabelClassesd(
        keys=["image", "label"],
        label_key="label",
        spatial_size=params['patch_size'],
        ratios=params['class_ratios'],
        num_classes=params['n_classes'],
        num_samples=4,
        image_threshold=0,
    ),
    RandZoomd(
        keys=["image", "label"],
        min_zoom=0.9,
        max_zoom=1.2,
        mode=("trilinear", "nearest"),
        align_corners=(True, None),
        prob=0.16,
    ),
    RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
    RandGaussianSmoothd(
        keys=["image"],
        sigma_x=(0.8, 1.15),
        sigma_y=(0.8, 1.15),
        sigma_z=(0.8, 1.15),
        prob=0.15,
    ),
    RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
    RandFlipd(["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
]

val_transformations = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(0.3, 0.3, 0.4), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=params['intensity_min'], a_max=params['intensity_max'], b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
]

train_transforms = Compose(train_transformations)
val_transforms = Compose(val_transformations)

params['train_transforms'] = [vars(x) for x in train_transformations]
params['val_transforms'] = [vars(x) for x in val_transformations]

with open(os.path.join(root_dir, 'experiments', '{}'.format(params['experiment_name']), 'parameters.txt'), 'w') as file:
    print(params, file=file)

# Load training and validation dataset
train_ds = CacheDataset(data=set_files['train'], transform=train_transforms, cache_rate=1.0, num_workers=0)
train_loader = DataLoader(train_ds, batch_size=params['bz_size'], shuffle=True, num_workers=0)

val_ds = CacheDataset(data=set_files['validation'], transform=val_transforms, cache_rate=1.0, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

# Initiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = UNETR(
    in_channels=1,
    out_channels=params['n_classes'],
    img_size=params['patch_size'],
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, params['patch_size'], 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [eval_num * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.show()
    plt.savefig(os.path.join(root_dir, 'experiments', '{}'.format(params['experiment_name']),
                             "loss.png"))
    plt.close()

    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = params['loss_function'](logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(),
                           os.path.join(root_dir, 'experiments', '{}'.format(params['experiment_name']),
                                        "best_during_training.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1

    return global_step, dice_val_best, global_step_best


max_iterations = params['epoch_num']
eval_num = 10
post_label = AsDiscrete(to_onehot=params['n_classes'])
post_pred = AsDiscrete(argmax=True, to_onehot=params['n_classes'])
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
