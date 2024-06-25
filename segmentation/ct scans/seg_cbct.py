import SimpleITK as sitk
import os
from worm_seg_utils import WormCounter

morph_stats = {'id': [], 'max_length (mm)': []}

input_im = sitk.ReadImage(r'D:\Meelworms\data\CT_comparison\CBCT_Box2.nii.gz')
working_dir = r'.\working_dir_CBCT'

# input_im = sitk.ReadImage(r'../data/3_Lunge_box.nii.gz')
# working_dir = r'../data/working_dir_worm_full'

worm_counter = WormCounter(input_im, working_dir)

print('Setting filter')
no_filter = sitk.Cast(input_im*0+1, sitk.sitkUInt8)
# box_filter = worm_counter.box_filter(x_min=35, x_max=995, y_min=240, y_max=450, z_min=20, z_max=1010)
box_filter = no_filter

worms = worm_counter.thresh_and_clean(selection_filter=box_filter, threshold=300, minimal_size=10) # originally -500
print('Finding seeds of worms')
seeds_worm = worm_counter.thresh_and_clean(selection_filter=box_filter, threshold=700, minimal_size=0)
# sitk.WriteImage(seeds_worm, os.path.join(working_dir, 'worms_seeds1_seg.nii.gz'))
seeds_worm = worm_counter.divide(seeds_worm, cutting_size=1000, threshold_old=300, split_twins=50, min_object_sz=6)
# sitk.WriteImage(seeds_worm, os.path.join(working_dir, 'worms_seeds2_seg.nii.gz'))
seeds_worm = worm_counter.divide(seeds_worm, min_object_sz=6)
sitk.WriteImage(seeds_worm, os.path.join(working_dir, 'worms_seeds3_seg.nii.gz'))
print('Segmentation of worms')
worms_seg, worm_leftover = worm_counter.ws(worms, seeds_worm, radius=1, minimal_size=6)
worms_seg = worm_counter.combine(worms_seg, worm_leftover)

worms_done2, worms_seg2 = worm_counter.split_divide_combine(worms_seg, cut_limit=1400, threshold=500, minimal_size=11, minimal_size_ws=6)
worms_done3, worms_seg3 = worm_counter.split_divide_combine(worms_seg2, cut_limit=1800, threshold=600, minimal_size=5, minimal_size_ws=6)

worm_seg_final = worm_counter.combine(worm_counter.combine(worms_done2, worms_done3), worms_seg3)
sitk.WriteImage(worm_seg_final, os.path.join(working_dir, 'worms_seg.nii.gz'))

worm_seg, worm_seg_outliers = worm_counter.split(worm_seg_final, cut_limit=5500)

worm_seg = sitk.ConnectedComponent(worm_seg)
worm_seg_outliers = sitk.ConnectedComponent(worm_seg_outliers)
worm_seg_outliers = worm_counter.show_count(worm_seg_outliers, 5500, 1000000)
sitk.WriteImage(worm_seg_outliers, os.path.join(working_dir, 'worms_outliers_seg.nii.gz'))
print('Statistics calculation for the worms')
shape_stats, csv_file_name = worm_counter.perform_count(worm_seg, ground_truth=False, individual=None, name_csv="stats_worms.csv", name_png="stats_worms_plot.png")

print('Segmentation of the substrate')
box_bright_and_worms = sitk.Or(box_and_bright, worms)
substrate = worm_counter.substrate_remover(selection_filter=sitk.And(box_filter, sitk.Not(box_bright_and_worms)), box=box, box_filter=box_filter, threshold=-920, minimal_size=50)

print('Segmentation of the background')
background = sitk.Not(worm_counter.thresh_and_clean(selection_filter=no_filter, threshold=-920, minimal_size=0))
background = sitk.And(background, sitk.Not(box))
sitk.WriteImage(background, os.path.join(working_dir, 'background_seg.nii.gz'))
# example of splitting objects by some statistical parameter
seg1, seg2 = worm_counter.split_by_stats(worm_seg_final, stat_id=6, cut_limit=90)
sitk.WriteImage(seg1, os.path.join(working_dir, 'seg1_seg.nii.gz'))
sitk.WriteImage(seg2, os.path.join(working_dir, 'seg2_seg.nii.gz'))

## diagnostics of outliers
## this takes some time, the function split_by_stats can be sped up
#
## brights spots outliers
#bright_seg = sitk.ReadImage(r'./working_dir_worm_full/bright_seg.nii.gz')
#seg1, seg2 = worm_counter.split_by_stats(bright_seg, stat_id=0, cut_limit=20)
#sitk.WriteImage(seg2, os.path.join(working_dir, 'bright_outlier_size_seg.nii.gz'))
#seg1, seg2 = worm_counter.split_by_stats(bright_seg, stat_id=1, cut_limit=2)
#sitk.WriteImage(seg2, os.path.join(working_dir, 'bright_outlier_elon_seg.nii.gz'))
#seg1, seg2 = worm_counter.split_by_stats(bright_seg, stat_id=2, cut_limit=2)
#sitk.WriteImage(seg2, os.path.join(working_dir, 'bright_outlier_flat_seg.nii.gz'))
#
## worms outliers
#worms_seg = sitk.ReadImage(r'./working_dir_worm_full/worms_seg.nii.gz')
#worms_seg_out = sitk.ReadImage(r'./working_dir_worm_full/worms_outliers_seg.nii.gz')
#worms_seg = worm_counter.combine(worms_seg, worms_seg_out)
#seg1, seg2 = worm_counter.split_by_stats(worms_seg, stat_id=0, cut_limit=130)
#sitk.WriteImage(seg2, os.path.join(working_dir, 'worms_outlier_size_seg.nii.gz'))
#seg1, seg2 = worm_counter.split_by_stats(worms_seg, stat_id=1, cut_limit=7.5)
#sitk.WriteImage(seg2, os.path.join(working_dir, 'worms_outlier_elon_seg.nii.gz'))
#seg1, seg2 = worm_counter.split_by_stats(worms_seg, stat_id=2, cut_limit=2.6)
#sitk.WriteImage(seg2, os.path.join(working_dir, 'worms_outlier_flat_seg.nii.gz'))
#seg1, seg2 = worm_counter.split_by_stats(worms_seg, stat_id=3, cut_limit=6.5)
#sitk.WriteImage(seg2, os.path.join(working_dir, 'worms_outlier_box_min_seg.nii.gz'))
#seg1, seg2 = worm_counter.split_by_stats(worms_seg, stat_id=4, cut_limit=18)
#sitk.WriteImage(seg2, os.path.join(working_dir, 'worms_outlier_box_max_seg.nii.gz'))
#seg1, seg2 = worm_counter.split_by_stats(worms_seg, stat_id=5, cut_limit=-170)
#sitk.WriteImage(seg2, os.path.join(working_dir, 'worms_outlier_int_mean_max_seg.nii.gz'))
#seg1, seg2 = worm_counter.split_by_stats(worms_seg, stat_id=5, cut_limit=-500)
#sitk.WriteImage(seg1, os.path.join(working_dir, 'worms_outlier_int_mean_min_seg.nii.gz'))
#seg1, seg2 = worm_counter.split_by_stats(worms_seg, stat_id=6, cut_limit=80)
#sitk.WriteImage(seg1, os.path.join(working_dir, 'worms_outlier_int_dev_seg.nii.gz'))
#seg1, seg2 = worm_counter.split_by_stats(worms_seg, stat_id=7, cut_limit=0.7)
#sitk.WriteImage(seg2, os.path.join(working_dir, 'worms_outlier_int_skew_seg.nii.gz'))
