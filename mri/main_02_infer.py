from imaging.mri.class_BrainLearn import BrainLearn

infer_on = "val"  # choose between validation and testing dataset

bl = BrainLearn()
bl.experiment = "2024.03.12 16.18.25"
bl.set_device("cuda")
bl.set_metric_function("l1")
bl.path_main = "E:\\gd_synthesis"
bl.patch_size_trn = (128, 128, 128)
bl.patch_size_val = bl.patch_size_trn
bl.patch_size_tst = bl.patch_size_trn
bl.build_model_unet()
bl.load_model_dictionary()
if infer_on == "val":
    bl.load_dataset_val_paths()
    bl.dataset_tst = bl.dataset_val
else:
    bl.load_dataset_tst_paths()
bl.compose_transforms_tst()
bl.cache_dataset_tst()
bl.infer()
