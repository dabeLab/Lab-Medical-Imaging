from imaging.mri.class_BrainLearn import BrainLearn

bl = BrainLearn()

bl.path_main = "E:\\gd_synthesis"
# Set parameters
bl.set_device("cuda")
bl.patch_size_trn = (128, 128, 128)
bl.patch_size_val = bl.patch_size_trn
bl.patch_size_tst = bl.patch_size_trn
bl.patch_number = 20
bl.batch_trn_size = 1
bl.batch_tst_size = 1
bl.batch_val_size = 1
bl.max_iteration_trn = 1000
bl.delta_iteration_trn = 10
bl.delta_iteration_save = 100
bl.batch_trn_size = 1
# Build dataset
bl.data_percentage = 1
bl.dataset_trn_ratio = 0.7
bl.dataset_val_ratio = 0.2
bl.dataset_tst_ratio = 0.1
bl.build_dataset(shuffle=False)
# Compose transformation - Edit the class method to amend
bl.compose_transforms_trn()
bl.compose_transforms_val()
bl.compose_transforms_tst()
# Cache data
bl.cache_dataset_trn()
bl.cache_dataset_val()
# bl.cache_dataset_tst()
# Build model
bl.build_model_unet()
bl.set_loss_function()
bl.set_metric_function()
bl.set_optimizer()
# Train model
bl.train()
# Save results
bl.save_dataset_trn_paths_to_csv()
bl.save_dataset_val_paths_to_csv()
bl.save_dataset_tst_paths_to_csv()
bl.save_model_attributes()
bl.save_loss()
bl.save_score()
# Test model
# bl.test()

