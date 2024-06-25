from imaging.tools.classes_cnn import *

# let's try with res_units = 0
# then pick the best and add one more layer: 2048 channels

os.chdir("/Users/berri/Desktop/ct")
wrm = NNMealworms()
wrm.params["dataset_trn_ratio"] = 0.7
wrm.params["dataset_val_ratio"] = 0.15
wrm.params["dataset_tst_ratio"] = 0.15
wrm.params["delta_iteration_trn"] = 100
wrm.set_max_iteration(1000)
wrm.build_dataset()
wrm.get_dxdydz()
wrm.compose_transforms_trn()
wrm.compose_transforms_val()
wrm.compose_transforms_tst()
wrm.cache_data()
wrm.build_model_unet(num_res_units=4,
                     channels=(64, 128, 256, 512, 1024),
                     strides=(2, 2, 2, 2))
# wrm.build_model_unetr(img_size=(128, 128, 128),
#                      feature_size=16,
#                      hidden_size=768,
#                      feedfwd_layers=3072,
#                      attention_heads=12)
wrm.save_params()
wrm.train()
wrm.plot_loss()
wrm.plot_metr()
plt.show()
