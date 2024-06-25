from imaging.tools.classes_cnn import *

os.chdir("/Users/berri/Desktop/ct")
wrm = NNMealworms()
wrm.params["dataset_trn_ratio"] = 0.7
wrm.params["dataset_val_ratio"] = 0.15
wrm.params["dataset_tst_ratio"] = 0.15
wrm.params["max_iteration_trn"] = 100
wrm.params["delta_iteration_trn"] = 10
wrm.build_dataset()
wrm.get_dxdydz()
wrm.compose_transforms_trn()
wrm.compose_transforms_val()
wrm.compose_transforms_tst()
wrm.cache_data()
wrm.build_model_unet()
wrm.save_params()
wrm.train()
wrm.plot_loss()
wrm.plot_metr()
plt.show()
