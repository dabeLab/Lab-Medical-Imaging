from imaging.tools.classes_cnn import *

wrm = NNMealworms()
os.chdir("/Users/berri/Desktop/ct")
wrm.load_model("2023.11.19 22.18.11 iter 1000 mdl.pth")
wrm.load_params("2023.11.19 22.18.11 params.dat")
wrm.params["dir_main"] = os.getcwd()

wrm.compose_transforms_trn()
wrm.compose_transforms_val()
wrm.compose_transforms_tst()
wrm.cache_data()
wrm.testing()
