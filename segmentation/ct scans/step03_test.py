from imaging.tools.classes_cnn import *

wrm = NNMealworms()
os.chdir("/Users/berri/Desktop/ct")
wrm.load_model("2023.11.28 10.14.38 iter 3000 mdl.pth")
wrm.load_params("2023.11.28 10.14.38 params.dat")
wrm.params["dir_main"] = "/Users/berri/Desktop/ct"
wrm.compose_transforms_tst()
for val in wrm.params["dataset_tst"]:
    val["img"] = val["img"].replace("\\", "/")
    val["lbl"] = val["lbl"].replace("\\", "/")
wrm.cache_data()
wrm.testing(morph=False)