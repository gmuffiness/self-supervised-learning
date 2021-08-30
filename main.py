import sys
import torch
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{torch.__version__[0:5:2]}"
])
print(version_str)

import vissl
import tensorboard
import apex
import torch

#
# json_data = {
#     "dummy_data_folder": {
#       "train": [
#         "./dummy_data/train", "./dummy_data/train"
#       ],
#       "val": [
#         "./dummy_data/val", "./dummy_data/val"
#       ]
#     }
# }
#
# # use VISSL's api to save or you can use your custom code.
# from vissl.utils.io import save_file
# save_file(json_data, "./configs/config/dataset_catalog.json")


from vissl.data.dataset_catalog import VisslDatasetCatalog

# list all the datasets that exist in catalog
print(VisslDatasetCatalog.list())

# get the metadata of dummy_data_folder dataset
print(VisslDatasetCatalog.get("dummy_data_folder"))