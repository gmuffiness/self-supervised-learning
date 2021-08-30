# install apex by checking system settings: cuda version, pytorch version, python version
import sys
import torch
import vissl
import tensorboard
import apex
import os
import argparse

from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_weights
from PIL import Image
import torchvision.transforms as transforms

# version_str="".join([
#     f"py3{sys.version_info.minor}_cu",
#     torch.version.cuda.replace(".",""),
#     f"_pyt{torch.__version__[0:5:2]}"
# ])
# print(version_str)

def set_config(args):

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("vissl/config/defaults.yaml")

    cfg = OmegaConf.merge(default_config, config)
    # print(cfg)

    cfg = AttrDict(cfg)
    # cfg.config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks."
    cfg.config.MODEL.WEIGHTS_INIT.PARAMS_FILE = args.pretrained_weights
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON = True
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY = True
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY = True
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS = False
    cfg.config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP = [["res5avg", ["Identity", []]]]
    # print(cfg.config.MODEL)

    model = build_model(cfg.config.MODEL, cfg.config.OPTIMIZER)
    # print(model)

    weights = load_checkpoint(checkpoint_path=cfg.config.MODEL.WEIGHTS_INIT.PARAMS_FILE)

    init_model_from_weights(
        config=cfg.config,
        model=model,
        state_dict=weights,
        state_dict_key_name="classy_state_dict",
        skip_layers=[],  # Use this if you do not want to load all layers
    )

    # print("Loaded...")
    return model

def get_embeddings(args, image):

    model = set_config(args)

    # image가 경로일 경우
    # image = Image.open(image)
    # image = image.convert("RGB")
    # image_path가 tensor일 경우


    pipeline = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    x = pipeline(image)

    features = model(x.unsqueeze(0))
    # print(features[0].shape)

    return features[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get embedding vectors from pretrained weights')
    # parser.add_argument('--arch', default='vit_small', type=str, help='Architecture (support only ViT atm).')
    parser.add_argument('--pretrained_weights', default='SimCLR_RN50_800ep_pretrain.torch', type=str, help="Path to pretrained weights to load.")
    parser.add_argument("--image_path", default='test_image.jpg', type=str, help="Path of the image to load.")
    parser.add_argument("--config_path", default='configs/config/simclr/simclr_8node_resnet.yaml', type=str, help="Path of the config to load.")

    args = parser.parse_args()
    get_embeddings(args=args, image_path=args.image_path)