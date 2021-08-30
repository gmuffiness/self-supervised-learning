# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import argparse
import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms as pth_transforms
import pandas as pd
import get_embeddings_simclr
import cv2
from torchvision import transforms

def get_embeddings(args, image):
    embeddings = get_embeddings_simclr.get_embeddings(args, image=image)
    return embeddings

def save_imagenet_embeddings(args):

    # 1 read folder
    img_name_list = []
    class_list = []
    embed_list = []
    subfolders_by_class = os.listdir(f'{args.image_dir}')

    for i, subfolder in enumerate(subfolders_by_class):
        folder_path = f'{args.image_dir}/{subfolder}/'

        # read images,labels list
        file_list = os.listdir(f'{folder_path}')
        jpeg_list = [file for file in file_list if file.endswith(".JPEG")]
        # img_names = os.listdir(f'{jpeg_list}')
        for img_name in jpeg_list:
            img_path = os.path.join(folder_path, img_name)
            embeddings = get_embeddings(args, image_path=img_path)
            embeddings = torch.reshape(embeddings, [1,-1])
            # print(len(embeddings[0]))

            img_name_list.append(img_name)
            class_list.append(subfolder)
            embed_list.append(embeddings)
        print(subfolder+'##'+str(i))
        if i % 11 == 0 and i != 0:
            df = pd.DataFrame({'img_name': img_name_list, 'class': class_list, 'embed': embed_list})
            df.to_pickle(os.path.join(args.output_dir, f"embeddings{i - 11}_to{i}_imagenet_pretrained_300.pkl"))
            print('embedding_saved')
            img_name_list = []
            class_list = []
            embed_list = []

def save_cifar100_embeddings(args):

    class_list = []
    embed_list = []

    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_data = CIFAR100(download=True, train=False, root="./data", transform=transform)
    # test_data = CIFAR100(root="./data", train=False, transform=transform)
    for i, (image, label) in enumerate(train_data):

        # convert tensor to image
        # tensor = image.cpu().numpy()  # make sure tensor is on cpu
        # cv2.imwrite(tensor, "image.png")

        image = transforms.ToPILImage()(image.squeeze_(0))
        print(i)

        if i % 9999 == 0 and i != 0:
            df = pd.DataFrame({'class': class_list, 'embed': embed_list})
            df.to_pickle(os.path.join(args.output_dir, f"embeddings_id_{i-1000}_to_{i}_cifar100.pkl"))
            class_list = []
            embed_list = []
            print(f'embeddings{i-1000}_to{i}_cifar100.pkl saved.')

        # print(image)

        embeddings = get_embeddings(args, image=image)
        # print(embeddings.shape)
        class_list.append(label)
        embed_list.append(embeddings.cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Save embedding vector values')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    # parser.add_argument('--pretrained_weights', default='/nas/home/gmuffiness/model/SimCLR_RN50_800ep_pretrain.torch', type=str, help="Path to pretrained weights to load.")
    parser.add_argument('--pretrained_weights', default='SimCLR_RN50_800ep_pretrain.torch', type=str, help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_dir", default='/nas/home/gmuffiness/data/imagenet/val', type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='/nas/home/gmuffiness/result/embeddings/simclr/pretrain_with_imagenet', help='Path where to save visualizations.')
    parser.add_argument("--config_path", default='configs/config/simclr/simclr_8node_resnet.yaml', type=str, help="Path of the config to load.")

    args = parser.parse_args()

    # save_imagenet_embeddings(args=args)
    save_cifar100_embeddings(args=args)
