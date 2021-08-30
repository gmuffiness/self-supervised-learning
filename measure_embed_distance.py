# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import gc
import sys
import argparse
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from cifar100_utils import sparse2coarse, CIFAR100Coarse
import pandas as pd
import pickle
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 100000

def compare_l2(emb_1, emb_2):
    # when embeddings are numpy, convert to tensor
    emb_1 = torch.from_numpy(emb_1).reshape([1,-1])
    emb_2 = torch.from_numpy(emb_2).reshape([1,-1])
    return float(torch.dist(emb_1, emb_2))

def compare_cosine(emb_1, emb_2):
    # when embeddings are numpy, convert to tensor
    emb_1 = torch.from_numpy(emb_1).reshape([1,-1])
    emb_2 = torch.from_numpy(emb_2).reshape([1,-1])

    emb_1 = emb_1.view(-1)
    emb_2 = emb_2.view(-1)
    return float(torch.dot(emb_1, emb_2) / (torch.norm(emb_1) * torch.norm(emb_2)))

def check_gap_between_sameclass_and_diffclass(data, embed_vectors):
    sc_l2_dist_avg = []
    sc_cos_dist_avg = []
    dc_l2_dist_avg = []
    dc_cos_dist_avg = []

    for i in range(0,len(data),50):
        sc_l2_dist_list = []
        sc_cos_dist_list = []
        dc_l2_dist_list = []
        dc_cos_dist_list = []
        for j in range(0,len(data)):
            if i <= j < (i + 50):
                # print(i, j)
                # print('='*30)
                l2_dist = compare_l2(embed_vectors[i],embed_vectors[j])
                cos_dist = compare_cosine(embed_vectors[i].view(-1), embed_vectors[j].view(-1))
                sc_l2_dist_list.append(l2_dist)
                sc_cos_dist_list.append(cos_dist)
            else:
                # print(i, j)
                l2_dist = compare_l2(embed_vectors[i], embed_vectors[j])
                cos_dist = compare_cosine(embed_vectors[i].view(-1), embed_vectors[j].view(-1))
                dc_l2_dist_list.append(l2_dist)
                dc_cos_dist_list.append(cos_dist)
        # if sc_l2_dist_list:
        sc_l2_dist_avg.append(sum(sc_l2_dist_list)/len(sc_l2_dist_list))
        sc_cos_dist_avg.append(sum(sc_cos_dist_list))
        # else:
        dc_l2_dist_avg.append(sum(dc_l2_dist_list)/len(dc_l2_dist_list))
        dc_cos_dist_avg.append(sum(dc_cos_dist_list))

    # print('\n')
    # print(sc_l2_dist_avg)
    # print('\n')
    # print(sc_cos_dist_avg)
    # print('\n')
    # print(dc_l2_dist_avg)
    # print('\n')
    # print(dc_cos_dist_avg)
    return sc_l2_dist_avg, sc_cos_dist_avg, dc_l2_dist_avg, dc_cos_dist_avg


def check_gap_between_two_diff_group(c1, c2):
    l2_dist_list = []
    cos_dist_list = []

    c1_embed = c1['embed'].values
    c2_embed = c2['embed'].values
    for i in range(len(c1_embed)):
        for j in range(len(c2_embed)):
            l2_dist = compare_l2(c1_embed[i], c2_embed[j])
            cos_dist = compare_cosine(c1_embed[i].view(-1), c2_embed[j].view(-1))
            l2_dist_list.append(l2_dist)
            cos_dist_list.append(cos_dist)
            # print(dc_l2_dist_list)
            # print(dc_cos_dist_list)
    # print(len(dc_l2_dist_list))
    # else:
    l2_dist_avg = sum(l2_dist_list) / len(l2_dist_list)
    cos_dist_avg = sum(cos_dist_list) / len(l2_dist_list)

    return l2_dist_avg, cos_dist_avg

def check_gap_between_two_same_group(embed1, embed2):
    l2_dist_list = []
    cos_dist_list = []

    c1_embed = c1['embed'].values
    c2_embed = c2['embed'].values
    for i in range(len(c1_embed)):
        for j in range(len(c2_embed)):
            l2_dist = compare_l2(c1_embed[i], c2_embed[j])
            cos_dist = compare_cosine(c1_embed[i].view(-1), c2_embed[j].view(-1))
            l2_dist_list.append(l2_dist)
            cos_dist_list.append(cos_dist)
            # print(dc_l2_dist_list)
            # print(dc_cos_dist_list)
    # print(len(dc_l2_dist_list))
    # else:
    l2_dist_avg = sum(l2_dist_list) / len(l2_dist_list)
    cos_dist_avg = sum(cos_dist_list) / len(l2_dist_list)

    return l2_dist_avg, cos_dist_avg

def main_cifar100(args):
    sg_l2_dist_list = []
    sg_cos_dist_list = []
    dg_l2_dist_list = []
    dg_cos_dist_list = []
    flag_list = []
    total_l2_list = []
    total_cos_list = []
    # load
    with open(args.embeddings_path, 'rb') as f:
        data = pickle.load(f)

    data['superclass'] = sparse2coarse(data['class'].values)
    data.sort_values(['superclass', 'class'], inplace=True)
    # data2['superclass'] = sparse2coarse(data2['class'].values)
    # trainset.targets = sparse2coarse(trainset.targets)  # update labels
    print(data)
    # print(data['class'].value_counts()[:30])
    # print(data['superclass'].value_counts())
    embed = data['embed'].values
    boy = data[data['class'] == 11]
    man = data[data['class'] == 46]
    possum = data[data['class'] == 64]
    racoon = data[data['class'] == 66]
    keyboard = data[data['class'] == 39]
    television = data[data['class'] == 87]

    data2 = pd.concat([boy, man, possum, racoon, keyboard, television])
    # print(data2)
    draw_heatmap(args.output_dir, data)

    # print(f'boy : {boy}')
    # print(f'man : {man}')
    # superclasses = data['superclass'].values
    # # print(f'superclasses : {superclasses}')
    # for i in range(len(data)):
    #     for j in range(1+i, len(data)):
    #         if superclasses[i] == superclasses[j]:
    #             flag_list.append(True)
    #
    #             l2_dist = compare_l2(embed[i], embed[j])
    #             cos_dist = compare_cosine(embed[i], embed[j])
    #             sg_l2_dist_list.append(l2_dist)
    #             sg_cos_dist_list.append(cos_dist)
    #             total_l2_list.append(float(l2_dist))
    #             total_cos_list.append(float(cos_dist))
    #             # print(l2_dist_avg, cos_dist_avg)
    #         else:
    #             flag_list.append(False)
    #             l2_dist = compare_l2(embed[i], embed[j])
    #             cos_dist = compare_cosine(embed[i], embed[j])
    #             dg_l2_dist_list.append(l2_dist)
    #             dg_cos_dist_list.append(cos_dist)
    #             total_l2_list.append(float(l2_dist))
    #             total_cos_list.append(float(cos_dist))
    # sg_l2_dist_avg = sum(sg_l2_dist_list) / len(sg_l2_dist_list)
    # sg_cos_dist_avg = sum(sg_cos_dist_list) / len(sg_cos_dist_list)
    # dg_l2_dist_avg = sum(dg_l2_dist_list) / len(dg_l2_dist_list)
    # dg_cos_dist_avg = sum(dg_cos_dist_list) / len(dg_cos_dist_list)
    #             # l2_dist_avg, cos_dist_avg = check_gap_between_two_diff_group(embed[i], embed[j])
    #             # print(l2_dist_avg, cos_dist_avg)
    # # print(sg_l2_dist_list)
    # # print(dg_l2_dist_list)
    # print(sg_l2_dist_avg)
    # print(dg_l2_dist_avg)
    # print(sg_cos_dist_avg)
    # print(dg_cos_dist_avg)
    # # l2_dist_avg, cos_dist_avg = check_gap_between_two_group(man, racoon)
    # # print(l2_dist_avg, cos_dist_avg)
    #
    # df2 = pd.DataFrame({'l2':total_l2_list, 'cos': total_cos_list,'flag':flag_list})
    # # print(df2)
    # # print(df2.plot())
    # groups = df2.groupby('flag')
    # for name, group in groups:
    #     plt.plot(group["cos"], group["l2"], marker="o", label=name)
    # plt.xlabel('cosine similiarity')
    # plt.ylabel('l2 distance')
    # # df2.boxplot(by='flag')
    # # plt.savefig('./l2_cos_fig.png')
    # # trainset = CIFAR100Coarse(root='./data', train=False, transform=None, target_transform=None, download=False)
    # embed_vectors = data['embed']
    #
    # sc_l2_dist_avg, sc_cos_dist_avg, dc_l2_dist_avg, dc_cos_dist_avg = check_gap_between_sameclass_and_diffclass(data, embed_vectors)
    # # plot(sc_l2_dist_avg, sc_cos_dist_avg, dc_l2_dist_avg, dc_cos_dist_avg)

def inner_class_distance(output_dir, data, class_name):

    l2_dist_list = []
    cos_dist_list = []

    sc_embed = data[data['class'] == class_name]['embed'].values
    # print(sc_embed)
    if len(sc_embed) == 1:
        return 0, 0
    for i in range(len(sc_embed)):
        for j in range(i+1, len(sc_embed)-i):
            l2_dist = compare_l2(sc_embed[i], sc_embed[i+j])
            cos_dist = compare_cosine(sc_embed[i], sc_embed[i+j])
            l2_dist_list.append(l2_dist)
            cos_dist_list.append(cos_dist)
    # print(l2_dist_list)

    l2_dist_avg = sum(l2_dist_list) / len(l2_dist_list)
    cos_dist_avg = sum(cos_dist_list) / len(l2_dist_list)
    # print(f'{class_name} class l2_dist_avg : {l2_dist_avg}, cos_dist_avg : {cos_dist_avg}')

    # draw_boxplot(output_dir, l2_dist_list, cos_dist_list, class_name, c2_name=None, inner=True)

    return l2_dist_avg, cos_dist_avg

def diff_class_distance(output_dir, data, c1_name, c2_name):

    l2_dist_list = []
    cos_dist_list = []

    c1_embed = data[data['class'] == c1_name]['embed'].values
    c2_embed = data[data['class'] == c2_name]['embed'].values

    for i in range(len(c1_embed)):
        for j in range(len(c2_embed)):
            l2_dist = compare_l2(c1_embed[i], c2_embed[j])
            cos_dist = compare_cosine(c1_embed[i], c2_embed[j])
            l2_dist_list.append(l2_dist)
            cos_dist_list.append(cos_dist)

    l2_dist_avg = sum(l2_dist_list) / len(l2_dist_list)
    cos_dist_avg = sum(cos_dist_list) / len(l2_dist_list)
    # print(f'{c1_name},{c2_name} class l2_dist_avg : {l2_dist_avg}, cos_dist_avg : {cos_dist_avg}')

    # draw_boxplot(output_dir, l2_dist_list, cos_dist_list, c1_name, c2_name, inner=False)

    return l2_dist_avg, cos_dist_avg

def draw_boxplot(output_dir, l2_dist_list, cos_dist_list, c1_name, c2_name=None, inner=False):

    l2 = pd.DataFrame({'l2': l2_dist_list})
    cos = pd.DataFrame({'cos': cos_dist_list})

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    l2.boxplot(ax=ax[0])
    cos.boxplot(ax=ax[1])

    if inner:
        plt.title(f'{c1_name} class distance boxplot')
        plt.savefig(os.path.join(output_dir, f'inner_{c1_name}_distance_boxplot.png'))
    else:
        plt.title(f'{c1_name},{c2_name} distance boxplot')
        plt.savefig(os.path.join(output_dir, f'{c1_name}_{c2_name}_distance_boxplot.png'))

def draw_heatmap(output_dir, data):

    l2_dist_avg_list = []
    cos_dist_avg_list = []

    heatmap_value_l2 = []
    heatmap_value_cos = []

    class_names = data['class'].unique()
    for c_name1 in class_names:
        for c_name2 in class_names:
            if c_name1 == c_name2:
                l2_dist_avg, cos_dist_avg = inner_class_distance(output_dir, data, c_name1)
            else:
                l2_dist_avg, cos_dist_avg = diff_class_distance(output_dir, data, c_name1, c_name2)
            l2_dist_avg_list.append(round(l2_dist_avg, 2))
            cos_dist_avg_list.append(round(1 - cos_dist_avg, 3))

        heatmap_value_l2.append(l2_dist_avg_list)
        heatmap_value_cos.append(cos_dist_avg_list)
        l2_dist_avg_list = []
        cos_dist_avg_list = []

    heatmap_value_l2 = np.array(heatmap_value_l2)
    heatmap_value_cos = np.array(heatmap_value_cos)
    # print(f'heatmap_value_l2:\n\n{heatmap_value_l2}')
    # print(f'heatmap_value_cos:\n\n{heatmap_value_cos}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(90,90))
    im = ax1.imshow(np.array(heatmap_value_l2))
    im = ax2.imshow(np.array(heatmap_value_l2))

    # class_names = ['boy', 'man', 'possum', 'racoon', 'keyboard', 'television']
    # We want to show all ticks...
    ax1.set_xticks(np.arange(len(class_names)))
    ax1.set_yticks(np.arange(len(class_names)))
    ax2.set_xticks(np.arange(len(class_names)))
    ax2.set_yticks(np.arange(len(class_names)))
    # ... and label them with the respective list entries
    ax1.set_xticklabels(class_names)
    ax1.set_yticklabels(class_names)
    ax2.set_xticklabels(class_names)
    ax2.set_yticklabels(class_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax1.text(j, i, heatmap_value_l2[i, j],
                           ha="center", va="center", color="w")
            text = ax2.text(j, i, heatmap_value_cos[i, j],
                           ha="center", va="center", color="w")


    ax1.set_title("l2 distance between two class")
    ax2.set_title("cosine distance between two class")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_of_distance6.png'))

def main(args):

    output_dir = args.output_dir
    # load
    # class : 'teddy_bear', 'maltest_dog', 'great_white_shark', 'airliner', 'aircraft_carrier', 'recreational_vehicle'
    with open(args.embeddings_dir, 'rb') as f:
        data = pickle.load(f)
    print(data)

    # 0. embedding들의 분포 확인해보기
    embed = data['embed'].values
    embed = [one_embed.cpu().numpy().flatten() for one_embed in embed]
    embed = np.array(embed).flatten()
    fig, ax = plt.subplots()
    ax.hist(embed, bins=100)
    # ax.set_ylim(-10.0, 10.0)
    ax.set_xlabel('embedding values')
    ax.set_ylabel('frequency')
    # print(embed.shape)
    # plt.savefig(os.path.join(output_dir, 'embed_hist_imagenet'))

    # 1. 같은 클래스끼리의 거리 boxplot 그려보기
    # ex) 개1 <=> 개2, 개1 <=> 개3, ... | m_C_2
    # inner_class_distance(output_dir, data, 'teddy_bear')

    # 2. 서로 다른 두 클래스끼리 거리 비교해보기
    # ex) 개1 <=> 고양이1, 개1 <=> 고양이2, ... | m x n
    # diff_class_distance(output_dir, data, 'teddy_bear', 'maltese_dog')
    # diff_class_distance(output_dir, data, 'teddy_bear', 'great_white_shark')

    # 3. 각 클래스 간 pair distance의 heatmap
    draw_heatmap(output_dir, data)

    # 4. 각 클래스 내의 분산 확인해보기

    # 3. 서로 다른 두 그룹(superclass)에 속하는 클래스끼리 거리 비교해보기
    # diff_group_distance()




    #
    #
    #
    # man = data[data['class'] == 46]
    # racoon = data[data['class'] == 66]
    # print(f'boy : {boy}')
    # print(f'man : {man}')
    #
    # # print(f'superclasses : {superclasses}')
    # for i in range(len(data)):
    #     for j in range(1+i, len(data)):
    #         if superclasses[i] == superclasses[j]:
    #             flag_list.append(True)
    #
    #             l2_dist = compare_l2(embed[i], embed[j])
    #             cos_dist = compare_cosine(embed[i], embed[j])
    #             sg_l2_dist_list.append(l2_dist)
    #             sg_cos_dist_list.append(cos_dist)
    #             total_l2_list.append(float(l2_dist))
    #             total_cos_list.append(float(cos_dist))
    #             # print(l2_dist_avg, cos_dist_avg)
    #         else:
    #             flag_list.append(False)
    #             l2_dist = compare_l2(embed[i], embed[j])
    #             cos_dist = compare_cosine(embed[i], embed[j])
    #             dg_l2_dist_list.append(l2_dist)
    #             dg_cos_dist_list.append(cos_dist)
    #             total_l2_list.append(float(l2_dist))
    #             total_cos_list.append(float(cos_dist))
    # sg_l2_dist_avg = sum(sg_l2_dist_list) / len(sg_l2_dist_list)
    # sg_cos_dist_avg = sum(sg_cos_dist_list) / len(sg_cos_dist_list)
    # dg_l2_dist_avg = sum(dg_l2_dist_list) / len(dg_l2_dist_list)
    # dg_cos_dist_avg = sum(dg_cos_dist_list) / len(dg_cos_dist_list)
    #             # l2_dist_avg, cos_dist_avg = check_gap_between_two_diff_group(embed[i], embed[j])
    #             # print(l2_dist_avg, cos_dist_avg)
    # # print(sg_l2_dist_list)
    # # print(dg_l2_dist_list)
    # print(sg_l2_dist_avg)
    # print(dg_l2_dist_avg)
    # print(sg_cos_dist_avg)
    # print(dg_cos_dist_avg)
    # # l2_dist_avg, cos_dist_avg = check_gap_between_two_group(man, racoon)
    # # print(l2_dist_avg, cos_dist_avg)
    #
    # df2 = pd.DataFrame({'l2':total_l2_list, 'cos': total_cos_list,'flag':flag_list})
    # # print(df2)
    # # print(df2.plot())
    # groups = df2.groupby('flag')
    # for name, group in groups:
    #     plt.plot(group["cos"], group["l2"], marker="o", label=name)
    # plt.xlabel('cosine similiarity')
    # plt.ylabel('l2 distance')
    # # df2.boxplot(by='flag')
    # plt.savefig('./l2_cos_fig.png')
    # # trainset = CIFAR100Coarse(root='./data', train=False, transform=None, target_transform=None, download=False)
    # embed_vectors = data['embed']
    #
    # sc_l2_dist_avg, sc_cos_dist_avg, dc_l2_dist_avg, dc_cos_dist_avg = check_gap_between_sameclass_and_diffclass(data, embed_vectors)
    # # plot(sc_l2_dist_avg, sc_cos_dist_avg, dc_l2_dist_avg, dc_cos_dist_avg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Measure embedding distance')
    parser.add_argument("--image_dir", default='/nas/home/gmuffiness/data/imagenet/val', type=str,
                        help="Path of the image to load.")
    parser.add_argument('--output_dir',
                        default='/nas/home/gmuffiness/result/embeddings/simclr/pretrain_with_imagenet',
                        help='Path where to save visualizations.')
    parser.add_argument('--embeddings_path',
                        default='/nas/home/gmuffiness/result/embeddings/simclr/pretrain_with_imagenet/embeddings_id_0_to_99_cifar100.pkl',
                        help='Path where to save visualizations.')

    args = parser.parse_args()
    # main(args=args)
    main_cifar100(args=args)

