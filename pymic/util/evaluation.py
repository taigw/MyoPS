# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys
import math
import random
import GeodisTK
import configparser
import numpy as np
from scipy import ndimage
from pymic.io_my.image_read_write import *
from pymic.util.image_process import *
from pymic.util.parse_config import parse_config

# Dice evaluation
def binary_dice(s, g, resize = False):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    resize: if s and g have different shapes, resize s to match g.
    """
    assert(len(s.shape)== len(g.shape))
    if(resize):
        size_match = True
        for i in range(len(s.shape)):
            if(s.shape[i] != g.shape[i]):
                size_match = False
                break
        if(size_match is False):
            s = resize_ND_volume_to_given_shape(s, g.shape, order = 0)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-5)/(s1 + s2 + 1e-5)
    return dice

def dice_of_images(s_name, g_name):
    s = load_image_as_nd_array(s_name)['data_array']
    g = load_image_as_nd_array(g_name)['data_array']
    dice = binary_dice(s, g)
    return dice

# IOU evaluation
def binary_iou(s,g):
    assert(len(s.shape)== len(g.shape))
    intersecion = np.multiply(s, g)
    union = np.asarray(s + g >0, np.float32)
    iou = intersecion.sum()/(union.sum() + 1e-10)
    return iou

def iou_of_images(s_name, g_name):
    s = load_image_as_nd_array(s_name)['data_array']
    g = load_image_as_nd_array(g_name)['data_array']
    margin = (3, 8, 8)
    g = get_detection_binary_bounding_box(g, margin)
    return binary_iou(s, g)

# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if(dim == 2):
        strt = ndimage.generate_binary_structure(2,1)
    else:
        strt = ndimage.generate_binary_structure(3,1)
    ero  = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8) 
    return edge 


def binary_hausdorff95(s, g, spacing = None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert(image_dim == len(g.shape))
    if(spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert(image_dim == len(spacing))
    img = np.zeros_like(s)
    if(image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif(image_dim ==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1)*0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2)*0.95)]
    return max(dist1, dist2)


def binary_assd(s, g, spacing = None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert(image_dim == len(g.shape))
    if(spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert(image_dim == len(spacing))
    img = np.zeros_like(s)
    if(image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif(image_dim ==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng) 
    return assd

# relative volume error evaluation
def binary_relative_volume_error(s_volume, g_volume):
    s_v = float(s_volume.sum())
    g_v = float(g_volume.sum())
    assert(g_v > 0)
    rve = abs(s_v - g_v)/g_v
    return rve

def get_evaluation_score(s_volume, g_volume, spacing, metric):
    if(len(s_volume.shape) == 4):
        assert(s_volume.shape[0] == 1 and g_volume.shape[0] == 1)
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    if(s_volume.shape[0] == 1):
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    metric_lower = metric.lower()

    if(metric_lower == "dice"):
        score = binary_dice(s_volume, g_volume)

    elif(metric_lower == "iou"):
        score = binary_iou(s_volume,g_volume)

    elif(metric_lower == 'assd'):
        from pymic.util.binary import assd
        score = assd(s_volume, g_volume, spacing)
        # score = binary_assd(s_volume, g_volume, spacing)

    elif(metric_lower == "hausdorff95"):
        score = binary_hausdorff95(s_volume, g_volume, spacing)

    elif(metric_lower == "rve"):
        score = binary_relative_volume_error(s_volume, g_volume)

    elif(metric_lower == "volume"):
        voxel_size = 1.0
        for dim in range(len(spacing)):
            voxel_size = voxel_size * spacing[dim]
        score = g_volume.sum()*voxel_size
    else:
        raise ValueError("unsupported evaluation metric: {0:}".format(metric))

    return score

def evaluation(config_file):
    config = parse_config(config_file)['evaluation']
    metric = config['metric']
    labels = config['label_list']
    organ_name = config['organ_name']
    ground_truth_label_convert_source = config.get('ground_truth_label_convert_source', None)
    ground_truth_label_convert_target = config.get('ground_truth_label_convert_target', None)
    segmentation_label_convert_source = config.get('segmentation_label_convert_source', None)
    segmentation_label_convert_target = config.get('segmentation_label_convert_target', None)
    s_folder_list = config['segmentation_folder_list']
    g_folder_list = config['ground_truth_folder_list']
    s_format  = config['segmentation_format']
    g_format  = config['ground_truth_format']
    s_postfix = config.get('segmentation_postfix',None)
    g_postfix = config.get('ground_truth_postfix',None)

    s_postfix_long = '.' + s_format
    if(s_postfix is not None):
        s_postfix_long = '_' + s_postfix + s_postfix_long
    g_postfix_long = '.' + g_format
    if(g_postfix is not None):
        g_postfix_long = '_' + g_postfix + g_postfix_long

    patient_names_file = config['patient_file_names']
    with open(patient_names_file) as f:
            content = f.readlines()
            patient_names = [x.strip() for x in content] 

    for s_folder in s_folder_list:
        score_all_data = []
        for i in range(len(patient_names)):
            # load segmentation and ground truth
            s_name = os.path.join(s_folder, patient_names[i] + s_postfix_long)
            if(not os.path.isfile(s_name)):
                break

            for g_folder in g_folder_list:
                g_name = os.path.join(g_folder, patient_names[i] + g_postfix_long)
                if(os.path.isfile(g_name)):
                    break
            s_dict = load_image_as_nd_array(s_name)
            g_dict = load_image_as_nd_array(g_name)
            s_volume = s_dict["data_array"][0]; s_spacing = s_dict["spacing"]
            g_volume = g_dict["data_array"][0]; g_spacing = g_dict["spacing"]
            # for dim in range(len(s_spacing)):
            #     assert(s_spacing[dim] == g_spacing[dim])
            if((ground_truth_label_convert_source is not None) and \
                ground_truth_label_convert_target is not None):
                g_volume = convert_label(g_volume, ground_truth_label_convert_source, \
                    ground_truth_label_convert_target)

            if((segmentation_label_convert_source is not None) and \
                segmentation_label_convert_target is not None):
                s_volume = convert_label(s_volume, segmentation_label_convert_source, \
                    segmentation_label_convert_target)

            # # fuse multiple labels
            # s_volume_sub = np.zeros_like(s_volume)
            # g_volume_sub = np.zeros_like(g_volume)
            # for lab in labels:
            #     s_volume_sub = s_volume_sub + np.asarray(s_volume == lab, np.uint8)
            #     g_volume_sub = g_volume_sub + np.asarray(g_volume == lab, np.uint8)
            
            # # get evaluation score
            # temp_score = get_evaluation_score(s_volume_sub > 0, g_volume_sub > 0,
            #             s_spacing, metric)
            # score_all_data.append(temp_score)
            # print(patient_names[i], temp_score)

            # caculate metric of every label, not fuse labels
            temp_score = []
            for lab in labels:
                s_volume_sub = np.asarray(s_volume == lab, np.bool)
                g_volume_sub = np.asarray(g_volume == lab, np.bool)
                temp_score_single_class = get_evaluation_score(s_volume_sub, g_volume_sub, s_spacing, metric)
                temp_score.append(temp_score_single_class)
            score_all_data.append(temp_score)
        score_all_data = np.asarray(score_all_data)
        score_mean = [score_all_data.mean(axis = 0)]
        score_std  = [score_all_data.std(axis = 0)]
        np.savetxt("{0:}/{1:}_{2:}_all.txt".format(s_folder, organ_name, metric), score_all_data)
        np.savetxt("{0:}/{1:}_{2:}_mean.txt".format(s_folder, organ_name, metric), score_mean)
        np.savetxt("{0:}/{1:}_{2:}_std.txt".format(s_folder, organ_name, metric), score_std)
        print("{0:} mean ".format(metric), score_mean)
        print("{0:} std  ".format(metric), score_std) 


def main():
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('    python pyMIC.util/evaluation.py config.cfg')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    evaluation(config_file)
    
if __name__ == '__main__':
    main()
