# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pymic.io_my.image_read_write import load_image_as_nd_array

class NiftyDataset(Dataset):
    """Dataset for loading images. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors 
    with dimention order [C, H, W] for 2D images"""

    def __init__(self, root_dir, csv_file, modal_num = 1, 
            with_label = False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with image names.
            modal_num (int): Number of modalities. 
            with_label (bool): Load the data with segmentation ground truth.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir   = root_dir
        self.csv_items  = pd.read_csv(csv_file)
        self.modal_num  = modal_num
        self.with_label = with_label
        self.transform  = transform

    def __len__(self):
        return len(self.csv_items)

    def __getitem__(self, idx):
        names_list, image_list = [], []
        for i in range (self.modal_num):
            image_name = self.csv_items.iloc[idx, i]
            image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
            image_dict = load_image_as_nd_array(image_full_name)
            image_data = image_dict['data_array']
            names_list.append(image_name)
            image_list.append(image_data)
        image = np.concatenate(image_list, axis = 0)

        # add two channels
        # D = image.shape[1]
        # H = image.shape[2]
        # W = image.shape[3]
        # temp = np.array([i for i in range(W)])
        # row = np.tile(np.expand_dims(temp, axis=(0, 1, 2)), reps=(1, D, H, 1))
        # temp = np.array([i for i in range(H)])
        # col = np.tile(np.expand_dims(temp, axis=(0, 1, 2)), reps=(1, D, W, 1))
        # image = np.concatenate((image, row, np.transpose(col, axes=(0, 1, 3, 2))), axis=0)

        image = np.asarray(image, np.float32)
        
        sample = {'image': image, 'names' : names_list[0], 
                 'origin':image_dict['origin'],
                 'spacing': image_dict['spacing'],
                 'direction':image_dict['direction']}
        if (self.with_label):    
            label_name = "{0:}/{1:}".format(self.root_dir, self.csv_items.iloc[idx, -1])
            label = load_image_as_nd_array(label_name)['data_array']
            label = np.asarray(label, np.int32)
            sample['label'] = label
        if self.transform:
            sample = self.transform(sample)

        return sample

class NiftyDatasetContour(NiftyDataset):

    def __getitem__(self, idx):
        names_list, image_list = [], []
        for i in range (self.modal_num):
            image_name = self.csv_items.iloc[idx, i]
            image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
            image_dict = load_image_as_nd_array(image_full_name)
            image_data = image_dict['data_array']
            names_list.append(image_name)
            image_list.append(image_data)
        image = np.concatenate(image_list, axis = 0)

        image = np.asarray(image, np.float32)
        
        sample = {'image': image, 'names' : names_list[0], 
                 'origin':image_dict['origin'],
                 'spacing': image_dict['spacing'],
                 'direction':image_dict['direction']}
        if (self.with_label):    
            label_name = "{0:}/{1:}".format(self.root_dir, self.csv_items.iloc[idx, -2])
            label = load_image_as_nd_array(label_name)['data_array']
            label = np.asarray(label, np.int32)
            sample['label'] = label
            contour_name = "{0:}/{1:}".format(self.root_dir, self.csv_items.iloc[idx, -1])
            contour = load_image_as_nd_array(contour_name)["data_array"]
            contour = np.asarray(contour, np.int32)
            sample["contour"] = contour
        if self.transform:
            sample = self.transform(sample)

        return sample