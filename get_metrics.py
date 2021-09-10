from medpy import metric
import pandas as pd
import numpy as np
import re
from pymic.util.image_process import convert_label
from batchgenerators.utilities.file_and_folder_operations import join, listdir
import SimpleITK as sitk



def check(arr, elem_list):
    elements = np.unique(arr)
    amount = []
    for elem in elements:
        amount.append(np.sum(arr == elem))
    print(elements)
    print(amount)
    if (not len(elements) == len(amount)) or (not len(elements) == len(elem_list)):
        raise ValueError("length of elements is not equal to length of amount")
    for elem in elem_list:
        if not elem in elements:
            raise ValueError("{0:d} is not in elements".format(elem))

if __name__ == "__main__":
    pred_dir = "myops/result_train_post/unet2d"
    gt_dir = "/mnt/data1/swzhai/dataset/MyoPS/data_preprocessed/labelsTr"
    df_dice = {"name": [], "label_1": [], "label_2": [], "label_3": []}
    for fold in range(1, 6):
        sub_pred_dir = join(pred_dir, "fold_{0:d}".format(fold))
        name_list = listdir(sub_pred_dir)
        for name in name_list:
            pred_sitk = sitk.ReadImage(join(sub_pred_dir, name))
            pred_arr = sitk.GetArrayFromImage(pred_sitk)
            print(name)
            check(pred_arr, [0, 1, 2, 3])

            gt_name = name.replace(".nii.gz", "_gd.nii.gz")
            gt_sitk = sitk.ReadImage(join(gt_dir, gt_name))
            gt_arr = sitk.GetArrayFromImage(gt_sitk)
            print(gt_name)
            check(gt_arr, [0, 1, 2, 3, 4, 5])
            gt_arr = convert_label(gt_arr, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 1, 1])

            df_dice["name"].append(name)
            for label in [1, 2, 3]:  # foreground label
                dice = metric.binary.dc(pred_arr == label, gt_arr == label)
                df_dice["label_{0:d}".format(label)].append(dice)
    
    df_dice["name"].append("mean")
    df_dice["name"].append("std")
    for label in [1, 2, 3]:
        mean_dice = np.mean(df_dice["label_{0:d}".format(label)])
        std_dice = np.std(df_dice["label_{0:d}".format(label)])
        df_dice["label_{0:d}".format(label)].append(mean_dice)
        df_dice["label_{0:d}".format(label)].append(std_dice)
    df_dice = pd.DataFrame(df_dice)
    df_dice.to_csv(join(pred_dir, "dice.csv"))

