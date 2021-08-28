import os
import numpy as np
import copy
import SimpleITK as sitk
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
from pymic.io_my.image_read_write import load_image_as_nd_array, save_nd_array_as_image, save_cropped_array_as_nifty_volume
from pymic.util.image_process import get_ND_bounding_box, crop_ND_volume_with_bounding_box, set_ND_volume_roi_with_bounding_box_range
from pymic.util.image_process import convert_label

first_crop = "/mnt/39E12DAE493BA6C1/datasets/MyoPS2020/test_crop.json"
second_crop = "/mnt/39E12DAE493BA6C1/datasets/MyoPS2020/coarse_to_fine/test_to_fine.json"
first_json = load_json((first_crop))
second_json = load_json(second_crop)
seg_dir = "/mnt/39E12DAE493BA6C1/datasets/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task601_not_fine/predsTs_2d"
save_dir = "/mnt/39E12DAE493BA6C1/datasets/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task601_not_fine/predsTs_2d_raw_size"
img_dir = "/mnt/39E12DAE493BA6C1/datasets/MyoPS2020/test20"
for pred_name in second_json.keys():
    print(pred_name)
    seg_name = pred_name.replace("_pred", "")
    save_name = pred_name.replace("_pred", "_seg")
    img_name = pred_name.replace("_pred", "_C0")
    seg_array = load_image_as_nd_array(join(seg_dir, seg_name))["data_array"].squeeze(0)
    img_array = load_image_as_nd_array(join(img_dir, img_name))["data_array"].squeeze(0)
    zero_array = np.zeros_like(img_array)
    crop_bbox_max = second_json[pred_name]["crop_bbox_max"]
    crop_bbox_min = second_json[pred_name]["crop_bbox_min"]
    first_json_bbox = first_json[join(img_dir, img_name)]
    print(crop_bbox_max)
    print(crop_bbox_min)
    print(first_json_bbox)
    for i, bbox in enumerate(["crop_bbox_d", "crop_bbox_h", "crop_bbox_w"]):
        # crop_bbox_max[i] += first_json_bbox[bbox][0]
        # crop_bbox_min[i] += first_json_bbox[bbox][0]
        crop_bbox_max[i] = first_json_bbox[bbox][1]
        crop_bbox_min[i] = first_json_bbox[bbox][0]
    print(crop_bbox_max)
    print(crop_bbox_min)
    print()
    print(seg_array.shape, zero_array.shape)
    output_seg_array = set_ND_volume_roi_with_bounding_box_range(zero_array, crop_bbox_min, crop_bbox_max, seg_array, False)
    output_seg_array = convert_label(output_seg_array, [1, 2, 3, 4, 5], [200, 500, 600, 1220, 2221])
    save_nd_array_as_image(output_seg_array, join(save_dir, save_name), join(img_dir, img_name))
