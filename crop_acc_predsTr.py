from pymic.util.image_process import crop_ND_volume_with_bounding_box
from pymic.io_my.image_read_write import save_cropped_array_as_nifty_volume
import SimpleITK as sitk
import os
import shutil
from shutil import Error
from crop_acc_labelsTr import crop_acc_mask
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json, load_pickle

def recursively_find_file(folder, file_name):
    # TODO: print a hint when not founding file_name
    """return the path of file_name

    Args:
        folder (string): the directory the file_name located in
        file_name ([string]): file_name you want to find

    Returns:
        [string]: the path of file_name
    """

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

if __name__ == "__main__":
    root_dir = "/mnt/39E12DAE493BA6C1/datasets/MyoPS_test/data_preprocessed"
    # Task112_MyoPS: use labelsTr(GTs) to crop and use predsTr(predictions) as coarse segmentation
    output_dir = "/mnt/39E12DAE493BA6C1/datasets/MyoPS_test/nnUNet_raw_data_base/nnUNet_raw_data/Task112_MyoPS"
    imagesTr_dir = os.path.join(root_dir, "imagesTr")
    labelsTr_dir = os.path.join(root_dir, "labelsTr")
    labelsTr_suffix = "gd"
    predsTr_dir = "myops/result_train_post/unet2d"
    imagesTr_output_dir = os.path.join(output_dir, "imagesTr")
    labelsTr_output_dir = os.path.join(output_dir, "labelsTr")
    crop_acc_mask(imagesTr_dir, imagesTr_output_dir, labelsTr_dir, labelsTr_suffix, labelsTr_output_dir)
    # In imagesTr_output_dir there are "XXXX_0003.nii.gz" derived from labelsTr(GTs) which should be replaced
    # with predsTr(predictions) in predsTr_dir
    json_dict = load_json(os.path.join(imagesTr_output_dir, "crop_information.json"))
    for label_path in json_dict.keys():
        crop_bbox_min = json_dict[label_path]["crop_bbox_min"]
        crop_bbox_max = json_dict[label_path]["crop_bbox_max"]
        label = label_path.split("/")[-1]
        pred = label.replace("_{0:}".format(labelsTr_suffix), "")
        image = label.replace("_{0:}".format(labelsTr_suffix), "_0003")
        pred_path = find(pred, predsTr_dir)
        print(pred, pred_path)
        pred_sitk = sitk.ReadImage(pred_path)
        pred_npy = sitk.GetArrayFromImage(pred_sitk)
        pred_output_npy = crop_ND_volume_with_bounding_box(pred_npy, crop_bbox_min, crop_bbox_max)
        save_cropped_array_as_nifty_volume(pred_output_npy, os.path.join(imagesTr_output_dir, image), pred_sitk)

    imagesTs_dir = os.path.join(root_dir, "imagesTs")
    imagesTs_output_dir = os.path.join(output_dir, "imagesTs")
    predsTs_dir = "myops/result_test_post/unet2d"
    crop_acc_mask(imagesTs_dir, imagesTs_output_dir, predsTs_dir, mask_suffix=None, masks_output_dir=None)

    pkl_path = "splits_final.pkl"
    pkl_dir = "/mnt/39E12DAE493BA6C1/datasets/MyoPS_test/nnUNet_preprocessed/Task112_MyoPS"
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    pkl = load_pickle(pkl_path)
    print(pkl)
    shutil.copy(pkl_path, pkl_dir)