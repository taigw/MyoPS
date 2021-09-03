import numpy as np
import os
import sys
import SimpleITK as sitk
from scipy import ndimage
from pymic.util.image_process import get_largest_component, convert_label
from pymic.io_my.image_read_write import save_cropped_array_as_nifty_volume

def postprocess(origin_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    preds_list = os.listdir(origin_folder)
    for pred in preds_list:
        pred_path = os.path.join(origin_folder, pred)
        if pred.endswith(".nii.gz"):
            print(pred_path)
            pred_sitk = sitk.ReadImage(pred_path)
            pred_npy = sitk.GetArrayFromImage(pred_sitk)
            pred_binary = (pred_npy != 0)
            # get_largest_component for all_classes
            pred_binary_post = get_largest_component(pred_binary)
            pred_npy_post_all_classes = pred_npy * pred_binary_post
            # get_largest_componet for per_class(1, 2, 3)
            pred_npy_post_per_class = np.zeros_like(pred_npy_post_all_classes)
            for cls in range(1, 4):
                pred_binary = (pred_npy_post_all_classes == cls)
                pred_binary_post = get_largest_component(pred_binary)
                pred_npy_post_per_class += pred_npy_post_all_classes * pred_binary_post
            save_cropped_array_as_nifty_volume(pred_npy_post_per_class, os.path.join(output_folder, pred), pred_sitk)

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print('Number of arguments should be 2. e.g.')
        print('    python postprrcess.py train')
        exit()
    stage = str(sys.argv[1])
    if stage == "train":
        preds_train_dir = "myops/result_train/unet2d/fold_1"
        assert "fold_1" in preds_train_dir
        fold_num = 5
        for fold in range(fold_num):
            fold_dir = preds_train_dir.replace("fold_1", "fold_{0:}".format(fold + 1))
            fold_post_dir = fold_dir.replace("result_train", "result_train_post")
            postprocess(fold_dir, fold_post_dir)
    elif stage == "test":
        preds_test_dir = "myops/result_test/unet2d"
        preds_test_post_dir = "myops/result_test_post/unet2d"
        postprocess(preds_test_dir, preds_test_post_dir)
    else:
        raise ValueError
