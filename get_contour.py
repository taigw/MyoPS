import os
import cv2
from pymic.util.image_process import get_contour, convert_label
from pymic.io_my.image_read_write import save_cropped_array_as_nifty_volume
import SimpleITK as sitk
import numpy as np

if __name__ == "__main__":
    input_dir = "/mnt/39E12DAE493BA6C1/datasets/MyoPS_test/data_preprocessed/labelsTr"
    output_dir = "/mnt/39E12DAE493BA6C1/datasets/MyoPS_test/data_preprocessed/contoursTr"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    labels_list = os.listdir(input_dir)
    # v1: get contour through difference
    # for label in labels_list:
    #     if label.endswith(".nii.gz"):
    #         contour = label.replace("_gd", "_contour")
    #         label_path = os.path.join(input_dir, label)
    #         label_sitk = sitk.ReadImage(label_path)
    #         label_npy = sitk.GetArrayFromImage(label_sitk)
    #         label_npy = convert_label(label_npy, [1, 2, 3, 4, 5], [1, 0, 0, 1, 1])
    #         depth = label_npy.shape[0]
    #         contour_npy = np.zeros_like(label_npy)
    #         for d in range(depth):
    #             label_slice = label_npy[d, :, :]
    #             contour_slice = get_contour(label_slice)
    #             contour_npy[d, :, :] = contour_slice
    #         save_cropped_array_as_nifty_volume(contour_npy, os.path.join(output_dir, contour), label_sitk)
    
    # v2: get contour through dilate
    for label in labels_list:
        if label.endswith(".nii.gz"):
            contour = label.replace("_gd", "_contour")
            label_path = os.path.join(input_dir, label)
            label_sitk = sitk.ReadImage(label_path)
            label_npy = sitk.GetArrayFromImage(label_sitk)
            label_npy = convert_label(label_npy, [1, 2, 3, 4, 5], [1, 0, 0, 1, 1])
            depth = label_npy.shape[0]
            contour_npy = np.zeros_like(label_npy)
            for d in range(depth):
                label_slice = label_npy[d, :, :]
                struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                label_slice_dilate = cv2.dilate(label_slice, struct)
                label_slice_erode = cv2.erode(label_slice, struct)
                contour_npy[d, :, :] = label_slice_dilate - label_slice_erode
            save_cropped_array_as_nifty_volume(contour_npy, os.path.join(output_dir, contour), label_sitk)