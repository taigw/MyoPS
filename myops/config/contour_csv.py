from genericpath import exists
import os
import csv
import shutil
import pandas as pd

if __name__ == "__main__":
    origin_dir = "/home/c1501/swzhai/projects/MyoPS_test/myops/config/data_train"
    output_dir = "/home/c1501/swzhai/projects/MyoPS_test/myops/config/data_train_contour"
    for i in range(5):
        origin_fold_dir = os.path.join(origin_dir, "fold_{0:}".format(i + 1))
        output_fold_dir = os.path.join(output_dir, "fold_{0:}".format(i + 1))
        if not os.path.exists(output_fold_dir):
            os.makedirs(output_fold_dir)
        files_list = os.listdir(origin_fold_dir)
        for file in files_list:
            origin_file_path = os.path.join(origin_fold_dir, file)
            output_file_path = os.path.join(output_fold_dir, file)
            print(origin_file_path)
            if file.endswith(".csv"):
                df = pd.read_csv(origin_file_path)
                df["contoursTr"] = df["labelsTr"]
                df["contoursTr"] = df["contoursTr"].map(lambda x: (x.replace("_gd", "_contour")))
                df["contoursTr"] = df["contoursTr"].map(lambda x: (x.replace("labelsTr", "contoursTr")))
                df.to_csv(output_file_path, index=False)
            if file.endswith(".txt"):
                shutil.copyfile(origin_file_path, output_file_path)