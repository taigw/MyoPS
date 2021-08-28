"""Script for writing cvs files
"""

import os
import csv
import pandas as pd
import random
from random import shuffle

def create_csv_file(data_root, output_file, fields, modality):
    """
    create a csv file to store the paths of files for each patient
    """
    filenames = []
    patient_names = os.listdir(data_root + '/' + fields[0])
    patient_names.sort()
    print(patient_names)
    print('total number of images {0:}'.format(len(patient_names)))
    for i, patient_name in enumerate(patient_names):
        # patient_image_names = []
        # for field in fields:
        #     image_name = field + '/' + patient_name
        #     if field == "labelsTs":
        #         for modal in modality:
        #             tmp_image_name = image_name.replace("_gd.nii.gz", "_%s.nii.gz" % modal)
        #             patient_image_names.append(tmp_image_name)
        #     else:
        #         patient_image_names.append(image_name)
        if i % 3 == 0:
            patient_image_names = []
        temp_image_name = fields[0] + "/" + patient_name
        patient_image_names.append(temp_image_name)
        if i % 3 == 2:
            filenames.append(patient_image_names)

    with open(output_file, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        print(fields[0] * len(modality) + fields[1])
        csv_writer.writerow([fields[0]] * len(modality) + [fields[1]])
        for item in filenames:
            csv_writer.writerow(item)
    obtain_patient_names(output_file)

def random_split_dataset(output_dir, fold_num):
    random.seed(2019)
    input_file = output_dir + "/image_all.csv"
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    shuffle(data_lines)
    N = len(data_lines)
    N_single_fold = N // fold_num
    for fold in range(fold_num):
        fold_dir = os.path.join(output_dir, "fold_%s" % str(fold + 1))
        if not os.path.isdir(fold_dir):
            os.makedirs(fold_dir)
        if fold == fold_num - 1:
            test_lines = data_lines[fold * N_single_fold:]
        else:
            test_lines = data_lines[fold * N_single_fold : (fold + 1) * N_single_fold]
        train_lines = [i for i in data_lines if i not in test_lines]
        
        with open(fold_dir + "/image_train.csv", 'w') as f:
            f.writelines(lines[:1] + train_lines)
        with open(fold_dir + "/image_valid.csv", 'w') as f:
            f.writelines(lines[:1] + train_lines)
        with open(fold_dir + "/image_test.csv", 'w') as f:
            f.writelines(lines[:1] + test_lines)
        obtain_patient_names(fold_dir + "/image_test.csv")

def obtain_patient_names(csv_file):
    """
    extract the patient names from csv files
    """
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    patient_names = []
    for data_line in data_lines:
        patient_name = data_line.split('/')[-1]
        patient_name = patient_name.split('.')[0]
        patient_name = patient_name.split("_T2")[0]
        print(patient_name)
        patient_names.append(patient_name)
    output_filename = csv_file.replace(".csv", "_names.txt")
    with open(output_filename, 'w') as f:
        for patient_name in patient_names:
            f.write('{0:}\n'.format(patient_name))
        
if __name__ == "__main__":
    # create cvs file for promise 2012
    fields      = ['imagesTs', 'labelsTs']
    data_dir    = '/home/c1501/swzhai/datasets/MyoPS2020'
    output_file = 'myops/config/data/image_test.csv'
    modality = ["C0", "DE", "T2"]
    create_csv_file(data_dir, output_file, fields, modality)

    # split the data into training, validation and testing
    # output_dir = "myops/config/data"
    # fold_num = 5
    # random_split_dataset(output_dir, fold_num)
