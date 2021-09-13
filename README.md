# MyoPS 2020 Challenge
[PyMIC_link]:https://github.com/HiLab-git/PyMIC
[nnUNet_link]:https://github.com/MIC-DKFZ/nnUNet
This repository provides source code for myocardial pathology segmentation. The method is detailed in the [paper](https://link.springer.com/chapter/10.1007/978-3-030-65651-5_5), and it won the 1st place of [MyoPS 2020](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/myops20). Our code is adapted from [PyMIC][PyMIC_link], a pytorch-based toolkit for medical image computing with deep learning, that is lightweight and easy to use, and [nnUNet][nnUNet_link], a self-adaptive segmentation method for medical images. We used an older version of [PyMIC][PyMIC_link] and changed a small part of code, so the package may be different from recent releases.
## Dataset
Download the dataset from [MyoPS 2020](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/myops20) and put the dataset in the `DataDir` such as `/mnt/data1/swzhai/dataset/MyoPS`, specifically, `DataDir/data_raw/imagesTr` for training images, `DataDir/data_raw/labelsTr` for training ground truth and `DataDir/data_raw/imagesTs` for test images.

Get the maximal bounding box according to the training ground truth. Crop all images and ground truth with the maximal bounding box and save them in `DataDir/data_preprocessed/imagesTr`, `DataDir/data_preprocessed/labelsTr` and `DataDir/data_preprocessed/imagesTs` respectively. `crop_information.json` in each folder contains bounding box coordinates that will be used at fine segmentation stage. Run:
```python
python find_bbox_of_dataset.py
```
## Requirements
Some important required packages include(our test environment):
* [Pytorch](https://pytorch.org) == 1.9.0
* Python == 3.8.10
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......
* Download [nnUNet][nnUNet_link] and [PyMIC][PyMIC_link], and put them in the `ProjectDir` such as `/mnt/data1/swzhai/projects/MyoPS`.
## Set environment variables
* Set python path for [PyMIC][PyMIC_link]. Run:
``` bash
export PYTHONPATH=$PYTHONPATH:ProjectDir
```
* Install [nnUNet][nnUNet_link] and set environment variables.
```bash
cd nnUNet
pip install -e .
export nnUNet_raw_data_base="DataDir/nnUNet_raw_data_base"
export nnUNet_preprocessed="DataDir/nnUNet_preprocessed"
export RESULTS_FOLDER="ProjectDir/myops/result_nnunet"


# in my case
export nnUNet_raw_data_base="/mnt/data1/swzhai/dataset/MyoPS/nnUNet_raw_data_base"
export nnUNet_preprocessed="/mnt/data1/swzhai/dataset/MyoPS/nnUNet_preprocessed"
export RESULTS_FOLDER="/mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet"
```

## Coarse segmentation
We adopt a coarse-to-fine method, [PyMIC][PyMIC_link] for coarse segmentation due to its legibility and expandability and [nnUNet][nnUNet_link] for fine segmentation.
### training
* Change "/fold_X" to "/fold_1", "/fold_2", "/fold_3", "/fold_4", "/fold_5" in turn in train.cfg and run the following two line of commands. You will get a model for each fold and corresponding predictions of validation dataset.
```python
python pymic/net_run/net_run.py train myops/config/train.cfg
python pymic/net_run/net_run.py test myops/config/train.cfg
```
* Get the largest connected component. Run:
```python
python postprocess.py train
```
* Here we have predictions of each fold as coarse segmentation in `myops/result_train_post`. Then crop the training images and training ground truth again. Run:
```python
python crop_acc_preds.py train
```
### training Dice scores
Get the Dice scores of 5 folds before and after postprocess and save them in `dice.csv` file. Set `pred_dir` and `gt_dir` and run:
```python
python get_metrics.py
```
My results are as follow:
|---|label_1|label_2|label_3|
|---|---|---|---|
|w/o pp|0.8709|0.9050|0.9076|
|w/ pp|0.8770|0.9117|0.9128|

### inference
Use the coarse model to infer test dataset. Run:
```python
python pymic/net_run/net_run.py ensemble myops/config/test.cfg
```
* Get the largest connected component. Run:
```python
python postprocess.py test
```
* Here we have predictions of test datset as coarse segmentation in `myops/result_test_post`. Then crop the test images again. Run:
```python
python crop_acc_preds.py test
```
## Fine segmentation
Here we get coarse segmentation results in `/mnt/data1/swzhai/dataset/MyoPS/nnUNet_raw_data_base`. Next, we use them as the "4th modality"(_0003) while the first 3 modalities are C0(_0000), DE(_0001) and T2(_0002). This section is highly dependent on [nnUNet][nnUNet_link], so to understand the following commands please refer to [nnUNet][nnUNet_link]. 

Tips: In order to save unnecessary time, you can change `self.max_num_epochs = 1000` to `self.max_num_epochs = 300` in `nnUNet/nnunet/training/network_training/nnUNetTrainerV2.py`.
### training
* Dataset conversion and preprocess. Run:
```python
python Task112_MyoPS.py
nnUNet_plan_and_preprocess -t 112 --verify_dataset_integrity
```
* Train 2D UNet. For FOLD in [0, 1, 2, 3, 4], run:
```python
nnUNet_train 2d nnUNetTrainerV2 Task112_MyoPS FOLD --npz
```
* Train 2.5D(3D) UNet. For FOLD in [0, 1, 2, 3, 4], run:
```python
nnUNet_train 3d_fullres nnUNetTrainerV2 Task112_MyoPS FOLD --npz
```
### training Dice scores
We can see multiple metrics in `summary.json` in the subfolders of `myops/result_nnunet`. My results are as follow:
|---|label_1|label_2|label_3|label_4|label_5|
|---|---|---|---|---|---|
|2D UNet(w/o pp)|0.7926|0.9208|0.9197|0.3873|0.6096|
|2D UNet(w/ pp)|0.7933|0.9211|0.9205|0.3873|0.6098|
|2.5D UNet(w/o pp)|0.7904|0.9206|0.9241|0.3644|0.6191|
|2.5D UNet(w/ pp)|0.7923|0.9206|0.9241|0.3644|0.6191|
|ensemble|0.8016|0.9244|0.9246|0.3933|0.6303|

### inference
Here we have 2 fine models(i.e. 2D UNet and 2.5D UNet). Run:
```python
nnUNet_find_best_configuration -m 2d 3d_fullres -t 112
```
The terminal will output some commands that are used to infer test dataset and get their ensemble. In my case, I get the following commands: 
```python
nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t Task112_MyoPS

nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL2 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task112_MyoPS

nnUNet_ensemble -f OUTPUT_FOLDER_MODEL1 OUTPUT_FOLDER_MODEL2 -o OUTPUT_FOLDER -pp /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/nnUNet/ensembles/Task112_MyoPS/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json


# in my case
nnUNet_predict -i /mnt/data1/swzhai/dataset/MyoPS/nnUNet_raw_data_base/nnUNet_raw_data/Task112_MyoPS/imagesTs -o /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/test_2D -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t Task112_MyoPS --save_npz

nnUNet_predict -i /mnt/data1/swzhai/dataset/MyoPS/nnUNet_raw_data_base/nnUNet_raw_data/Task112_MyoPS/imagesTs -o /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/test_3D -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task112_MyoPS --save_npz

nnUNet_ensemble -f /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/test_2D /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/test_3D -o /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/test_ensemble -pp /mnt/data1/swzhai/projects/MyoPS/myops/result_nnunet/nnUNet/ensembles/Task112_MyoPS/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json --npz
```
Replace `FOLDER_WITH_TEST_CASES` with the test dataset folder `DataDir/nnUNet_raw_data_base/nnUNet_raw_data/Task112_MyoPS/imagesTs`, replace `OUTPUT_FOLDER_MODEL1` with 2D model folder `ProjectDir/myops/result_nnunet/test_2D`, replace `OUTPUT_FOLDER_MODEL2` with 3D model folder `ProjectDir/myops/result_nnunet/test_3D`, replace `OUTPUT_FOLDER` with ensemble folder `ProjectDir/myops/result_nnunet/test_ensemble` and run above commands.

Notice: Add arguments "--save_npz" and "--npz" to save .npz file which are model probability for future ensemble.

Because we crop the images twice in the whole process, we need to insert the cropped images into the original images by using `crop_information.json`. Set your foler path and Run:
```python
python get_final_test.py
```
## Citation
```
@inproceedings{zhai2020myocardial,
  title={Myocardial edema and scar segmentation using a coarse-to-fine framework with weighted ensemble},
  author={Zhai, Shuwei and Gu, Ran and Lei, Wenhui and Wang, Guotai},
  booktitle={Myocardial Pathology Segmentation Combining Multi-Sequence CMR Challenge},
  pages={49--59},
  year={2020},
  organization={Springer}
}
```
***This README is to be improved and questions are welcome.***