# MyoPS
Our code is adapted from [PyMIC](https://github.com/HiLab-git/PyMIC), a pytorch-based toolkit for medical image computing with deep learning, that is lightweight and easy to use, and [nnUNet](https://github.com/MIC-DKFZ/nnUNet), a self-adaptive segmentation method for medical image. We used an older version of [PyMIC](https://github.com/HiLab-git/PyMIC) and changed a small part of code, so the package may be different from recent releases.
## Dataset
Download the dataset from [MyoPS 2020](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/myops20) and put the dataset in a ```DATADIR``` such as ```/home/messi/dataset/MyoPS_test```, specifically, ```DATADIR/imagesTr``` for training image, ```DATADIR/labelsTr``` for training ground truth and ```DATADIR/imagesTs``` for test image.
## Train model
We adopt a coarse-to-fine method, [PyMIC](https://github.com/HiLab-git/PyMIC) for coarse segmentation due to its legibility and expandability and [nnUNet](https://github.com/MIC-DKFZ/nnUNet) for fine segmentation.
```
python find_bbox_of_dataset.py
export PYTHONPATH=$PYTHONPATH:/home/messi/projects/MyoPS/pymic
python pymic/net_run/net_run.py train myops/config/train.cfg
python pymic/net_run/net_run.py test myops/config/train.cfg
pyhton crop_acc_labelsTr.py
```
Here we get coarse segmentation results. Next, we use them as the "4th modality" while the first 3 modalities are C0, DE and T2. Please refer to README of [nnUNet](https://github.com/MIC-DKFZ/nnUNet) to get fine segmentation results.

***This README is to be improved and questions are welcome.***