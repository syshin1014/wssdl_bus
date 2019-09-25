# Joint Weakly and Semi-Supervised Deep Learning for Localization and Classification of Masses in Breast Ultrasound Images

This is the code for ["Joint Weakly and Semi-Supervised Deep Learning for Localization and Classification of Masses in Breast Ultrasound Images"](https://ieeexplore.ieee.org/abstract/document/8471199).

This is basically based on a Tensorflow implementation of Faster R-CNN [(https://github.com/smallcorgi/Faster-RCNN_TF)](https://github.com/smallcorgi/Faster-RCNN_TF), which is adopted as the mass detector in the proposed general framework as a choice. Some part of the following descriptions might be a repetition of those in the repository.

## Dependency
* Tensorflow 1.12
* Cython 0.27.3
* easydict 1.7
* pyyaml 3.12
* scikit-image 0.14.2

## Installation
* Building Cython codes
```
cd $ROOT/code/lib
make
```

## Testing a Model
1. Download available trained models. [[OneDrive]](https://onedrive.live.com/?authkey=%21AD9rIvC4ejVaD0s&id=613AC2A23C01CB69%2185607&cid=613AC2A23C01CB69)
2. Run `$ROOT/code/main/test.py` with appropriate input arguments, including the path for the downloaded model.

## Training a Model
1. Download ImageNet pretrained models
   * VGG-16 : [https://github.com/smallcorgi/Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)
   * ResNet-18,34,50,101 : [[OneDrive]](https://onedrive.live.com/?authkey=%21AM51TLQkoYZH7KQ&id=613AC2A23C01CB69%2185606&cid=613AC2A23C01CB69)
2. Run `$ROOT/code/main/train.py` (combined mini-batch) or `$ROOT/code/main/train_alter.py` (alternating mini-batches) with appropriate input arguments, including the path for the downloaded pretrained model.

## SNUBH Dataset
We provide sample images corresponding to those in Fig. 6 of our paper. The original result images also can be found in `$ROOT/code/qual_res/fig6`.

## Citation
```
@article{shin_tmi19,
  author = {S. Y. {Shin} and S. {Lee} and I. D. {Yun} and S. M. {Kim} and K. M. {Lee}},
  journal = {IEEE Transactions on Medical Imaging},
  title = {Joint Weakly and Semi-Supervised Deep Learning for Localization and Classification of Masses in Breast Ultrasound Images},
  year = {2019},
  volume = {38},
  number = {3},
  pages = {762-774},
  doi = {10.1109/TMI.2018.2872031},
  month = {March},
}
```
