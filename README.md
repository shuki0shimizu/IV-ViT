# 1. Joint learning of images and videos with a single Transformer

- [1. Joint learning of images and videos with a single Transformer](#1-joint-learning-of-images-and-videos-with-a-single-transformer)
  - [1.1. Preparing](#11-preparing)
    - [1.1.1. 1.datasets](#111-1datasets)
    - [1.1.2. pretrained weights](#112-pretrained-weights)
    - [1.1.3. library](#113-library)
  - [1.2. Quick Start](#12-quick-start)
    - [1.2.1. Training](#121-training)
    - [1.2.2. Serching hyperparameters](#122-serching-hyperparameters)
  - [1.3. Training (Detailed)](#13-training-detailed)
    - [1.3.1. example](#131-example)

This is an official repo of IV-ViT paper "Joint learning of images and videos with a single Vision Transformer".

```BibTeX
@inproceedings{Shimizu_MVA2023_IV_VIT,
  author       = {Shuki Shimizu and Toru Tamaki},
  title        = {Joint learning of images and videos with a single Vision Transformer},
  booktitle    = {18th International Conference on Machine Vision and Applications,
                  {MVA} 2023, Hamamatsu, Japan, July 23-25, 2023},
  pages        = {1--6},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.23919/MVA57639.2023.10215661>,
  doi          = {10.23919/MVA57639.2023.10215661},
}
```

- [IEEE open access](https://ieeexplore.ieee.org/document/10215661)
- [DOI:10.23919/MVA57639.2023.10215661](https://doi.org/10.23919/MVA57639.2023.10215661)
- [poster](https://drive.google.com/file/d/1uQ75VkOmpfuOc3aIpe4JpN5rZiMYFUa_/view)
- [arXiv:2308.10533](https://arxiv.org/abs/2308.10533)

## 1.1. Preparing

You can learn our proposed IV-ViT in various settings with this code.
You will need to make the following preparations.

1. prepare datasets
2. prepare pretrained weights
3. prepare library

### 1.1.1. 1.datasets

In this code, Tiny-ImageNet and CIFAR100 can be used as image datasets, and UCF101 and mini-Kinetics as video datasets.
You need to prepare the datasets under `datasets/`  with the following directory structure.

```text
datasets/
  ├─Tiny-ImageNet/
  │   └─tiny-imagenet/
  │       ├─train/
  │       │   ├─[category0]/
  │       │   ├─[category1]/
  │       │   ├─...
  │       │
  │       └─val/
  │           ├─[category0]/
  │           ├─[category1]/
  │           ├─...
  │
  ├─CIFAR100/
  │   └─cifar-100-python/
  │
  ├─UCF101/
  │   └─ucfTrainTestlist/
  │       ├─trainlist01.txt
  │       └─testlist01.txt
  │
  └─Kinetics200/
      ├─train/
      │   ├─[category0]/
      │   ├─[category1]/
      │   ├─...
      │
      └─val/
          ├─[category0]/
          ├─[category1]/
          ├─...
```

### 1.1.2. pretrained weights

In this paper we use multiple pretrained weights.
You will need to download the pretrained weights under `pretrained_weight/` with the following directory structure.

```text
pretrained_weight/
  ├─ImageNet21k/
  │   └─video_model_imagenet21k_pretrained.pth
  └─Kinetics400/
      └─video_model_kinetics_pretrained.pth
```

You can download each pretrained weight with the following command.

```bash
sh download.sh
```

### 1.1.3. library

You can install all libraries required by this code with the following command.

```bash
pip install -r requirements.txt
```

## 1.2. Quick Start

You can do two things with this code, model training and searching hyperparameters.

### 1.2.1. Training

```bash
python main.py --mode train
```

### 1.2.2. Serching hyperparameters

```bash
python main.py --mode optuna
```

## 1.3. Training (Detailed)

We use argument to manage the experimental setup. The following is a description of the main arguments (see args.py for details).

- `i (int)`: you can set training iteration.
- `bsl (list[int])`: you can set batch size for each datset. Then you must same order with dataset (`dn`).
- `dn (list[string])`: you can set dataset with following choices: [Tiny-ImageNet, CIFAR100, UCF101, Kinetics200].
- `model (string)`: you can set model with following choices: [IV-ViT, TokenShift, MSCA, polyvit].
- `pretrain (string)`: you can set pretrained weights with following choices: [Kinetics400, ImageNet-21k, ImageNet-1k, polyvit].
- `use_comet (bool)`: you can select if use comet or not by given `use_comet`.
- `root_path (string)`: you must set root path, e.g. `~/data_root/`.

### 1.3.1. example

For example, you want to train with following settings.

- iteration：10000
- batch size
  - Tiny-ImageNet：16
  - CIFAR100：16
  - UCF101：4
  - Kinetics200：4
- model：IV-ViT
- pretrain weight：Kinetics400
- you don't use comet
- root path：`~/data_root/`

Then, you execute the following command.

```bash
python main.py -i 10000 -bsl 16 16 4 4 -dn Tiny-ImageNet CIFAR100 UCF101 Kinetics200 --model IV-ViT --pretrain Kinetics400 --use_comet --root_path ~/data_root/
```
