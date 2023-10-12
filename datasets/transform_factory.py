from torchvision import transforms

from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

import torchvision.transforms as trans_image
import pytorchvideo.transforms as trans_video


def get_transform(args, config, subset, dataset_name):
    """
    create transforms for each dataset.

    Args:
        args (argparse.ArgumentParser): training argument
        config (configparser.ConfigParser): training config
        subset (string): "train" or "val"
        dataset_name (list[string]): dataset name list: [dataset1, dataset2,...]

    Returns:
        list: transform list of the dataset
    """

    if dataset_name in ["Tiny-ImageNet", "CIFAR100"]:
        transform = get_image_transform(subset)
    elif dataset_name in ["UCF101", "Kinetics200"]:
        transform = get_video_transform(args, config, subset, dataset_name)

    return transform


def get_image_transform(subset):
    """
    to get transform for image

    Args:
        subset (string): "train" or "val"

    Returns:
        list: transform list of the dataset
    """

    transforms_list = []

    if subset == 'train':
        transforms_list.extend([
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            trans_image.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ])
    elif subset == "val":
        transforms_list.extend([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            trans_image.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ])

    return Compose(transforms_list)


def get_video_transform(args, config, subset, dataset_name):
    """
    to get transform for video

    Args:
        args (argparse.ArgumentParser): training argument
        config (configparser.ConfigParser): training config
        subset (string): "train" or "val"
        dataset_name (list[string]): dataset name list: [dataset1, dataset2,...]

    Returns:
        list: transform list of the dataset
    """

    video_key_transform = [
        UniformTemporalSubsample(
            int(config[args.model]["num_frames"])),
        transforms.Lambda(lambda x: x / 255.),
        trans_video.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
    ]

    if subset == 'train':  # set train/val transform
        video_key_transform.extend([
            RandomShortSideScale(min_size=256, max_size=320,),
            RandomCrop(224),
            RandomHorizontalFlip(),
        ])
    elif subset == "val":
        video_key_transform.extend([
            ShortSideScale(256),
            CenterCrop(224)
        ])

    transforms_list = [
        ApplyTransformToKey(
            key="video",
            transform=Compose(video_key_transform),
        ),
        RemoveKey("audio"),
    ]

    if dataset_name == 'UCF101':  # set only UCF101 transform
        transforms_list.append(
            ApplyTransformToKey(
                key="label",
                transform=transforms.Lambda(lambda x: x - 1),
            ))

    return Compose(transforms_list)
