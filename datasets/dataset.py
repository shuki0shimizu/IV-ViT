import itertools
import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from torchvision.datasets import (
    ImageFolder,
    CIFAR100
)

from pytorchvideo.data import (
    Ucf101,
    RandomClipSampler,
    Kinetics,
)

from . import transform_factory


class MyCifar100(CIFAR100):
    """
    Overwraping CIFAR100 to change each batch shape following:
    (batch, label) -> {'image': batch, 'label': label}
    """

    def __getitem__(self, index):
        data, label = super().__getitem__(index)
        return {'image': data, 'label': label}


class DictImageFolder(ImageFolder):
    """
    Overwraping ImageFolader to change each batch shape following:
    (batch, label) -> {'image': batch, 'label': label}
    """

    def __getitem__(self, index):
        data, label = super().__getitem__(index)
        return {'image': data, 'label': label}


class LimitDataset(torch.utils.data.Dataset):
    """
    create two times dataset iterator by __getitem__ function.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        if hasattr(self.dataset, 'num_videos'):
            return self.dataset.num_videos
        else:
            return len(self.dataset)


def get_kinetics200(subset, args, config):
    """
    load Kinetics200 datasets

    Args:
        subset (string): dataset type "train" or "val"
        args (argparse.ArgumentParser): training argument
        config (configparser.ConfigParser): training config

    Returns:
        LabeledVideoDataset: Kinetics200 dataset
    """

    transform = transform_factory.get_transform(args, config, subset, "Kinetics200")

    root_path = args.root_path + 'datasets/Kinetics200/'

    dataset = Kinetics(
        data_path=root_path + subset,
        video_path_prefix=root_path + subset,
        clip_sampler=RandomClipSampler(
            clip_duration=80 / 30),
        video_sampler=RandomSampler,
        decode_audio=False,
        transform=transform,
    )

    return dataset


def get_ucf101(subset, args, config):
    """
    load UCF101 datasets

    Args:
        subset (string): dataset type "train" or "val"
        args (argparse.ArgumentParser): training argument
        config (configparser.ConfigParser): training config

    Returns:
        LabeledVideoDataset: UCF101 dataset
    """

    root_path = args.root_path + 'datasets/UCF101/'
    subset_path = 'ucfTrainTestlist/trainlist01.txt' if subset == "train" else 'ucfTrainTestlist/testlist.txt'

    transform = transform_factory.get_transform(args, config, subset, "UCF101")

    dataset = Ucf101(
        data_path=root_path + subset_path,
        video_path_prefix=root_path + 'video/',
        clip_sampler=RandomClipSampler(
            clip_duration=64 / 25),
        video_sampler=RandomSampler,
        decode_audio=False,
        transform=transform,
    )

    return dataset


def get_cifar100(subset, args):
    """
    load CIFAR100 datasets

    Args:
        subset (string): dataset type "train" or "val"

    Returns:
        DictImageFolder: CIFAR100 dataset
    """

    root_path = args.root_path + 'datasets/CIFAR100/'

    transform = transform_factory.get_transform(subset, "CIFAR100")

    subset_bool = False if subset == "val" else True

    dataset = MyCifar100(root=root_path, train=subset_bool, transform=transform)

    return dataset


def get_tinyimagenet(subset, args):
    """
    load Tiny-ImageNet datasets

    Args:
        subset (string): dataset type "train" or "val"

    Returns:
        DictImageFolder: Tiny-ImageNet dataset
    """

    path = args.root_path + 'datasets/Tiny-ImageNet/tiny-imagenet-200/' + subset

    transform = transform_factory.get_transform(subset, "Tiny-ImageNet")

    dataset = DictImageFolder(path, transform=transform)

    return dataset


def make_loader(dataset, args, config, batch_size, dataset_name):
    """
    make dataloader

    Args:
        dataset (LabeledVideoDataset or DictImageFolader):
            dataset loading in get_dataset()

    Returns:
        torch.utils.data.DataLoader: Dataloader of loaded dataset
    """

    if config[dataset_name]['modality'] == 'image':
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            drop_last=True,
                            num_workers=args.num_workers,
                            shuffle=True)
    else:
        loader = DataLoader(LimitDataset(dataset),
                            batch_size=batch_size,
                            drop_last=True,
                            num_workers=args.num_workers,
                            shuffle=False)
    return loader


def make_named_loader(dataset_name, subset, args, config, batch_size):
    """
    load dataset and make dataloader

    Args:
        dataset_name (str): dataset name to load
        subset (str): dataset type"train" or "val
        args (argparse.ArgumentParser): training argument
        config (configparser.ConfigParser): training config
        batch_size (int): batch size of the dataset

    Raises:
        NameError: if the selected dataset name is not existed, raises error.

    Returns:
        torch.utils.data.DataLoader: Dataloader of loaded dataset
    """

    if dataset_name == "Kinetics200":
        dataset = get_kinetics200(subset, args, config)
    elif dataset_name == "UCF101":
        dataset = get_ucf101(subset, args, config)
    elif dataset_name == "Tiny-ImageNet":
        dataset = get_tinyimagenet(subset, args)
    elif dataset_name == "CIFAR100":
        dataset = get_cifar100(subset, args)
    else:
        raise NameError("invalide dataset name")
    loader = make_loader(dataset, args, config, batch_size, dataset_name)
    return loader


def make_loader_list(args, config):
    """
    make dataloader list as following:
    [dataloader1, dataloader2, ...]

    Args:
        args (argparse.ArgumentParser): training argument
        config (configparser.ConfigParser): training config

    Returns:
        list: list of each dataset Dataloader
    """
    assert len(args.dataset_names) == len(args.batch_size_list), "dataset_names and batch_size_list is not same"

    train_loader_list = []
    val_loader_list = []
    dataset_list = args.dataset_names
    batch_size_dict = dict(
        zip(args.dataset_names, list(map(int, args.batch_size_list))))

    for dataset_name in dataset_list:
        train_loader_list.append(make_named_loader(
            dataset_name, "train", args, config, batch_size_dict[dataset_name]))
        val_loader_list.append(make_named_loader(
            dataset_name, "val", args, config, batch_size_dict[dataset_name]))

    return train_loader_list, val_loader_list
