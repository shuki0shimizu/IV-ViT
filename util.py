import math
import torch
import os
import os.path as osp

from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L411
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, filename, dir_name, ex_name):
    """_summary_

    Args:
        state (dict): model state dict
        filename (string): filename
        dir_name (string): directory path
        ex_name (string): experiment name
    """
    dir_name = osp.join(dir_name, ex_name)
    file_path = osp.join(dir_name, filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    torch.save(state.state_dict(), file_path)


def make_optimizer(dataset_list, config, model):
    """
    make optimizer dict like following:
    {"dataset1": optimizer1, "dataset2": optimizer2...}

    Args:
        dataset_list (list): dataset name list
        config (configparser.ConfigParser): training config
        model (nn.Module): training model

    Returns:
        dict: optimizer dict
    """
    opt_dict = {}

    for name in dataset_list:
        opt_name = config[name]["optimizer"]
        lr = float(config[name]["lr"])
        weight_decay = float(config[name]["weight_decay"])

        if opt_name == "SGD":
            opt_dict[name] = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        elif opt_name == "Adam":
            opt_dict[name] = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_name == "AdamW":
            opt_dict[name] = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
    return opt_dict


def make_scheduler(optimizer_dict, config, args):
    """
    make scheduler dict like following:
    {"dataset1": scheduler1, "dataset2": scheduler2...}

    Args:
        optimizer_dict (dict): optimizer dict
        config (configparser.ConfigParser): training config
        args (argparse.ArgumentParser): training argument


    Returns:
        dict: scheduler dict
    """
    sche_dict = {}

    for name in optimizer_dict.keys():
        sche_name = config[name]["scheduler"]

        if sche_name == "MultiStepLR":
            sche_dict[name] = MultiStepLR(
                optimizer_dict[name], args.sche_list, args.lr_gamma
            )
        elif sche_name == "CosineAnnealingLR":
            sche_dict[name] = CosineAnnealingLR(
                optimizer_dict[name], args.iteration
            )
    return sche_dict


def make_optimizer_op(trial, dataset_list, config, model):
    """
    make optimzier dict by optuna parameters

    Args:
        trial (optuna.trial): optuna containing module
        dataset_list (list): dataset name list
        config (configparser.ConfigParser): training config
        model (nn.Module): training model

    Returns:
        dict: optimizer dict
    """
    opt_dict = {}

    opt_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW'])
    lr = trial.suggest_uniform('lr', 1e-6, 1e-2)
    weight_decay = trial.suggest_uniform('weight decay', 5e-6, 5e-4)

    for name in dataset_list:

        if opt_name == "SGD":
            opt_dict[name] = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        elif opt_name == "Adam":
            opt_dict[name] = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_name == "AdamW":
            opt_dict[name] = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
    return opt_dict


def make_scheduler_op(trial, optimizer_dict, config, args):
    """
    make optimzier list by optuna parameters

    Args:
        trial (optuna.trial): optuna containing module
        optimizer_dict (dict): {"dataset": optimizer...} dict
        config (configparser.ConfigParser): training config
        args (argparse.ArgumentParser): training argument

    Returns:
        dict: scheduler dict
    """
    sche_dict = {}

    sche_name = trial.suggest_categorical('scheduler', ['MultiStepLR', 'CosineAnnealing'])

    for name in optimizer_dict.keys():

        if sche_name == "MultiStepLR":
            sche_dict[name] = MultiStepLR(
                optimizer_dict[name], args.sche_list, args.lr_gamma
            )
        elif sche_name == "CosineAnnealingLR":
            sche_dict[name] = CosineAnnealingLR(
                optimizer_dict[name], args.iteration
            )
    return sche_dict
