import statistics
import torch
from collections import deque


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    https://github.com/machine-perception-robotics-group/attention_branch_network/blob/ced1d97303792ac6d56442571d71bb0572b3efd8/utils/misc.py#L59
    """

    def __init__(self, length=500):
        self.reset()
        self.reset_queue(length)
        self.avg_q = 0

    def reset_queue(self, length=500):
        self.val_q = deque([], length)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        self.val_q.append(val)
        self.avg_q = statistics.mean(self.val_q)


def create_log_variable(args):
    """
    to create comet experiment list following:
    train_acc_list, train_loss_list, train_top5_acc_list,
    val_acc_list, val_loss_list, val_top5_acc_list

    Args:
        args (argparse.ArgumentParser): training argument

    Returns:
        _type_: _description_
    """
    train_acc_list = []
    train_loss_list = []
    train_top5_acc_list = []
    val_acc_list = []
    val_loss_list = []
    val_top5_acc_list = []

    for _ in args.dataset_names:
        train_acc_list.append(AverageMeter(length=args.loss_weight_length))
        train_top5_acc_list.append(AverageMeter(length=args.loss_weight_length))
        train_loss_list.append(AverageMeter(length=args.loss_weight_length))
        val_acc_list.append(AverageMeter(length=args.loss_weight_length))
        val_top5_acc_list.append(AverageMeter(length=args.loss_weight_length))
        val_loss_list.append(AverageMeter(length=args.loss_weight_length))

    comet_log_dict = {
        "train_top1": train_acc_list,
        "train_top5": train_top5_acc_list,
        "train_loss": train_loss_list,
        "val_top1": val_acc_list,
        "val_top5": val_top5_acc_list,
        "val_loss": val_loss_list
    }

    return comet_log_dict


def log(log_list, comet_log_dict, dataset_name_list, experiment, step):
    """
    you can update experiment in comet_log_dict to select update type

    Args:
        log_list (list): update type list
        comet_log_dict (dict): comet experiment dict for top1, top5, loss
        dataset_name_list (list): dataset name list
        experiment (Experiment): comet experiment
        step (int): global step
    """

    for i, name in enumerate(dataset_name_list):
        for name in log_list:
            if name == "batch_accuracy":
                experiment.log_metric(
                    "train_accuracy_" + name,
                    comet_log_dict["train_top1"][i].val,
                    step=step,
                )
            elif name == "batch_top5_accuracy":
                experiment.log_metric(
                    "batch_top5_accuracy_" + name,
                    comet_log_dict["train_top5"][i].val,
                    step=step,
                )
            elif name == "batch_loss":
                experiment.log_metric(
                    "batch_loss_" + name,
                    comet_log_dict["train_loss"][i].val,
                    step=step
                )
            elif name == "train_top1":
                experiment.log_metric(
                    "train_accuracy_" + name,
                    comet_log_dict[name][i].avg,
                    step=step,
                )
            elif name == "train_top5":
                experiment.log_metric(
                    "train_top5_accuracy_" + name,
                    comet_log_dict[name][i].avg,
                    step=step,
                )
            elif name == "train_loss":
                experiment.log_metric(
                    "train_loss_" + name,
                    comet_log_dict[name][i].avg,
                    step=step
                )
            elif name == "val_top1":
                experiment.log_metric(
                    "val_accuracy_" + name,
                    comet_log_dict[name][i].avg,
                    step=step
                )
            elif name == "val_top5":
                experiment.log_metric(
                    "val_top5_accuracy_" + name,
                    comet_log_dict[name][i].avg,
                    step=step,
                )
            elif name == "val_loss":
                experiment.log_metric(
                    "val_loss_" + name,
                    comet_log_dict[name][i].avg,
                    step=step
                )


def reset(args, comet_log_dict):
    """
    you can reset experiment in comet_log_dict

    Args:
        args (argparse.ArgumentParser): training argument
        comet_log_dict (dict): comet experiment dict for top1, top5, loss
    """

    for dict_name in comet_log_dict.keys():
        for i, name in enumerate(args.dataset_names):
            comet_log_dict[dict_name][i].reset()
