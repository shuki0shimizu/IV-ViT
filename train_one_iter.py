import torch
import comet
import torch.nn as nn
import argparse

from util import accuracy
from comet_ml import Experiment


def train_one_iter(
    train_loader_itr_list: list,
    train_loader_list: list,
    optimizer_dict: dict,
    device: torch.device,
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    dataset_name_list: list,
    args: argparse.ArgumentParser,
    scheduler_list: list,
    scaler: torch.cuda.amp.GradScaler,
    experiment: Experiment,
    step: int,
    comet_log_dict: dict
):
    """_summary_

    Args:
        train_loader_itr_list (list): train dataloader iterator list
        train_loader_list (list): train dataloader list
        optimizer_dict (dict): optimizer for each dataset
        device (torch.device): cuda device setting
        model (nn.Module): learing model
        criterion (nn.CrossEntropyLoss): loss
        dataset_name_list (list): dataset name list
        args (argparse.ArgumentParser): training argument
        scheduler_list (list): scheduler list for each dataset
        scaler (torch.cuda.amp.GradScaler): you can calculate single precision
        experiment (Experiment): comet experiment
        step (int): global step in training
        comet_log_dict (dict): comet experiment dict for top1, top5, loss
    """

    batch_list = []

    # make batch for each datasets
    for i, loader in enumerate(train_loader_itr_list):
        try:
            batch = next(loader)
            batch_list.append(batch)
        except StopIteration:
            train_loader_itr_list[i] = iter(train_loader_list[i])
            batch = next(train_loader_itr_list[i])
            batch_list.append(batch)

    # zero_grad() for each dataset optimizers
    for name in optimizer_dict.keys():
        optimizer_dict[name].zero_grad()

    # train for each dataset batches
    for i, batch in enumerate(batch_list):
        with torch.cuda.amp.autocast():
            if "video" in batch.keys():
                inputs = batch["video"].to(device)
                labels = batch["label"].to(device)
            else:
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)

            bs = inputs.size(0)
            outputs = model(inputs, dataset_name_list[i])
            loss = criterion(outputs, labels)

        top1, top5 = accuracy(outputs, labels, topk=(1, 5))

        scaler.scale(loss).backward()

        # comet update
        if args.use_comet:
            comet_log_dict["train_top1"][i].update(top1, bs)
            comet_log_dict["train_top5"][i].update(top5, bs)
            comet_log_dict["train_loss"][i].update(loss, bs)

        scaler.step(optimizer_dict[dataset_name_list[i]])

    for name in scheduler_list.keys():
        scheduler_list[name].step()

    if args.use_comet:
        log_list = [
            "batch_accuracy",
            "batch_top5_accuracy",
            "batch_loss"
        ]

        comet.log(
            log_list,
            comet_log_dict,
            dataset_name_list,
            experiment,
            step
        )
