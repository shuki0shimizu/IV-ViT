import torch
import torch.nn as nn
import comet
import argparse

from tqdm import tqdm
from util import accuracy
from comet_ml import Experiment


def val(args: argparse.ArgumentParser,
        model: nn.Module,
        val_loader_list: list,
        device: torch.device,
        dataset_name_list: list,
        criterion: nn.CrossEntropyLoss,
        step: int,
        experiment: Experiment,
        comet_log_dict: dict
        ):
    """_summary_

    Args:
        args (argparse.ArgumentParser): training argument
        model (nn.Module): learing model
        val_loader_list (list): validation dataloader list
        device (torch.device): cuda device setting
        dataset_name_list (list): dataset name list
        criterion (nn.CrossEntropyLoss): loss
        step (int): global step in training
        experiment (Experiment): comet experiment
        comet_log_dict (dict): comet experiment dict for top1, top5, loss
    """
    model.eval()

    with torch.no_grad():
        for i, loader in enumerate(val_loader_list):
            for val_batch in tqdm(loader, leave=False):
                if "video" in val_batch.keys():
                    inputs = val_batch["video"].to(device)
                    labels = val_batch["label"].to(device)
                else:
                    inputs = val_batch["image"].to(device)
                    labels = val_batch["label"].to(device)

                bs = inputs.size(0)
                val_outputs = model(inputs, dataset_name_list[i])
                loss = criterion(val_outputs, labels)
                top1, top5 = accuracy(val_outputs, labels, (1, 5))

                if args.use_comet:
                    comet_log_dict["val_top1"][i].update(top1, bs)
                    comet_log_dict["val_top5"][i].update(top5, bs)
                    comet_log_dict["val_loss"][i].update(loss, bs)

        if args.use_comet:
            log_list = [
                "train_top1",
                "train_top5",
                "train_loss",
                "val_top1",
                "val_top5",
                "val_loss"
            ]

            comet.log(
                log_list,
                comet_log_dict,
                dataset_name_list,
                experiment,
                step
            )

            comet.reset(args, comet_log_dict)
