
import torch
import torch.nn as nn
import comet

from tqdm import tqdm
from comet_ml import Experiment

from validation import val
from train_one_iter import train_one_iter
from util import make_optimizer, make_scheduler
from models.vit_video import load_model
from datasets.dataset import make_loader_list


def fit(args, config):
    """
    training

    Args:
        args (argparse.ArgumentParser): training argument
        config (configparser.ConfigParser): training config
    """

    # set hyperparameters
    dataset_name_list = args.dataset_names

    train_loader_list, val_loader_list = make_loader_list(args, config)
    train_loader_itr_list = [iter(d) for d in train_loader_list]

    model = load_model(args, config)
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer_dict = make_optimizer(dataset_name_list, config, model)
    scheduler_list = make_scheduler(optimizer_dict, config, args)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    torch.backends.cudnn.benchmark = True

    # prepare comet variables
    if args.use_comet:
        hyper_params = {
            "Dataset": args.dataset_names,
            "Iteration": args.iteration,
            "batch_size": args.batch_size_list,
            "optimizer": optimizer_dict,
            "pretrain weight": args.pretrain,
            "model": args.model
        }

        experiment = Experiment()
        experiment.set_name(args.ex_name + "_train")
        experiment.add_tag(args.comet_tag)
        experiment.log_parameters(hyper_params)

        comet_log_dict = comet.create_log_variable(args)

    step = 0

    # training
    with tqdm(range(args.iteration)) as pbar_itrs:
        for itr in pbar_itrs:
            pbar_itrs.set_description("[Iteration %d]" % (itr))

            """Training mode"""

            model.train()

            train_one_iter(
                train_loader_itr_list,
                train_loader_list,
                optimizer_dict,
                device,
                model,
                criterion,
                dataset_name_list,
                args,
                scheduler_list,
                scaler,
                experiment,
                step,
                comet_log_dict
            )

            step += 1

            if (step) % args.val_per_iteration == 0:
                val(
                    args,
                    model,
                    val_loader_list,
                    device,
                    dataset_name_list,
                    criterion,
                    step,
                    experiment,
                    comet_log_dict
                )

    experiment.end()
