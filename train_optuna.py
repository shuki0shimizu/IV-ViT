
import torch
import torch.nn as nn
import comet
from comet_ml import Experiment
from tqdm import tqdm
from models.vit_video import load_model
from train_one_iter import train_one_iter
from datasets.dataset import make_loader_list
from validation import val
from util import make_optimizer_op, make_scheduler_op


def fit(args, config, trial):
    """
    search hyperparameters by optuna

    Args:
        args (argparse.ArgumentParser): training argument
        config (configparser.ConfigParser): training config
        trial (optuna.trial): optuna containing module

    Returns:
        tensor: validation loss
    """

    dataset_name_list = args.dataset_names

    train_loader_list, val_loader_list = make_loader_list(args, config)
    train_loader_itr_list = [iter(d) for d in train_loader_list]

    model = load_model(args, config)
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer_dict = make_optimizer_op(trial, dataset_name_list, config, model)
    scheduler_list = make_scheduler_op(trial, optimizer_dict, config, args)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    torch.backends.cudnn.benchmark = True

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

        comet_log_dict = comet.create_log_vriable()

    step = 0
    num_iters = args.iteration

    if args.task_prio:
        loss_weights = []
        tp_weight = 1
        for _ in dataset_name_list:
            loss_weights.append(1)
    elif args.dyna_weight:
        loss_weights = []
        tau = []
        for _ in dataset_name_list:
            loss_weights.append(1)
            tau.append(1)

    with tqdm(range(num_iters)) as pbar_itrs:
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
                comet_log_dict,
                loss_weights,
                tp_weight,
                scheduler_list,
                tau,
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
                    comet_log_dict,
                    step,
                    experiment,
                    comet_log_dict
                )
    experiment.end()

    return comet_log_dict['val_loss'][0].avg
