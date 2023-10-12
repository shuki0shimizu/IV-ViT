import configparser
import train
import train_optuna
import optuna

from args import get_arguments


def objective_wrapper(args, config):
    """objective function for optuna

    Args:
        args (argparse): commandline argments
        config (ConfigParser): config object
    """

    def objective(trial):
        val_loss = train_optuna.fit(args, config, trial)

        return val_loss

    return objective


def main():
    """
    main function
    """

    args = get_arguments()

    # load config
    config = configparser.ConfigParser()
    config.read("config.ini")

    # train
    if args.mode == "train":
        train.fit(args, config)
    elif args.mode == "optuna":
        study = optuna.create_study()
        study.optimize(objective_wrapper(args, config), n_trials=args.n_trials)


if __name__ == "__main__":
    main()
