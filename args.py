import argparse


def get_arguments():
    """
    args function

    Returns:
        argparse.ArgumentParser: training arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--iteration",
        type=int,
        default=14000,
        help="",
    )
    parser.add_argument(
        "-bsl",
        "--batch_size_list",
        nargs="*",
        default=[4],
        help="",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--sche_list",
        nargs="*", default=[200, 250],
        help="",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float, default=0.1,
        help="",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-5,
        help="",
    )
    parser.add_argument(
        "-dn",
        "--dataset_names",
        nargs="*",
        default=["UCF101"],
        help="set dataset name list",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="cuda:1",
        help="set GPU number",
    )
    parser.add_argument(
        "--ex_name",
        type=str,
        default="temp",
        help="set comet experiment name",
    )
    parser.add_argument(
        "-amp",
        "--mixed_precision",
        action="store_true",
        help=""
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "optuna"],
        help="set train mode",

    )
    parser.add_argument(
        "--use_comet",
        action="store_false",
        help="if you don't use comet, set this",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tokenshift",
        choices=["ViT", "tokenshift", "MSCA", "polyvit"],
        help="set model name: ViT, tokenshift, MSCA, polyvit",
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default="Kinetics400",
        choices=[
            "Kinetics400",
            "ImageNet-1k",
            "ImageNet-21k",
            "polyvit",
            "emb_scratch_K400",
            "None",
        ],
        help="set pretrain name"
    )
    parser.add_argument(
        "--p_K400_1",
        type=str,
        default="Kinetics400/image_model_kinetics_pretrained.pth",
        help="image model K400 pretrain path",
    )
    parser.add_argument(
        "--p_K400_2",
        type=str,
        default="Kinetics400/video_model_kinetics_pretrain.pth",
        help="video model K400 pretrain path",
    )
    parser.add_argument(
        "--p_I21K_1",
        type=str,
        default="ImageNet21k/image_model_imagenet21k_pretrained.pth",
        help="image model I21K pretrain path",
    )
    parser.add_argument(
        "--p_I21K_2",
        type=str,
        default="ImageNet21k/video_model_imagenet21k_pretrained.pthh",
        help="video model I21K pretrain path",
    )
    parser.add_argument(
        "--p_esK400_1",
        type=str,
        default="Kinetics400/embedding_scratch_tshift_K400.pth",
        help="image model emb_scratch_K400 pretrain path",
    )
    parser.add_argument(
        "--p_esK400_2",
        type=str,
        default="Kinetics400/embedding_scratch_tshift_K400_videover.pth",
        help="video model emb_scratch_K400 pretrain path",

    )
    parser.add_argument(
        "--p_poly",
        type=str,
        default="Kinetics400/polyvit_pretrain.pth",
        help="polyvit pretrain path (K400+I21k)",
    )
    parser.add_argument(
        "--comet_tag",
        type=str,
        help="set comet tag",
    )
    parser.add_argument(
        "--loss_weight_length",
        type=int,
        default=500,
        help="set DWA averaging kength",
    )
    parser.add_argument(
        "--val_per_iteration",
        type=int,
        default=10,
        help="set val interval iteration",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="set optuna number of trial",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        help="set root path like this: '~/2022_09_shimizu_MDL_image_video/'",
    )

    return parser.parse_args()
