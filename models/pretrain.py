import torch
from torchvision._internally_replaced_utils import load_state_dict_from_url

model_urls = {
    "vit_b_16": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
}


def load_pretrain(args):
    """load pretrain weight for each condition

    Args:
        args (argparse.ArgumentParser): training argument

    Returns:
        dict: model's state_dict
    """
    if args.model == "MSCA" or args.model == "tokenshift":
        if args.pretrain == "Kinetics400":
            state_dict = torch.load(
                args.root_path + "pretrained_weight/" + args.p_K400_2
            )
        elif args.pretrain == "ImageNet-21k":
            state_dict = torch.load(
                args.root_path + "pretrained_weight/" + args.p_I21K_2
            )
    return state_dict
