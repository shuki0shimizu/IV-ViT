import torch
import torch.nn as nn
import torchinfo
from collections import OrderedDict

from models import vit, polyvit
from models.pretrain import load_pretrain
import models.configs as configs
from models import shift_model


class MyHeadDict(nn.Module):
    """model head layers for each dataset"""

    def __init__(self, in_channel, dataset_names, class_dict):
        super().__init__()
        self.head = nn.ModuleDict({})

        head_dict = {}
        for name in dataset_names:
            head = nn.Linear(in_channel, class_dict[name])
            head_dict[name] = head
        self.head.update(head_dict)

    def forward(self, x, domain):
        x = self.head[domain](x)
        return x


def make_class_dict(args, config):
    """
    make dict of dataset name and number of  classes in each dataset like follow:
    {"dataset1": num_classes1, "dataset2", num_classes2...}

    Args:
        args (argparse.ArgumentParser): training argument
        config (configparser.ConfigParser): training config

    Returns:
        dict: {"dataset1": num_classes1, "dataset2", num_classes2...}
    """

    config.read("config.ini")
    num_class_dict = {}
    for name in args.dataset_names:
        num_class_dict[name] = int(config[name]["num_class"])
    return num_class_dict


def change_key(dic):
    """
    change dic.keys for match state_dict keys to model

    Args:
        dic (OrderedDict): _description_

    Returns:
        OrderedDict:
    """

    out = OrderedDict(
        (".".join(k.split(".")[2:]) if True else k, v) for k, v in dic.items()
    )
    return out


class SegmentConsensus(torch.nn.Module):
    """fusion layer of the temporal information"""

    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == "avg":
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == "identity":
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):
    """wrapping class of SegmentConsensus module"""

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != "rnn" else "identity"
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


class IVViT(nn.Module):
    """proposed model:IV-ViT"""

    def __init__(
        self,
        args,
        config,
        consensus_type="avg",
        hidden_dim=768,
    ):
        super(IVViT, self).__init__()

        self.base_model = vit.vit_b_16(args)
        self.class_dict = make_class_dict(args, config)
        self.num_segments = int(config[args.model]["num_frames"])
        self.consensus_type = consensus_type
        self.consensus = ConsensusModule(consensus_type)
        self.base_model.heads = MyHeadDict(
            hidden_dim, args.dataset_names, self.class_dict
        )

    def input_is_video(self, input):
        return len(input) != 4

    def forward(self, input, domain):
        is_video = False
        if self.input_is_video(input):
            is_video = True
            input = input.view((-1, 3) + input.size()[-2:])

        base_out = self.base_model(input, domain)

        if is_video:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            base_out = self.consensus(base_out)
            base_out = base_out.squeeze(1)
        return base_out


class MSCA(nn.Module):
    """MSCA model"""

    def __init__(self, args, config, consensus_type="avg", hidden_dim=768):
        super(MSCA, self).__init__()
        vit_conf = configs.get_b16_config()
        self.num_segments = int(vit_conf.n_seg)
        self.base_model = shift_model.VisionTransformer(config=vit_conf, shift_type="MSCA")

        if args.pretrain != "None":
            state_dict = load_pretrain(args)
            state_dict["state_dict"] = change_key(state_dict["state_dict"])
            self.base_model.load_state_dict(
                state_dict["state_dict"], strict=False
            )

        self.class_dict = make_class_dict(args, config)
        self.consensus_type = consensus_type
        self.consensus = ConsensusModule(consensus_type)
        self.base_model.head = MyHeadDict(
            hidden_dim, args.dataset_names, self.class_dict
        )

    def forward(self, input, domain):
        is_video = False
        if len(input.shape) != 4:
            is_video = True
            input = input.view((-1, 3) + input.size()[-2:])

        base_out, _ = self.base_model(input, domain, is_video)

        if is_video:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            base_out = self.consensus(base_out)
            base_out = base_out.squeeze(1)
        return base_out


class TokenShift(nn.Module):
    """TokenShift Transformer model"""

    def __init__(self, args, config, consensus_type="avg", hidden_dim=768):
        super(TokenShift, self).__init__()
        vit_conf = configs.get_b16_config()
        self.num_segments = int(vit_conf.n_seg)
        self.base_model = shift_model.VisionTransformer(config=vit_conf, shift_type="tokenshift")

        if args.pretrain != "None":
            state_dict = load_pretrain(args)
            state_dict["state_dict"] = change_key(state_dict["state_dict"])
            self.base_model.load_state_dict(
                state_dict["state_dict"], strict=False
            )  # ImageNet pretrain だとパスが良くなくてエラー

        self.class_dict = make_class_dict(args, config)
        self.consensus_type = consensus_type
        self.consensus = ConsensusModule(consensus_type)
        self.base_model.head = MyHeadDict(
            hidden_dim, args.dataset_names, self.class_dict
        )

    def forward(self, input, domain):
        is_video = False
        if len(input.shape) != 4:
            is_video = True
            input = input.view((-1, 3) + input.size()[-2:])

        base_out, _ = self.base_model(input, domain, is_video)

        if is_video:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            base_out = self.consensus(base_out)
            base_out = base_out.squeeze(1)
        return base_out


class PolyViT(nn.Module):
    """PolyViT model"""

    def __init__(self, args, config, hidden_dim=768):
        super(PolyViT, self).__init__()

        self.base_model = polyvit.polyvit_b_16(args)
        self.class_dict = make_class_dict(args, config)
        self.num_segments = int(config[args.model]["num_frames"])
        self.base_model.heads = MyHeadDict(
            hidden_dim, args.dataset_names, self.class_dict
        )

    def forward(self, input, domain):
        if domain == "UCF101" or domain == "Kinetics200":
            is_video = True
        else:
            is_video = False

        base_out = self.base_model(input, domain, is_video)
        return base_out


def load_model(args, config):
    """
    load model

    Args:
        args (argparse.ArgumentParser): training argument
        config (configparser.ConfigParser): training config

    Returns:
        nn.Module: base model
    """
    if args.model == "ViT":
        base_model = IVViT(args, config)
    elif args.model == "tokenshift":
        base_model = TokenShift(args, config)
    elif args.model == "MSCA":
        base_model = MSCA(args, config)
    elif args.model == "polyvit":
        base_model = PolyViT(args, config)

    return base_model


def test_frame_per_vit_model():
    """ test IV-ViT to work """

    model = IVViT(num_classes=51, num_frames=16)
    bs = 4
    input_size = (bs, 16, 3, 224, 224)
    torchinfo.summary(
        model=model,
        input_size=input_size,
        depth=8,
        col_names=["input_size", "output_size"],
    )
    dummy_input = torch.zeros(input_size)
    assert model(dummy_input).shape == (bs, 51)
