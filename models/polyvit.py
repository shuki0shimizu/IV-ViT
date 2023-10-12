import torch
import torch.nn as nn
import math
import argparse

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Optional
from models.pretrain import load_pretrain


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length_image: int,
        seq_length_video: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(
            torch.empty(
                1, seq_length_image, hidden_dim).normal_(
                std=0.02))  # from BERT
        self.pos_embedding_video = nn.Parameter(
            torch.empty(
                1, seq_length_video, hidden_dim).normal_(
                std=0.02))
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, is_video):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        if is_video:
            input = input + self.pos_embedding_video
        else:
            input = input + self.pos_embedding

        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        patch_size_frame: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # _log_api_usage_once(self)
        torch._assert(
            image_size %
            patch_size == 0,
            "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_size_frame = patch_size_frame
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        input_channels = 3

        # The conv_proj is a more efficient version of reshaping, permuting
        # and projecting the input
        self.conv_proj = nn.Conv2d(
            input_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size)

        self.conv3d_proj = nn.Conv3d(
            input_channels,
            hidden_dim,
            kernel_size=(patch_size_frame, patch_size, patch_size),
            stride=(patch_size_frame, patch_size, patch_size))

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        seq_length_image = (image_size // patch_size) ** 2
        seq_length_video = ((image_size // patch_size) ** 2) * (32 // patch_size_frame)

        seq_length_image += 1
        seq_length_video += 1

        self.encoder = Encoder(
            seq_length_image,
            seq_length_video,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length_image = seq_length_image
        self.seq_length_video = seq_length_video

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(
                hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)
        self._init_weights()

    def _init_weights(self):
        fan_in = self.conv_proj.in_channels * \
            self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(self.conv_proj.bias)

        fan_in = self.conv3d_proj.in_channels * \
            self.conv3d_proj.kernel_size[0] * self.conv3d_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv3d_proj.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(self.conv3d_proj.bias)

        if hasattr(self.heads, "pre_logits"):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(
                self.heads.pre_logits.weight,
                std=math.sqrt(
                    1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        nn.init.zeros_(self.heads.head.weight)
        nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor, domain=None, is_video=False) -> torch.Tensor:
        p = self.patch_size
        p_t = self.patch_size_frame

        if is_video:
            bs, c, t, h, w = x.shape
            n_t = t // p_t
        else:
            bs, c, h, w = x.shape

        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p
        n_w = w // p

        # x = self.conv_proj(x)

        if is_video:
            # bs, t(n_t p_t), c, h(n_h p1), w(n_w p2) -> bs, hidden_dim, n_t, n_h, n_w
            x = self.conv3d_proj(x)
            # (bs, hidden_dim, n_h, n_w) -> (bs, hidden_dim, (n_h * n_w))
            x = x.reshape(bs, self.hidden_dim, n_t * n_h * n_w)
        else:
            # bs, c, h(n_h p1) w(n_w p2) -> bs, hidden_dim, n_h, n_w
            x = self.conv_proj(x)
            # (bs, hidden_dim, n_h, n_w) -> (bs, hidden_dim, (n_h * n_w))
            x = x.reshape(bs, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor, domain=None, is_video=False):

        # Reshape and permute the input tensor
        if domain is not None:
            x = self._process_input(x, domain, is_video)
        else:
            x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x, is_video)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        if domain is not None:
            x = self.heads(x, domain)
        else:
            x = self.heads(x)

        return x


def _vision_transformer(
    arch: str,
    patch_size: int,
    patch_size_frame: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    args: argparse.ArgumentParser,
    **kwargs: Any,
) -> VisionTransformer:
    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        patch_size_frame=patch_size_frame,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if args.pretrain != "None":
        state_dict = load_pretrain(args)
        model.load_state_dict(state_dict, strict=False)

    return model


def polyvit_b_16(args, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vision_transformer(
        arch="vit_b_16",
        patch_size=16,
        patch_size_frame=2,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        args=args,
        **kwargs,
    )
