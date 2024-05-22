r"""Decoder part of a transformer architecture.
Code adapted from https://nlp.seas.harvard.edu/annotated-transformer/.

Use https://pytorch.org/docs/stable/nn.html#transformer-layers instead.
"""
import copy
import math
from typing import Literal, Optional, Tuple

import torch
from torch import BoolTensor, Tensor, nn

from commonroad_geometric.common.torch_utils.helpers import assert_size


def clone_module(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(d_mask: int) -> BoolTensor:
    shape = (1, d_mask, d_mask)
    mask = torch.triu(torch.ones(shape), diagonal=1).type(torch.uint8)
    return mask == 0


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        dim: int,
        max_length: int,
        dropout: float,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(0, max_length).unsqueeze(-1)
        pos_factor = torch.exp(-math.log(1e4) / dim * torch.arange(0, dim, 2))

        pos_enc = torch.zeros((max_length, dim), dtype=torch.float32)
        pos_enc[:, 0::2] = torch.sin(pos * pos_factor)
        pos_enc[:, 1::2] = torch.cos(pos * pos_factor)
        # batch x tokens x dim
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer("positional_encoding", pos_enc)

    def forward(self, x: Tensor, mode: Literal["add", "concat"] = "add", d_concat: int = 0) -> Tensor:
        sequence_length = x.size(1)
        if mode == "add":
            x = x + self.positional_encoding[:, :sequence_length].requires_grad_(False)
        elif mode == "concat":
            assert d_concat > 0
            x = torch.cat([
                x,
                self.positional_encoding[:, :sequence_length, :d_concat].requires_grad_(False)
            ], dim=-1)
        return self.dropout(x)


class SublayerConnection(nn.Module):
    r"""Layer norm & residual connection"""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        x_norm = self.norm(x)
        x_sublayer = sublayer(x_norm)
        return x + self.dropout(x_sublayer)


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[BoolTensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[Tensor, Tensor]:
    # size of query, key, value: batch x head x token x dim
    dim = query.size(-1)
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim)
    # scores: batch x head x query tokens x key tokens
    if mask is not None:
        scores = scores.masked_fill(mask.logical_not(), 1e-9)

    attn = scores.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    return attn @ value, attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0
        self.d_head = d_model // heads
        self.heads = heads
        self.linears = clone_module(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[BoolTensor] = None) -> Tensor:
        # size of query, key, value: batch x tokens x dim
        if mask is not None:
            # batch x token -> batch x 1 (head) x token
            # same mask is applied to all attention heads
            mask = mask.unsqueeze(1)
        N = query.size(0)

        query, key, value = [
            lin(x).view(N, -1, self.heads, self.d_head).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # size of query, key, value: batch x head x tokens x head dim

        x, self_attention = attention(
            query=query, key=key, value=value,
            mask=mask,
            dropout=self.dropout,
        )

        x = x.transpose(1, 2).contiguous().view(N, -1, self.heads * self.d_head)
        x = self.linears[3](x)
        return x


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden, bias=True)
        self.linear2 = nn.Linear(d_hidden, d_model, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x).relu()
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        self_attention: MultiHeadedAttention,
        encoder_decoder_attention: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.feed_forward = feed_forward
        self.sublayers = clone_module(SublayerConnection(d_model=d_model, dropout=dropout), n=3)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        source_mask: BoolTensor,
        target_mask: BoolTensor,
    ) -> Tensor:
        # masked self-attention
        x = self.sublayers[0](
            x=x,
            sublayer=lambda x: self.self_attention(query=x, key=x, value=x, mask=target_mask),
        )
        # masked encoder-decoder attention
        x = self.sublayers[1](
            x=x,
            sublayer=lambda x: self.encoder_decoder_attention(query=x, key=memory, value=memory, mask=source_mask),
        )
        # feed forward
        x = self.sublayers[2](x=x, sublayer=self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, layer: DecoderLayer, num_layers: int):
        super().__init__()
        self.layers = clone_module(layer, num_layers)
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        source_mask: BoolTensor,
        target_mask: BoolTensor,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x=x, memory=memory, source_mask=source_mask, target_mask=target_mask)
        return self.norm(x)


class TransformerDecoderModel(nn.Module):

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        d_feedforward: int = 2048,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.decoder = Decoder(
            layer=DecoderLayer(
                d_model=d_model,
                self_attention=MultiHeadedAttention(d_model=d_model, heads=heads, dropout=dropout),
                encoder_decoder_attention=MultiHeadedAttention(d_model=d_model, heads=heads, dropout=dropout),
                feed_forward=PositionwiseFeedForward(d_model=d_model, d_hidden=d_feedforward, dropout=dropout),
                dropout=dropout,
            ),
            num_layers=num_layers,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    # source token sequence = input to transformer encoder
    # target token sequence = previous outputs from transformer decoder
    def forward(
        self,
        predictions: Tensor,  # batch x predicted tokens x d_model
        predictions_mask: BoolTensor,  # batch x predicted tokens x predicted tokens
        memory: Tensor,  # batch x memory tokens x d_model
        memory_mask: BoolTensor,  # batch x predicted tokens x memory tokens
    ) -> Tensor:
        N, L_pred, L_mem = predictions.size(0), predictions.size(1), memory.size(1)
        assert_size(predictions, (N, L_pred, self.d_model))
        assert_size(predictions_mask, (1, L_pred, L_pred))
        assert_size(memory, (N, L_mem, self.d_model))
        assert_size(memory_mask, (1, L_pred, L_mem))
        return self.decoder(
            x=predictions,
            memory=memory,
            source_mask=memory_mask,
            target_mask=predictions_mask,
        )
