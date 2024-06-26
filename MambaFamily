from __future__ import annotations
import math
import json
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


@dataclass
class MambaConfig:
    d_model: int  # D
    dec_in: int
    n_layers: int = 1
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    embed: str = 'timeF'
    freq: str = 'h'
    dropout: float = 0.3
    n_heads: int = 4
    d_layers: int = 1

    def __post_init__(self):
        self.d_inner = int(self.expand_factor * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaBlock(nn.Module):
    def __init__(self, args: MambaConfig):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n],
                                    dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.max_len = max_len
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_len)])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))
        self.position_encoding = nn.Embedding(max_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len, max_len):
        input_pos = torch.zeros((len(input_len), max_len), dtype=torch.int, device=input_len.device)
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                input_pos[ind, pos_ind - 1] = pos_ind
        return self.position_encoding(input_pos)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        # year_size =

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
        # self.year_embed = Embed()

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)
        # nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, visit_len, x_mark):
        x1 = self.value_embedding(x)
        x2 = self.temporal_embedding(x_mark)
        x3 = self.position_embedding(visit_len, x.shape[1])
        x = x1 + x2 +x3
        return self.dropout(x)


class MFModel(nn.Module):
    """
    MambaFormer
    """

    def __init__(self, config):
        super().__init__()
        configs = MambaConfig(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'],
                              dropout=config['dropout'], embed=config['embed'], freq=config['freq'],
                              n_heads=config['n_heads'], dec_in=config['dec_in'], d_layers=config['d_layers'])
        # d_model 输出的维度
        # Embedding
        self.configs = configs
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.mamba_preprocess = Mamba_Layer(MambaBlock(configs), configs.d_model)
        self.AM_layers = nn.ModuleList(
            [
                AM_Layer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    MambaBlock(configs),
                    configs.d_model,
                    configs.dropout
                )
                for i in range(configs.d_layers)
            ]
        )
        # 预测
        # self.out_proj = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_dec, visit_len, x_mark_dec, dec_self_mask=None):
        x = self.dec_embedding(x_dec, visit_len, x_mark_dec)
        # b,max_visit_len,d_model
        x = self.mamba_preprocess(x)
        for i in range(self.configs.d_layers):
            x = self.AM_layers[i](x, dec_self_mask)
        # (B,L,d_model)
        return x


class AMModel(nn.Module):
    """
        attention_mamba
    """

    def __init__(self, config):
        super().__init__()
        configs = MambaConfig(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'],
                              dropout=config['dropout'], embed=config['embed'], freq=config['freq'],
                              n_heads=config['n_heads'], dec_in=config['dec_in'], d_layers=config['d_layers'])
        # d_model 输出的维度
        # Embedding
        self.configs = configs
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.mamba_preprocess = Mamba_Layer(MambaBlock(configs), configs.d_model)
        self.AM_layers = nn.ModuleList(
            [
                AM_Layer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    MambaBlock(configs),
                    configs.d_model,
                    configs.dropout
                )
                for i in range(configs.d_layers)
            ]
        )
        # 预测
        # self.out_proj = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_dec, visit_len, x_mark_dec, dec_self_mask=None):
        x = self.dec_embedding(x_dec, visit_len, x_mark_dec)
        # b,max_visit_len,d_model
        for i in range(self.configs.d_layers):
            x = self.AM_layers[i](x, dec_self_mask)
        # (B,L,d_model)
        return x
class MambaModel(nn.Module):
    """
    Mamba
    """

    def __init__(self, config):
        super().__init__()
        configs = MambaConfig(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'],
                              dropout=config['dropout'], embed=config['embed'], freq=config['freq'],
                              n_heads=config['n_heads'], dec_in=config['dec_in'], d_layers=config['d_layers'])
        # d_model 输出的维度
        # Embedding
        self.configs = configs
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.mamba_layers = nn.ModuleList(
            [
                Mamba_Layer(
                    MambaBlock(configs),
                    configs.d_model
                )
                for i in range(configs.d_layers)
            ]
        )

    def forward(self, x_dec, visit_len, x_mark_dec, att_mask=None):
        x = self.dec_embedding(x_dec, visit_len, x_mark_dec)
        for i in range(self.configs.d_layers):
            x = self.mamba_layers[i](x)
        return x


class MAModel(nn.Module):
    """
        Mamba-attention
    """

    def __init__(self, config):
        super().__init__()
        configs = MambaConfig(d_model=config['d_model'], d_state=config['d_state'], d_conv=config['d_conv'],
                              dropout=config['dropout'], embed=config['embed'], freq=config['freq'],
                              n_heads=config['n_heads'], dec_in=config['dec_in'], d_layers=config['d_layers'])
        # d_model 输出的维度
        # Embedding
        self.configs = configs
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.MA_layers = nn.ModuleList(
            [
                MA_Layer(
                    MambaBlock(configs),
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.dropout
                )
                for i in range(configs.d_layers)
            ]
        )

    def forward(self, x_dec, visit_len, x_mark_dec, dec_self_mask=None):
        x = self.dec_embedding(x_dec, visit_len, x_mark_dec)
        for i in range(self.configs.d_layers):
            x = self.MA_layers[i](x, dec_self_mask)
        return x

class Decoder_wo_cross_Layer(nn.Module):
    def __init__(self, self_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(Decoder_wo_cross_Layer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, x_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        y = x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Decoder(nn.Module):
    """
    Decoder Only Transformer
    """

    def __init__(self, configs):
        super().__init__()
        # Embedding
        self.configs = configs
        self.dec_embedding = DataEmbedding(configs['dec_in'], configs['d_model'], configs['embed'], configs['freq'],
                                           configs['dropout'])
        self.decoder_layers = nn.ModuleList(
            [
                Decoder_wo_cross_Layer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs['dropout'], output_attention=False),
                        configs['d_model'], configs['n_heads']),
                    configs['d_model']
                )
                for i in range(configs['d_layers'])
            ]
        )

    def forward(self, x_dec, visit_len, x_mark_dec, att_mask=None):
        x = self.dec_embedding(x_dec, visit_len, x_mark_dec)
        for i in range(self.configs['d_layers']):
            x = self.decoder_layers[i](x, att_mask)
        return x

class AM_Layer(nn.Module):
    def __init__(self, self_attention, mamba, d_model, dropout):
        super(AM_Layer, self).__init__()
        self.self_attention = self_attention
        self.mamba = mamba
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)
        x = x + self.mamba(x)
        x = self.norm2(x)

        return x



class MA_Layer(nn.Module):
    def __init__(self, mamba, self_attention, d_model, dropout):
        super(MA_Layer, self).__init__()
        self.mamba = mamba
        self.self_attention = self_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask=None):
        x = x + self.mamba(x)
        x = self.norm1(x)

        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm2(x)

        return x


class Mamba_Layer(nn.Module):
    def __init__(self, mamba, d_model):
        super(Mamba_Layer, self).__init__()
        self.mamba = mamba
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mamba(x)
        x = self.norm(x)
        return x


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, -np.inf)
            else:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                attn_mask.expand(B, H, L, S)
                scores.masked_fill_(attn_mask, -np.inf)

        scores = torch.nan_to_num(scores)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
