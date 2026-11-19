# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import socket
import time

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 2
    n_heads: int = 32
    n_kv_heads: Optional[int] = 4
    n_local_heads: int = 1  # 每个设备的头数目
    hidden_dim: int = 512  # 每个设备MLP的hidden_dim维度
    # vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 100


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
"""
def run_inference(model, tokens, start_pos):    
    # Establish the socket connection
        result = model(tokens, start_pos, s)
        
        # Optionally close the connection after the whole process is done
    
    
    return result
"""

def send_and_receive(output):
    t_transform_begin = time.perf_counter()
    output_16 = output.to(torch.float16)
    output_np = output_16.detach().cpu().numpy()
    t_transform_end = time.perf_counter()
    t_transform = t_transform_end-t_transform_begin
    print('result pre-deal time: ', t_transform)
        
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(('0.0.0.0', 12346))
        output_bytes = output_np.tobytes()
        total_size = len(output_bytes)
        buffer_size = 65503
        num_packets = (total_size + buffer_size - 1) // buffer_size

        t_attention_send_begin = time.perf_counter()
        for i in range(num_packets):
            start = i * buffer_size
            end = min(start + buffer_size,total_size)
            packet = output_bytes[start:end]
            sequence_number = i.to_bytes(4, byteorder='big')
            s.sendto(sequence_number + packet, ('172.31.69.240',12345))
        t_attention_send_end = time.perf_counter()
        t_attention_send = t_attention_send_end - t_attention_send_begin
        print('send time: ', t_attention_send)
            
        received_data = bytearray(total_size)

        received_packets = 0
            
        t_receive_begin = time.perf_counter()
        while received_packets < num_packets:
            packet, _ = s.recvfrom(65536)  # 包含序列号的数据包
            start = received_packets * buffer_size
            end = start + len(packet)
            received_data[start:end] = packet
            received_packets += 1
        t_receive_end = time.perf_counter()
        print('receive time: ',t_receive_end-t_receive_begin)

    t_de_begin = time.perf_counter()
    received_np = np.frombuffer(received_data, dtype=output_np.dtype).copy()
    received_output = torch.from_numpy(received_np).view_as(output).to(output.device)
    received_output = received_output.to(torch.bfloat16)
    t_de_end = time.perf_counter()
    print('de-serialized time: ', t_de_end-t_de_begin)
    return received_output

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_local_heads
        self.n_local_kv_heads = args.n_local_heads
        self.n_rep = self.n_local_heads*8 // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads//8

        self.wq = nn.Linear(
            args.dim,
            args.n_local_heads * self.head_dim*8,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_local_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_local_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_local_heads * self.head_dim*8,
            args.dim,
            bias=False,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        
        # x = x.to(torch.bfloat16)

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads*8, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        print('---attention---')
        output = send_and_receive(output)
        return output

class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        """
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        """
        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        output = self.w2(F.silu(self.w1(x)) * self.w3(x))  
        print('---MLP begins sending---')
        output = send_and_receive(output)
        return output
        

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,  # 修改的hidden_dim
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        # self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        """
        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )
        """

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        """
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )
        """

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads//8,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = torch.randn(_bsz,seqlen,self.params.dim,device=tokens.device,dtype=torch.bfloat16)
        # h = 从交换机接收embeddings(tokens)
        # h = self.receiveFromSwithcer()
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = mask.to(torch.float32)		
            mask = torch.triu(mask, diagonal=1)
            mask = mask.to('cuda', dtype = torch.bfloat16)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            t1=time.perf_counter()
            h = layer(h, start_pos, freqs_cis, mask)
            t2=time.perf_counter()
            print('one block time: ', t2-t1)
            print(' ')
        # h = self.norm(h)      交换机完成
        # output = self.output(h).float()   交换机完成
        # output = 交换机给的新的token序列
        # return output

    def receiveFromSwithcer(self) -> torch.Tensor:
        pass

torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)        
args = ModelArgs(dim=8192, n_layers=20, n_heads=64, n_kv_heads=8,n_local_heads=19, hidden_dim=8181)
model = Transformer(args)
print('model finished!!!')
model.load_state_dict(torch.load('device6.pth'))
print('load finished!!!')
model.eval()
tokens = torch.randint(0, 512, (1, args.max_seq_len))
# t1=time.perf_counter()
with torch.no_grad():
    start_pos = 0
    output = model(tokens, start_pos)
# t2=time.perf_counter()
# print(t2-t1)
# print(output)
