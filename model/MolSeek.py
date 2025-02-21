import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional

from torch.nn.utils.rnn import pad_sequence
from configs.ModelConfigs import ModelConfigs


class TopFrag(nn.Module):
    
    def __init__(self, args: ModelConfigs):
        super(TopFrag, self).__init__()
        
        self.args = args
        
        self.pro_emb = nn.Embedding(args.voc.pro_voc_len, args.dim, args.voc.pro_pad_idx)
        self.smi_emb = nn.Embedding(args.voc.smi_voc_len, args.dim, args.voc.smi_pad_idx)
       
        self.pro_pe = PositionalEncoding(args.dim, args.voc.pro_max_len, args.dropout)
        self.smi_pe = PositionalEncoding(args.dim, args.voc.smi_max_len, args.dropout)
        self.layers = torch.nn.ModuleList()
        
        for layerid in range(args.num_layers):
            self.layers.append(Block(args, layerid))
        
        self.d_norm = nn.RMSNorm(args.dim)
        
        self.linear = nn.Linear(args.dim, args.voc.smi_voc_len)

        self._reset_parameters()

        # self.to(args.device)

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_padding_mask(self, seq: Tensor, pad_idx: int) -> Tensor:
        return seq.ne(pad_idx)
    
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        # src and tgt are lists of tensors
        src = pad_sequence(src, batch_first=True, padding_value=self.args.pad_idx)
        tgt = pad_sequence(tgt, batch_first=True, padding_value=self.args.pad_idx)

        src_mask = self.make_padding_mask(src, self.args.pad_idx)
        tgt_mask = self.make_padding_mask(tgt, self.args.pad_idx)

        _, seq_len = tgt.shape

        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

        all_gate_indices: List[Tensor] = []
        
        src = self.pro_emb(src)
        src = self.pro_pe(src)
            
        tgt = self.smi_emb(tgt)
        tgt = self.smi_pe(tgt)

        e_out = src
        d_out = tgt
        
        for mod in self.layers:
            if self.training:
                e_out, d_out, gate_indices = mod(e_out, d_out, src_mask, tgt_mask, causal_mask)
                
                all_gate_indices.append(gate_indices)
            else:
                e_out, d_out = mod(e_out, d_out, src_mask, tgt_mask, causal_mask)
            
            # Normalize the encoder/decoder outputs
        d_out = self.d_norm(d_out)
            
        out = self.linear(d_out)
        
        out = F.log_softmax(out, dim=-1)
        
        if self.training:
            return out, all_gate_indices

        return out


class Block(nn.Module):
    def __init__(self, args: ModelConfigs, id: int):
        super(Block, self).__init__()

        self.encoder = EncoderBlock(id, args)
        self.decoder = DecoderBlock(id, args)

        self.isMoE = id > args.num_dense_layers

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor, causal_mask: Tensor) -> Tuple[Tensor, Tensor]:
        # Forward pass through encoder/decoder layers.
        if self.training and self.isMoE:
            all_gate_indices: List[Tensor] = []

            out1, gate_indices1 = self.encoder(src, src_mask)
            out2, gate_indices2 = self.decoder(out1, tgt, src_mask, tgt_mask, causal_mask)
            
            all_gate_indices.append((gate_indices1, gate_indices2))
        else:
            out1 = self.encoder(src)
            out2 = self.decoder(out1, tgt)

        if self.training:
            return out1, out2, all_gate_indices

        return out1, out2


class EncoderBlock(nn.Module):
    
    def __init__(self, id: int, args: ModelConfigs):
        super(EncoderBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(args.dim, args.nheads, dropout=args.dropout)
        self.feed_forward = FeedForward(args.dim, args.dim_feedforward) if id < args.num_dense_layers else MoELayer(args)
        self.norm1 = nn.RMSNorm(args.dim) 
        self.norm2 = nn.RMSNorm(args.dim)
        self.dropout = nn.Dropout(p=args.dropout)
        
        self.isMoE = id > args.num_dense_layers
        
    def forward(self, x: Tensor, src_mask: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        x = self.norm1(x)
        x = x + self.attention(x, x, x, key_padding_mask=src_mask)[0]

        if self.isMoE:
            out, gate_indices = self.feed_forward(self.norm2(x))
        else:
            out = self.feed_forward(self.norm2(x))

        x = x + out
        
        x = self.dropout(x)
        
        if self.training and self.isMoE:
            return x, gate_indices
        return x


class DecoderBlock(nn.Module):
    
    def __init__(self, id: int, args: ModelConfigs):
        super(DecoderBlock, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(args.dim, args.nheads, dropout=args.dropout)
        self.multi_attention = nn.MultiheadAttention(args.dim, args.nheads, dropout=args.dropout)
        self.feed_forward = FeedForward(args.dim, args.dim_feedforward) if id < args.num_dense_layers else MoELayer(args)
        self.norm1 = nn.RMSNorm(args.dim) 
        self.norm2 = nn.RMSNorm(args.dim)
        self.norm3 = nn.RMSNorm(args.dim)
        self.dropout = nn.Dropout(p=args.dropout)
        
        self.isMoE = id > args.num_dense_layers
        
    def forward(self, x: Tensor, mem: Tensor, src_mask: Tensor, tgt_mask: Tensor, causal_mask: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        x = self.norm1(x)
        x = x + self.self_attention(x, x, x, key_padding_mask=tgt_mask, attn_mask=causal_mask)[0]
        x = self.norm2(x)
        x = x + self.multi_attention(x, mem, mem, key_padding_mask=src_mask)[0]

        if self.isMoE:
            out, gate_indices = self.feed_forward(self.norm3(x))
        else:
            out = self.feed_forward(self.norm3(x))

        x = x + out
        
        x = self.dropout(x)
       
        if self.training and self.isMoE:
            return x, gate_indices
        return x


class Gate(nn.Module):
    
    def __init__(self, args: ModelConfigs):
        super(Gate, self).__init__()

        self.args = args
        
        self.num_experts = args.num_experts
        self.exp_per_tok = args.num_experts_per_tok
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts))
        
        self.linear = nn.Linear(args.dim, self.num_experts, bias=False)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        gate_logits = self.linear(x)
        gate_weights = torch.sigmoid(gate_logits)
        gate_logits = gate_logits + self.expert_bias
        
        _, topk_indices = torch.topk(gate_logits, self.exp_per_tok, dim=-1)
        
        topk_weights = gate_weights.gather(dim=-1, index=topk_indices)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        return topk_weights, topk_indices


class MoELayer(nn.Module):
    
    def __init__(self, args: ModelConfigs):
        super(MoELayer, self).__init__()

        self.num_experts = args.num_experts
        self.experts = nn.ModuleList([
            FeedForward(args.dim, args.moe_dim_feedforward)
            for i in range(args.num_experts)
        ])

        self.dim = args.dim
        self.shared_experts = FeedForward(args.dim, args.moe_dim_feedforward * args.shared_experts)
        
        self.gate = Gate(args)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.num_experts).tolist()
        for i in range(self.num_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).view(shape), indices


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_feedforward: int):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.linear3 = nn.Linear(dim, dim_feedforward)
        
    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.linear1.weight.device)
        return self.linear2(F.silu(self.linear1(x) * self.linear3(x)))


class PositionalEncoding(nn.Module):
    
    def __init__(self, dim: int, max_len: int, dropout: float = .01):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
