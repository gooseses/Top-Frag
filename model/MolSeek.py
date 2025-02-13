import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from typing import Optional

from configs.ModelConfigs import ModelConfigs


class Transformer(nn.Module):
    
    def __init__(self, args: ModelConfigs):
        super(Transformer, self).__init__()
        
        self.args = args
        
        self.pro_emb = nn.Embedding(args.pro_voc_len, args.dim, args.pad_idx)
        self.smi_emb = nn.Embedding(args.smi_voc_len, args.dim, args.pad_idx)
       
        self.pro_pe = PositionalEncoding(args.dim, args.max_len, args.dropout)
        self.smi_pe = PositionalEncoding(args.dim, args.max_len, args.dropout)
        self.layers = torch.nn.ModuleList()
        
        for layerid in range(args.num_layers):
            self.layers.append(Block(args, layerid))
        
        self.d_norm = nn.RMSNorm(args.dim)
        
        self.linear = nn.Linear(args.dim, self.smi_voc_len)

        self._reset_parameters()

        self.to(args.device)

    def _reset_parameters(self):
        # Initialize model parameters using Xavier uniform distribution.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_padding_mask(self, seq: torch.Tensor, pad_idx: int):
        return seq.ne(pad_idx)
    
    def forward(self, src, tgt):

        src = pad_sequence(src, batch_first=True, padding_value=self.args.pad_idx)
        tgt = pad_sequence(tgt, batch_first=True, padding_value=self.args.pad_idx)

        src_mask = self.make_padding_mask(src, self.args.pad_idx)
        tgt_mask = self.make_padding_mask(tgt, self.args.pad_idx)

        _, seq_len = tgt.shape

        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

        all_gate_outs = []    
        all_gate_indices = []
        
        src = self.pro_emb(src)
        src = self.pro_pe(src)
            
        tgt = self.smi_emb(tgt)
        tgt = self.smi_pe(tgt)

        e_out = src
        d_out = tgt
        
        for mod in self.layers:
            if(self.training):
                e_out, d_out, gate_outs, gate_indices = mod(e_out, d_out, src_mask, tgt_mask, causal_mask)
                
                all_gate_outs.append(gate_outs)
                all_gate_indices.append(gate_indices)
            else:
                e_out, d_out = mod(e_out, d_out, src_mask, tgt_mask, causal_mask)
            
            # Normalize the encoder/decoder outputs
        d_out = self.d_norm(d_out)
            
        out = self.linear(d_out)
        
        out = F.log_softmax(out, dim=-1)
        
        
        if self.training:
            return out, all_gate_outs, all_gate_indices

        return out
            
        
        
class Block(nn.Module):
    def __init__(self, args: ModelConfigs, id: int):
        super(Block, self).__init__()
    
        # Connected encoder and decoder layers for better feature transfer.
        self.encoder = EncoderBlock(args, id)
        self.decoder = DecoderBlock(args, id)

        self.isMoE = id > args.num_dense_layers

    def forward(self, src, tgt, src_mask, tgt_mask, casual_mask):
        # Forward pass through encoder/decoder layers.

       
        if self.training and self.isMoE:
            all_gate_outs = []    
            all_gate_indices = []

            out1, gate_out1, gate_indices1 = self.encoder(src, src_mask)
            out2, gate_out2, gate_indices2 = self.decoder(out1, tgt, src_mask, tgt_mask, casual_mask)
            
            all_gate_outs.append(gate_out1)
            all_gate_indices.append(gate_indices1)
            all_gate_outs.append(gate_out2)
            all_gate_indices.append(gate_indices2)
        else:
            out1 = self.encoder(src)
            out2 = self.decoder(out1, tgt)

        if self.training:
            return out1, out2, all_gate_outs, all_gate_indices

        return out1, out2

        
class EncoderBlock(nn.Module):
    
    def __init__(self, id: int , args: ModelConfigs):
        super(EncoderBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(args.dim, args.nheads, dropout = args.dropout)
        self.feed_forward =  FeedForward(args.dim, args.dim_feedforward, args) if id < args.num_dense_layers else MoELayer(args)
        self.norm1 = nn.RMSNorm(args.dim) 
        self.norm2 = nn.RMSNorm(args.dim)
        self.isMoE = id > args.num_dense_layers
        
    def forward(self, x, src_mask):
        
        x = self.norm1(x)
        x = x + self.attention(x,x,x, key_padding_mask = src_mask)[0]

        if self.isMoE:
            out, gate_out, gate_indices = self.feed_forward(self.norm2(x))
        else:
            out = self.feed_forward(self.norm2(x))

        x = x + out
        
        if self.training and self.isMoE:
            return x, gate_out, gate_indices
        return x
 
class DecoderBlock(nn.Module):
    
    def __init__(self, id: int ,args: ModelConfigs):
        super(DecoderBlock, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(args.dim, args.nheads, dropout = args.dropout)
        self.multi_attention = nn.MultiheadAttention(args.dim, args.nheads, dropout = args.dropout)
        self.feed_forward =  FeedForward(args.dim, args.dim_feedforward, args) if id < args.num_dense_layers else MoELayer(args)
        self.norm1 = nn.RMSNorm(args.dim) 
        self.norm2 = nn.RMSNorm(args.dim)
        self.norm3 = nn.RMSNorm(args.dim)
        
        self.isMoE = id > args.num_dense_layers
        
    def forward(self, x, mem, src_mask, tgt_mask, causal_mask):
        
        x = self.norm1(x)
        x = x + self.self_attention(x,x,x, key_padding_mask = tgt_mask, attn_mask = causal_mask)[0]
        x = self.norm2(x)
        x = x + self.multi_attention(x, mem, mem, key_padding_mask = src_mask)[0]

        if self.isMoE:
            out, gate_out, gate_indices = self.feed_forward(self.norm3(x))
        else:
            out = self.feed_forward(self.norm3(x))

        x = x + out
       
        if self.training:
            return x, gate_out, gate_indices
        return x
    
class Gate(nn.Module):
    
    def __init__(self, args: ModelConfigs):
        super(Gate, self).__init__()

        self.args = args
        
        self.num_experts = args.num_experts
        self.exp_per_tok = args.num_experts_per_tok
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts))
        
        self.linear = nn.Linear(args.dim, self.num_experts, bias=False)
        
    def forward(self, x):
        
        gate_logits = self.linear(x)
        gate_weights = torch.sigmoid(gate_logits)
        gate_logits = gate_logits + self.expert_bias
        
        _, topk_indices = torch.topk(gate_logits, self.exp_per_tok, dim = -1)
        
        topk_weights = gate_weights.gather(dim = -1, index = topk_indices)
        topk_weights = topk_weights / topk_weights.sum(dim = -1, keepdim = True)
        
        return topk_weights, topk_indices
        
       
"MoE Layer"
class MoELayer(nn.Module):
    
    def __init__(self, args: ModelConfigs):
        super(MoELayer, self).__init__()

        self.num_experts = args.num_experts
        # Initialize a list of FNN experts.
        self.experts = nn.ModuleList([
            FeedForward(args.dim, args.moe_dim_feedforward)
            for i in range(args.num_experts)
        ])
        
        self.dim = args.dim
        self.shared_experts = FeedForward(args.dim, args.moe_feedforward*args.shared_experts)
        
        self.gate = Gate(args)
        
    def forward(self, x):
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
        return (y + z).view(shape), weights, indices
    
"FNN Layer (Expert)"
class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_feedforward: int):
        super(FeedForward, self).__init__()
        
        # Linear layers and feed-forward processing.
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.linear3 = nn.Linear(dim, dim_feedforward)
        
    def forward(self, x):
        # Ensure the input tensor is on the same device.
        x = x.to(self.linear1.weight.device)
        # Apply linear layers with SiLU activation.
        return self.linear2(F.silu(self.linear1(x) * self.linear3(x)))
        

"Positional Encoding"
class PositionalEncoding(nn.Module):
    
    def __init__(self, dim: int, max_len: int, dropout: float = .01):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p = dropout)
        
        pe = torch.zeros(max_len, dim)
        
        # Compute postiional encodings based on sin/cos functions.
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim,2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

    