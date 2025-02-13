import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from configs.ModelConfigs import ModelConfigs

class Transformer(nn.Module):
    
    def __init__(self, args: ModelConfigs):
        super(Transformer, self).__init__()
        
        self.args = args
        
        self.pro_emb = nn.Embedding(args.pro_voc_len, args.d_model, args.pro_pad_idx).to(args.device)
        self.smi_emb = nn.Embedding(args.smi_voc_len, args.d_model, args.smi_pad_idx).to(args.device)
       
        self.pro_pe = PositionalEncoding(args.d_model, args.max_len, args.dropout)
        self.smi_pe = PositionalEncoding(args.d_model, args.max_len, args.dropout)
        self.layers = torch.nn.ModuleList()
        
        for layerid in range(args.num_layers):
            self.layers.append(Block(args, layerid))
        
        self.e_norm = nn.RMSNorm(args.d_model)
        self.d_norm = nn.RMSNorm(args.ad_model)
        
        self.linear = nn.Linear(args.d_model, self.smi_voc_len)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize model parameters using Xavier uniform distribution.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    
    def forward(self, x, tgt):
        
        all_gate_outs = []    
        all_gate_indices = []
        
        x = self.pro_emb(x)
        x = self.pro_pe(x)
            
        tgt = self.smi_emb(tgt)
        tgt = self.smi_pe(tgt)
        
        for mod in self.layers:
            e_out, d_out, gate_outs, gate_indices = mod(e_out, d_out)
            
            all_gate_outs.append(gate_outs)
            all_gate_indices.append(gate_indices)
            
            # Normalize the encoder/decoder outputs
            e_out = self.e_norm(e_out)
            d_out = self.d_norm(d_out)
            
        out = self.linear(d_out)
        out = F.log_softmax(out, dim=-1)
        
        return out, all_gate_outs, all_gate_indices
            
        
        
class Block(nn.Module):
    def __init__(self, args: ModelConfigs, id: int):
        super(Block, self).__init__()
    
        # Connected encoder and decoder layers for better feature transfer.
        self.encoder = EncoderBlock(args, id)
        self.decoder = DecoderBlock(args, id)

    def forward(self, src, tgt):
        # Forward pass through encoder/decoder layers.
        all_gate_outs = []    
        all_gate_indices = []
        
        out1, gate_out1, gate_indices1 = self.encoder(src, tgt)
        out2, gate_out2, gate_indices2 = self.decoder(out1, tgt)
        
        all_gate_outs.append(gate_out1)
        all_gate_indices.append(gate_indices1)
        all_gate_outs.append(gate_out2)
        all_gate_indices.append(gate_indices2)
        
        return out1, out2, all_gate_outs, all_gate_indices

        
class EncoderBlock(nn.Module):
    
    def __init__(self, id: int ,args: ModelConfigs):
        super(EncoderBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(args.d_model, args.nheads, dropout = args.dropout)
        self.feed_forward =  FeedForward(args.d_model, args.dim_feedforward, args) if id < args.num_moe_layers else MoELayer(args)
        self.norm1 = nn.RMSNorm(args.d_model) 
        self.norm2 = nn.RMSNorm(args.d_model)
        
        
    def forward(self, x):
        
        x = self.norm1(x)
        x = x + self.attention(x,x,x, is_causal=True)[0]
        out, gate_out, gate_indices = self.feed_forward(self.norm2(x))
        x = x + out
        
        return x, gate_out, gate_indices

class DecoderBlock(nn.Module):
    
    def __init__(self, id: int ,args: ModelConfigs):
        super(DecoderBlock, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(args.d_model, args.nheads, dropout = args.dropout)
        self.multi_attention = nn.MultiheadAttention(args.d_model, args.nheads, dropout = args.dropout)
        self.feed_forward =  FeedForward(args.d_model, args.dim_feedforward, args) if id < args.num_moe_layers else MoELayer(args)
        self.norm1 = nn.RMSNorm(args.d_model) 
        self.norm2 = nn.RMSNorm(args.d_model)
        self.norm3 = nn.RMSNorm(args.d_model)
        
        
    def forward(self, x):
        
        x = self.norm1(x)
        x = x + self.self_attention(x,x,x, is_causal=True)[0]
        out, gate_out, gate_indices = self.feed_forward(self.norm2(x))
        x = x + out
        
       
        return x, gate_out, gate_indices
    
class Gate(nn.Module):
    
    def __init__(self, args: ModelConfigs):
        self.args = args
        
        self.dim = args.dim
        self.num_experts = args.num_experts
        self.exp_per_tok = args.num_experts_per_tok
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts))
        
        self.linear = nn.Linear(self.dim, self.num_experts, bias=False)
        
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

        # Initialize a list of FNN experts.
        self.experts = nn.ModuleList([
            FeedForward(args.d_model, args.moe_dim_feedforward).to(args.device)
            for i in range(args.num_experts)
        ])
        
        self.shared_experts = FeedForward(args.d_model, args.moe_feedforward*args.shared_experts).to(args.device)
        
        self.gate = Gate(args)
        
    def forward(self, x):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).view(shape), weights, indices
    
"FNN Layer (Expert)"
class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int):
        super(FeedForward, self).__init__()
        
        # Linear layers and feed-forward processing.
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        
    def forward(self, x):
        device = x.device
        # Ensure the input tensor is on the same device.
        x = x.to(self.linear1.weight.device)
        # Apply linear layers with SiLU activation.
        return self.linear2(F.silu(self.linear1(x) * self.linear3(x))).to(device)
        

"Positional Encoding"
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int, dropout: float = .01):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p = dropout)
        
        pe = torch.zeros(max_len, d_model)
        
        # Compute postiional encodings based on sin/cos functions.
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model,2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)
        