import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


"Model"
class TMOE(nn.Module):
    
    def __init__(self, voc, d_model=768, nheads=8, num_layers=8, 
                 dim_feedforward = 3072, **kwargs):
        
        # Initialize TMOE model.
        super(TMOE, self).__init__()

        self.proVoc = voc.proVoc  # Protein vocabulary.
        self.smiVoc = voc.smiVoc  # SMILES vocabulary.

        self.pro_voc_len = voc.pro_voc_len  # Length of protein vocabulary.
        self.smi_voc_len = voc.smi_voc_len  # Length of SMILES vocabulary.
        self.pro_pad_idx = voc.pro_pad_idx  # Padding index for protein.
        self.smi_pad_idx = voc.smi_pad_idx  # Padding index for SMILES.
        self.pro_max_len = voc.pro_max_len  # Maximum length for protein sequences.
        self.smi_max_len = voc.smi_max_len  # Maximum length for SMILES sequences.
        
        # Model parameters.
        self.d_model = d_model
        self.proEmbedding = nn.Embedding(self.pro_voc_len, d_model, self.pro_pad_idx).to(kwargs["device"])
        self.smiEmbedding = nn.Embedding(self.smi_voc_len, d_model, self.smi_pad_idx).to(kwargs["device"])
       
        self.proPositionalEncoding = PositionalEncoding(d_model, self.pro_max_len, kwargs['dropout'])
        self.smiPositionalEncoding = PositionalEncoding(d_model, self.smi_max_len, kwargs['dropout'])
        
        self.layers = torch.nn.ModuleList()
        for layerid in range(num_layers):
            self.layers.append(EncoderDecoderLayer(layerid, d_model, nheads, dim_feedforward, kwargs))
        
        self.e_norm = nn.LayerNorm(d_model)
        self.d_norm = nn.LayerNorm(d_model)
        
        self.linear = nn.Linear(d_model, self.smi_voc_len)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize model parameters using Xavier uniform distribution.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
       
    def forward(self, src, tgt, smiMask, proMask, tgt_mask):
        tgt_mask = tgt_mask.squeeze(0)

        # Protein embedding & encoding.
        src = self.proEmbedding(src)
        src = self.proPositionalEncoding(src)
        src = src.permute(1, 0, 2)

        # SMILES embedding & encoding.
        tgt = self.smiEmbedding(tgt)
        tgt = self.smiPositionalEncoding(tgt)
        tgt = tgt.permute(1, 0, 2)

        # Masks.
        src_key_padding_mask = ~(proMask.to(torch.bool))
        tgt_key_padding_mask = ~(smiMask.to(torch.bool))
        memory_key_padding_mask = ~(proMask.to(torch.bool))
        
        # Encoder-decoder pass through.
        e_out = src
        d_out = tgt
        for mod in self.layers:
            e_out, d_out = mod(e_out, d_out, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,\
                tgt_key_padding_mask= tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

            # Normalize the encoder/decoder outputs
            e_out = self.e_norm(e_out)
            d_out = self.d_norm(d_out)
            
        # Linear transformation & softmax for decoder output.
        out = d_out.permute(1, 0, 2)
        out= self.linear(out)
        out1 = F.log_softmax(out, dim=-1)
        return out1


"Encoder Decoder Layer"
class EncoderDecoderLayer(nn.Module):
    def __init__(self,layerid: int, d_model: int, nheads: int, dim_feedforward: int, kwargs):
        super(EncoderDecoderLayer, self).__init__()
    
        # Connected encoder and decoder layers for better feature transfer.
        self.encoder = EncoderLayer(layerid, d_model, nheads, dim_feedforward, kwargs)
        self.decoder = DecoderLayer(layerid, d_model, nheads, dim_feedforward, kwargs)

    def forward(self, src, tgt, tgt_mask = None, src_key_padding_mask = None, tgt_key_padding_mask=None, memory_key_padding_mask= None):
        # Forward pass through encoder/decoder layers.
        out1 = self.encoder(src, src_key_padding_mask = src_key_padding_mask)
        out2 = self.decoder(tgt, out1, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask)
        
        return out1, out2


"Decoder"
class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, d_model: int, nheads: int, dim_feedforward:int, kwargs):
        super(DecoderLayer, self).__init__()
        
        # Decoder layer parameters.
        self.layer_id = layer_id
        self.nheads = nheads
        self.dim = d_model
        self.head_dim = d_model // nheads

        # Self-attention and multihead attention layers.
        self.self_attention = nn.MultiheadAttention(d_model, nheads, dropout = kwargs['dropout'])
        self.multihead_attention = nn.MultiheadAttention(d_model, nheads, dropout = kwargs['dropout'])
        
        # Feed-forward network with mixture-of-experts.
        self.feed_forward = MoELayer(self.dim, dim_feedforward, kwargs["num_experts"], kwargs)
        
        # Layer normalization for attention and feed-forward layers.
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.self_attention_norm = nn.LayerNorm(d_model)
        self.multihead_attention_norm = nn.LayerNorm(d_model)
        
        # Dropout layers.
        self.dropout1 = nn.Dropout(kwargs['dropout'])
        self.dropout2 = nn.Dropout(kwargs['dropout'])
        self.dropout3 = nn.Dropout(kwargs['dropout'])
        
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        ) -> torch.Tensor:
        
        "Modified Forward Pass from nn.TransformerDecoderLayer"

        # Self-attention, multihead attention, and feed-forward network with normalization.
        x = tgt
        x = self.self_attention_norm(x+ self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
        x = self.multihead_attention_norm(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
        x = self.feed_forward_norm(x + self.dropout3(self.feed_forward.forward(x)))
        
        return x


    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], is_causal: bool = False) -> torch.Tensor:
        # Self-attention with dropout.
        x = self.self_attention(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)


    def _mha_block(self, x: torch.Tensor, mem: torch.Tensor,
                   attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], is_causal: bool = False) -> torch.Tensor:
        # Multihead-attention with dropout.
        x = self.multihead_attention(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)


"Encoder"
class EncoderLayer(nn.Module):
    def __init__(self, layer_id: int, d_model: int, nheads: int, dim_feedforward:int, kwargs):
        super(EncoderLayer, self).__init__()
        
        # Encoder layer parameters.
        self.layer_id = layer_id
        self.nheads = nheads
        self.dim = d_model
        self.head_dim = d_model // nheads

        # Self-attention and FNN layers.
        self.self_attention = nn.MultiheadAttention(d_model, nheads, dropout = kwargs["dropout"])
        self.feed_forward = MoELayer(self.dim, dim_feedforward, kwargs["num_experts"], kwargs)

        # Layer normalization for attention and FNN.
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.self_attention_norm = nn.LayerNorm(d_model)

        # Dropout layers.
        self.dropout1 = nn.Dropout(kwargs['dropout'])
        self.dropout2 = nn.Dropout(kwargs['dropout'])
        
        
    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False) -> torch.Tensor:
        
        "Modified Forward Pass from nn.TransformerEncoderLayer"

        # Standardize the key padding mask for compatibilitu
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        # Standardize the source mask.
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        
        # Input tensor (src).
        x = src
        
        # Apply self-attention with layer normalization.
        x = self.self_attention_norm(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
        
        # Apply FFN with layer normalizaiton.
        x = self.feed_forward_norm(x + self.dropout2(self.feed_forward.forward(x)))
        
        return x

    
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], is_causal: bool = False) -> torch.Tensor:
        # Apply multi-head self-attention.
        x = self.self_attention(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)
        
        
"MOE Layer"
class MoELayer(nn.Module):
    def __init__(self, d_model: int, dim_feedforward:int, num_experts:int, kwargs):
        super(MoELayer, self).__init__()

        # Initialize a list of FNN experts.
        self.experts = nn.ModuleList([
            FeedForward(d_model, dim_feedforward).to(kwargs["device"])
            for i in range(num_experts)
        ])
        
        # Initalize a shared expert FFN.
        self.sharedExperts = FeedForward(d_model, dim_feedforward*kwargs["shared_experts"]).to(kwargs["device"])
        
        # Linear layer to compute gating probabilities of each expert.
        self.gate = nn.Linear(
            d_model, num_experts, bias=False)
        
        # Number of experts selected per token.
        self.num_experts_per_token = kwargs["num_experts_per_tok"]
        
    def forward(self, x):
        identity = x
        shape = x.shape

        # Flatten the input for expert selection.
        x = x.view(-1, x.shape[-1])

        # Compute gating scores and select top-k experts
        scores = self.gate(x).softmax(dim=-1)
        expert_weights, expert_indices = torch.topk(
            scores, self.num_experts_per_token, dim = -1
        )

        # Apply softmax to selected expert weights.
        expert_weights = expert_weights.softmax(dim = -1)
        
        # Flatten expert indicies.
        flat_expert_indices = expert_indices.view(-1)
        
        print(flat_expert_indices)
        
        # Repeat input tensor for the number of tokens.
        x = x.repeat_interleave(self.num_experts_per_token, dim = 0)
        
        # Empty tensor for storing expert outputs.
        y = torch.empty_like(x)

        # Route input to selected experts.
        for i, expert in enumerate(self.experts):
            y[flat_expert_indices == i] = expert(x[flat_expert_indices == i])
       
        # Combine expert outputs and apply weighted sum based on gating.
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1).view(*shape)
       
        # Add the output of shared experts and reshape to orginal input.
        y = y + self.sharedExperts(identity)

        return y.view(*shape)


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
    
    nn.Transformer