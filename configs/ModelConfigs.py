from data.voc import Voc    

class ModelConfigs:
    
    def __init__(self, voc: Voc):
        
        self.model_name = "TMOE"
        self.device = "cuda"
        self.num_experts = 8
        self.num_dense_layers = 1
        self.num_layers = 6
        self.shared_experts = 2
        self.num_experts_per_tok = 2
        self.dropout = 0.1
        self.dim = 768
        self.nheads = 8
        self.dim_feedforward = 3072
        self.moe_dim_feedforward = 1536
        self.voc = voc