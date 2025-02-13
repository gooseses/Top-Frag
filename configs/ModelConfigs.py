

class ModelConfigs:
    
    def __init__(self):
        
        self.model_name = "TMOE"
        self.device = "cuda"
        self.num_experts = 8
        self.shared_experts = 2
        self.num_experts_per_tok = 2
        self.dropout = 0.1
        self.smi_max_len = 128
        self.pro_max_len = 128
        self.batch_size = 32
        self.lr = 1e-4
        self.epochs = 10
        self.dim = 768
        self.nheads = 8
        self.dim_feedforward = 3072
        self.moe_feedforward = 3072/2
    