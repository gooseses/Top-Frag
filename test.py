from model.MolSeek import TopFrag
from data.voc import Voc
from configs.ModelConfigs import ModelConfigs
from configs.TrainConfigs import TrainConfigs
from data.data import get_dataloaders


if __name__ == '__main__':
    
    voc = Voc()
    configs = ModelConfigs(voc)
    model = TopFrag(configs)

    train_settings = TrainConfigs()
    
    train_data, valid_data, voc = get_dataloaders(train_settings)
    
    for pro, smi in train_data:
        print(pro)
        print(smi)
        break
    
    print(voc.pro_max_len)
    print(voc.pro_pad_idx)
    print(voc.pro_voc_len)
    print(voc.smi_max_len)
    print(voc.smi_pad_idx)
    print(voc.smi_voc_len)
    
    