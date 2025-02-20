import sentencepiece as spm
import atomInSmiles as ais
import os, argparse
import time
import pandas as pd
import pickle as pkl
import json
import csv
from torch.utils.data import DataLoader
from loguru import logger
import ast

class SmilesTokenizer:
   
    def __init__(self, model_prefix='tokenizer'):
        self.sp = spm.SentencePieceProcessor()
        self.model_prefix = model_prefix
        self.model_path = f"{model_prefix}.model"
     
    def tokenize(self, data):
        tokenized = self.sp.EncodeAsIds(data)
        tokenized.insert(0, self.sp.PieceToId('<s>'))
        tokenized.append(self.sp.PieceToId('</s>'))
        return tokenized
    
    def detokenize(self, tokens):
        return ''.join(tokens)
   
    def decode(self, data): 
        return self.sp.decode(data)
    
    def encode(self, data):
        try:
            x = ais.encode(data)
            print("success")
        except:
            print("fail")
            return ""
        return x
    
    def load(self, path):
        if os.path.exists(path):
            self.sp.load(path)
        else:
            raise FileNotFoundError(f"Model file not found: {path}")
        
    def getVoc(self):
        vocab = [self.sp.IdToPiece(i) for i in range(self.sp.GetPieceSize())]
        return vocab
 
def createSplit(filename, seed=42):
    start_time = time.time()  
    
    data = pd.read_csv("data/" + filename + '.tsv', sep="\t")
    
    train = data.sample(frac=0.8, random_state=seed)
    remaining = data.drop(train.index)
    val = remaining.sample(frac=0.5, random_state=seed)
    test = remaining.drop(val.index)
    
    dictionary = {
        "Train": train.index.to_list(),
        "Validate": val.index.to_list(),
        "Test": test.index.to_list()
    }
    
    with open("data/Train_Val_Test_Splits.pkl", "wb") as outfile:
        pkl.dump(dictionary, outfile)
    
    elapsed_time = round(time.time() - start_time, 3)
    
    print('Data successfully parsed in ' + str(elapsed_time) + 
          "\n   Training Size: " + str(len(train)) +
          "\n   Validation Size: " + str(len(val)) +
          "\n   Test Size: " + str(len(test)))

def get_dataloaders(config):
    logger.info('preparing training data')
    with open('data/train-val-split.json', 'r') as f:
        splits = json.load(f)

    train_slice = splits["train"]
    validate_slice = splits["valid"]
    
    data = pd.read_csv('data/tokenized_data.csv', sep = '\t')
    
    train_data = data.loc[train_slice]
    validate_data = data.loc[validate_slice]
    
    logger.info('preparing dataloaders')
   
    train_data = DataLoader(train_data, shuffle=True, batch_size=config.batchsize, drop_last=False)
    validate_data = DataLoader(validate_data, shuffle=False, batch_size=config.batchsize, drop_last=False)

    return train_data, validate_data

def encode_data(tokenizer, input_file, encoded_output_file, pro_voc_output):
    data = pd.read_csv(input_file, sep='\t')

    # Extract protein and SMILES data
    pro_data = data['protein']
    smi_data = data['smiles']
    
    pro_voc, _= get_pro_voc(pro_data)

    with open(pro_voc_output, "w")as f:
        json.dump(pro_voc, f) 
    
    # Encode protein data (assuming the tokenizer processes this as a list of tokens)
    encoded_pro_data = pro_data.apply(list).tolist()

    # Encode SMILES data using tokenizer
    encoded_smi_data = smi_data.apply(tokenizer.encode).tolist()

    # Write the encoded protein and SMILES data into the TSV file
    with open(encoded_output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')  # Use tab as delimiter for TSV
        writer.writerow(["protein", "smiles"])
        for pro, smi in zip(encoded_pro_data, encoded_smi_data):
            # Write each row of the encoded data as lists of tokens
            writer.writerow([pro, smi])

    print(f"Encoded data saved to {encoded_output_file}")  
    
def get_pro_voc(data):
    proMaxLen = max(list(data.apply(len))) + 2

    pros_split = data.apply(list)
    proVoc = sorted(list(['<s>', '</s>', '<pad>'] + set([i for j in pros_split for i in j]))) 

    return proVoc, proMaxLen

def tokenize_protein(pro):
    pro = ast.literal_eval(pro)
    with open('data/proVoc.json', 'r') as f:
        vocab = json.load(f)
    indices = []
    indices.append(vocab.index(tok) for tok in pro)
    
    indices.insert(0, vocab.index('<s>'))
    indices.append(vocab.index('</s>'))
    
    return indices
            

def tokenize_data(tokenizer, input_file, output_file):
    
    data = pd.read_csv(input_file, sep='\t')
    # Remove rows where 'smiles' is empty or NaN
    data = data[data['smiles'].notna() & (data['smiles'] != '')]

    pro_data = data['protein']
    smi_data = data['smiles']

    tokenized_pro_data = pro_data.apply(tokenize_protein).tolist()
    
            
    tokenized_smi_data = smi_data.apply(tokenizer.tokenize).tolist()
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["protein", "smiles"])
        writer.writerows(zip(tokenized_pro_data, tokenized_smi_data))

    print(f"Tokenized data saved to {output_file}")
    
if __name__ == '__main__':
    tokenizer = SmilesTokenizer()
    encoded_file = "data/encoded_data.csv"
    sp_training_file = "data/sp_training_data.txt"
    tokenized_file = "data/tokenized_data.csv"
    
    # encoded_training_file = encode_data(tokenizer, "data/train-val-data.tsv", encoded_file, sp_training_file, "data/proVoc.json")
    # 
    
    tokenizer.load("AIS_src_sp.model")
    
    # tokenize_data(tokenizer, encoded_file, tokenized_file)
    print(tokenizer.getVoc())
