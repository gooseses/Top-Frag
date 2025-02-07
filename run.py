import torch
from util.baseline import sample
from model.tmoe import TMOE    
from model.Lmser_Transformer import MFT
from util.metrics import CaculateAffinity, calcScore
import os
from util.data import getVoc, getVocFromCache
from easydict import EasyDict 
from util.tokenizer import Tokenizer    
from torch.utils.data import DataLoader
import json
from Bio import PDB
from tqdm import tqdm
import pandas as pd 


def parse_protein(pdbid):
    orign = 'test_pdbs'
    parser = PDB.PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdbid, './data/%s/%s/%s_protein.pdb'%(orign,pdbid,pdbid))
    ppb = PDB.PPBuilder()
    
    seq = ''
    for pp in ppb.build_peptides(structure):
        seq += pp.get_sequence()
    
    return seq

def run_generation(model, protein, pro_mask, vocab, device, pro_tokenizer, smi_tokenizer):
    sequences = []
    
    for i in range(10):
        sequence  = sample(model, [protein], vocab, device)
    
    sequences.append(smi_tokenizer.decode(sequence))
    
    return sequences

if __name__ == '__main__':

    with open('/teamspace/studios/this_studio/Deepest-Adjuvants/configs/model_config.json') as f: 
        configs = json.load(f)
        
    configs = EasyDict(configs)

    vocab = getVocFromCache()
    device = configs.device
    
    model = TMOE(vocab,  **configs).to(device)
    model.load_state_dict(torch.load('/teamspace/studios/this_studio/Deepest-Adjuvants/experiments/pretrain2024-09-05_05-20-11/model/3.pt'))
    
    
    # test_pdblist = sorted(os.listdir('./data/test_pdbs/'))
    # pro_files = ['./data/test_pdbs/%s/%s_protein.pdb'%(pdb,pdb) for pdb in test_pdblist]
    # ligand_files = ['./data/test_pdbs/%s/%s_ligand.sdf'%(pdb,pdb) for pdb in test_pdblist]
    # protein_seq = [parse_protein(pdb) for pdb in test_pdblist]
    
    pro_tokenizer = Tokenizer(vocab.proVoc, vocab.pro_max_len)
    smi_tokenizer = Tokenizer(vocab.smiVoc, vocab.smi_max_len)

    protein = 'MAFMKKYLLPILGLFMAYYYYSANEEFRPEMLQGKKVIVTGASKGIGREMAYHLAKMGAHVVVTARSKETLQKVVSHCLELGAASAHYIAGTMEDMTFAEQFVAQAGKLMGGLDMLILNHITNTSLNLFHDDIHHVRKSMEVNFLSYVVLTVAALPMLKQSNGSIVVVSSLAGKVAYPMVAAYSASKFALDGFFSSIRKEYSVSRVNVSITLCVLGLIDTETAMKAVSGIVHMQAAPKEECALEIIKGGALRQEEVYYDSSLWTTLLIRNPCRKILEFLYSTSYNMDRFINK'
    protein =[[char] for char in protein]
    protein_seq, _ , pro_mask = pro_tokenizer.tokenize(protein)
    scores = []
    for protein_seq, pro_mask in tqdm(zip(protein_seq, pro_mask)):
        batch_scores = []
        smiles = run_generation(model, protein_seq, pro_mask, vocab, device, pro_tokenizer, smi_tokenizer)
        for smile in smiles:
            smiscores = {}
            
            # binding = CaculateAffinity(smile, pro_file, ligand_file)
            mol = calcScore(smile) 
            
            # smiscores['smile'] = smile
            # # smiscores['binding'] = binding
            # smiscores['MolLogP'] = MolLogP
            # smiscores['qed'] = qed
            # smiscores['tpsa'] = tpsa
            # smiscores['MolWt'] = MolWt

            
            batch_scores.append(mol)
            print(smiscores)
        scores.append(batch_scores)
        
    with open('./testing_data.txt', 'w') as f:
        json.dump(scores, f)