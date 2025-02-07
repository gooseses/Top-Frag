import torch
import time
import os
import shutil
import numpy as np
import random as rd
import argparse
import json
from loguru import logger
from rdkit import Chem
from Bio import PDB
from easydict import EasyDict
from util.data import getVocFromCache
from model.tmoe import TMOE
import torch.nn as nn
from util.docking import CaculateAffinity

@torch.no_grad()
def sample(model, path, vocabulary, proVoc, smiMaxLen, proMaxLen, device, sampleTimes, protein_seq):
    model.eval()

    pathList = path[:]
    length = len(pathList)
    pathList.extend(['^'] * (smiMaxLen - length))

    protein = '&' + protein_seq +'$'
    proList = list(protein)
    lp = len(protein)

    proList.extend(['^'] * (proMaxLen - lp))
    
    proteinInput = [proVoc.index(pro) for pro in proList]
    currentInput = [vocabulary.index(smi) for smi in pathList]
    
    src = torch.as_tensor([proteinInput]).to(device)
    tgt = torch.as_tensor([currentInput]).to(device)

    smiMask = [1] * length + [0] * (smiMaxLen - length)
    smiMask = torch.as_tensor([smiMask]).to(device)
    proMask = [1] * lp + [0] * (proMaxLen - lp)
    proMask = torch.as_tensor([proMask]).to(device)

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(smiMaxLen).tolist()
    tgt_mask = [tgt_mask] * 1
    tgt_mask = torch.as_tensor(tgt_mask).to(device)

    sl = length - 1
    out = model(src, tgt, smiMask, proMask, tgt_mask)[:, sl, :]
    out = out.tolist()[0]
    pr = np.exp(out) / np.sum(np.exp(out))
    prList = np.random.multinomial(1, pr, sampleTimes)
    
    indices = list(set(np.argmax(prList, axis=1)))
    
    atomList = [vocabulary[i] for i in indices]
    logpList = [np.log(pr[i] + 1e-10) for i in indices]
    
    atomListExpanded = []
    logpListExpanded = []
    for idx, atom in enumerate(atomList):
        if atom == '&' or atom == '^':
            continue
        atomListExpanded.append(atom)
        logpListExpanded.append(logpList[idx])
    # logger.info(atomListExpanded)
    return atomListExpanded, logpListExpanded


def parse_protein(pdbid):
    orign = 'test_pdbs'
    parser = PDB.PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdbid, 'data/%s/%s/%s_protein.pdb'%(orign,pdbid,pdbid))
    ppb = PDB.PPBuilder()
    
    seq = ''
    for pp in ppb.build_peptides(structure):
        seq += pp.get_sequence()
    
    return seq



QE = 9
QMIN = QE
QMAX = QE
groundIndex = 0 # MCTS Node唯一计数
infoma = {}

class Node:

    def __init__(self, parentNode=None, childNodes=[], path=[],p=1.0, smiMaxLen=999):
        global groundIndex
        self.index = groundIndex
        groundIndex += 1

        self.parentNode = parentNode
        self.childNodes = childNodes
        self.wins = 0
        self.visits = 0
        self.path = path  #MCTS 路径
        self.p = p
        self.smiMaxLen = smiMaxLen

    def SelectNode(self):
        nodeStatus = self.checkExpand()
        if nodeStatus == 4:
            puct = []
            for childNode in self.childNodes:
                puct.append(childNode.CaculatePUCT())
            
            m = np.max(puct)
            indices = np.nonzero(puct == m)[0]
            ind=rd.choice(indices)
            return self.childNodes[ind], self.childNodes[ind].checkExpand()
        
        return self, nodeStatus

    def AddNode(self, content, p):
        n = Node(self, [], self.path + [content], p=p, smiMaxLen=self.smiMaxLen)
        self.childNodes.append(n)
        return n
    
    def UpdateNode(self, wins):
        self.visits += 1
        self.wins += wins
        
    def CaculatePUCT(self):
        if not self.parentNode:
            return 0.0 # 画图用的
        c = 1.5
        if QMAX == QMIN:
            wins = 0
        else:
            if self.visits:
                wins = (self.wins/self.visits - QMIN) / (QMAX - QMIN)
            else: 
                wins = (QE - QMIN) / (QMAX - QMIN)
        
        return wins + c*self.p*np.sqrt(self.parentNode.visits)/(1+self.visits)
        # return wins/self.visits+50*self.p*np.sqrt(self.parentNode.visits)/(1+self.visits)
    
    def checkExpand(self):
        """
            node status: 1 terminal; 2 too long; 3 legal leaf node; 4 legal noleaf node
        """

        if self.path[-1] == '$':
            return 1
        elif not (len(self.path) < self.smiMaxLen):
            return 2
        elif len(self.childNodes) == 0:
            return 3
        return 4
        
def JudgePath(path, smiMaxLen):
    return (path[-1] != '$') and (len(path) < smiMaxLen)

def Select(rootNode):
    while True:
        rootNode, nodeStatus = rootNode.SelectNode()
        if nodeStatus != 4:
            return rootNode, nodeStatus
  
def Expand(rootNode, atomList, plist):
    if JudgePath(rootNode.path, rootNode.smiMaxLen):
        for i, atom in enumerate(atomList):
            rootNode.AddNode(atom, plist[i])

def Update(node, wins):
    while node:
        node.UpdateNode(wins)
        node = node.parentNode

def updateMinMax(node):
    # muzero method
    global QMIN
    global QMAX
    if node.visits:
        QMAX = max(QMAX, node.wins/node.visits)
        QMIN = min(QMIN, node.wins/node.visits)
        for child in node.childNodes:
            updateMinMax(child)

def rollout(node, model):
    path = node.path[:]
    smiMaxLen = node.smiMaxLen
    
    allScore = []
    allValidSmiles = []
    allSmiles = []

    while JudgePath(path, smiMaxLen):
        # 快速走子
        atomListExpanded, pListExpanded = sample(model, path, smiVoc, proVoc, smiMaxLen, proMaxLen, device, 30, protein_seq)
        
        m = np.max(pListExpanded)
        indices = np.nonzero(pListExpanded == m)[0]
        ind=rd.choice(indices)
        path.append(atomListExpanded[ind])
    
    if path[-1] == '$':
        smileK = ''.join(path[1:-1])
        allSmiles.append(smileK)
        try:
            mols = Chem.MolFromSmiles(smileK)
        except:
            pass
        if mols and len(smileK) < smiMaxLen:
            global infoma
            if smileK in infoma:
                affinity = infoma[smileK]
            else:
                affinity = CaculateAffinity(smileK, file_protein=pro_file[ik], file_lig_ref=ligand_file[ik], out_path=resFolderPath)
                infoma[smileK] = affinity
            
            if affinity == 500:
                Update(node, QMIN)
            else:
                logger.success(smileK + '       ' + str(-affinity))
                Update(node, -affinity)
                allScore.append(-affinity)
                allValidSmiles.append(smileK)
        else:
            logger.error('invalid: %s'%(''.join(path)))
            Update(node, QMIN)
    else:
        logger.warning('abnormal ending: %s'%(''.join(path)))
        Update(node, QMIN)

    return allScore, allValidSmiles, allSmiles
    
def MCTS(rootNode):
    allScore = []
    allValidSmiles = []
    allSmiles = []
    currSimulationTimes = 0
    
    while currSimulationTimes < simulation_times:
        
        global QMIN
        global QMAX
        QMIN = QE
        QMAX = QE
        updateMinMax(rootNode)
        currSimulationTimes += 1
        
        #MCTS SELECT
        node, _ = Select(rootNode)
        # VisualizeInterMCTS(rootNode, modelName, './', times, QMAX, QMIN, QE)

        #rollout
        score, validSmiles, aSmiles = rollout(node, model)
        allScore.extend(score)
        allValidSmiles.extend(validSmiles)
        allSmiles.extend(aSmiles)

        #MCTS EXPAND 
        atomList, logpListExpanded = sample(model, node.path, smiVoc, proVoc, smiMaxLen, proMaxLen, device, 30, protein_seq)
        pListExpanded = [np.exp(p) for p in logpListExpanded]
        Expand(node, atomList, pListExpanded)

        
    if args.max:
        indices = np.argmax([n.visits for n in rootNode.childNodes])
    else:
        allvisit = np.sum([n.visits for n in rootNode.childNodes]) * 1.0
        prList = np.random.multinomial(1, [(n.visits)/allvisit for n in rootNode.childNodes], 1)
        indices = list(set(np.argmax(prList, axis=1)))[0]
        logger.info([(n.visits)/allvisit for n in rootNode.childNodes])

    return rootNode.childNodes[indices], allScore, allValidSmiles, allSmiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=50, help='protein index')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('-st', type=int, default=10, help='simulation times')
    parser.add_argument('--source', type=str, default='new')
    parser.add_argument('-p', type=str, default='LT', help='pretrained model')

    parser.add_argument('--max', action="store_true", help='max mode')

    args = parser.parse_args()
    for ik in range(6,100):
            
        if args.source == 'new':
            test_pdblist = sorted(os.listdir('data/test_pdbs'))
            pro_file = ['data/test_pdbs/%s/%s_protein.pdb'%(pdb,pdb) for pdb in test_pdblist]
            ligand_file = ['data/test_pdbs/%s/%s_ligand.sdf'%(pdb,pdb) for pdb in test_pdblist]
            protein_seq = parse_protein(test_pdblist[ik])
        
        
        else:
            raise NotImplementedError('Unknown source: %s' % args.source)


        simulation_times = args.st
        experimentId = os.path.join('experiments/', args.p)
        ST = time.time()

        modelName = 'simga_drug'
        hpc_device = "gpu"
        mode = "max" if args.max else "freq"
        resFolder = '%s_%s_mcts_%s_%s_%s_%s'%(hpc_device,mode,simulation_times, modelName, ik, test_pdblist[ik])

        resFolderPath = 'experiments/' + resFolder
        
        if not os.path.isdir(resFolderPath):
            os.mkdir(resFolderPath)
        logger.add(os.path.join(experimentId, resFolder, "{time}.log"))
        
        
        
        if len(protein_seq) > 999:
            logger.info('skipping %s'%test_pdblist[ik])
        else:
            
            with open('configs/model_config.json') as f: 
                configs = json.load(f)
                
            configs = EasyDict(configs)

            vocab = getVocFromCache()
            device = configs.device


            smiVoc = vocab.smiVoc
            proVoc = vocab.proVoc
            smiMaxLen = vocab.smi_max_len
            proMaxLen = vocab.pro_max_len

            model = TMOE(vocab,  **configs).to(device)
            model.load_state_dict(torch.load('model/3.pt'))
            
            model = model.to(device) # 模型加载到设备0
            model.eval()
            
            node = Node(path=['&'],smiMaxLen=vocab.smi_max_len)
            
            times = 0
            allScores = []
            allValidSmiles = []
            allSmiles = []
            
            for i in range(15):
                node, scores, validSmiles, smiles = MCTS(node)
                
                allScores.append(scores)
                allValidSmiles.append(validSmiles)
                allSmiles.append(smiles)
                
                with open(resFolderPath+'/all.txt', 'a') as f:
                        for smi in smiles:
                            f.write(f"{smi}\n" )
                            
                with open(resFolderPath+'/valid.txt', 'a') as f:
                        for smi, score in zip(validSmiles, scores):
                            f.write(f"{smi}"+ " " + f"{score}\n")
    
        
            alphaSmi = ''
            affinity = 500
            if node.path[-1] == '$':
                alphaSmi = ''.join(node.path[1:-1])
                if Chem.MolFromSmiles(alphaSmi):
                    logger.success(alphaSmi)
                    if alphaSmi in infoma:
                        affinity = infoma[alphaSmi]
                    else:
                        affinity = CaculateAffinity(alphaSmi, file_protein=pro_file[ik], file_lig_ref=ligand_file[ik], out_path=resFolderPath)
                    
                    logger.success(-affinity)
                else:
                    logger.error('invalid: ' + ''.join(node.path))
            else:
                logger.error('abnormal ending: ' + ''.join(node.path))
                
        with open(resFolderPath+'/alpha.txt', 'w') as f:
            f.write(f"{alphaSmi}\n")


        ET = time.time()
        logger.info('time {}'.format((ET-ST)//60))

