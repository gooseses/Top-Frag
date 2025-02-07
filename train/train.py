import sys
sys.path.append('.')
import torch
from easydict import EasyDict 
import json
from util.baseline import prepareFolder
from util.data import getPretrainData, getPretrainDataFromCache, getVoc, getVocFromCache
from model.tmoe import TMOE
from model.Lmser_Transformer import MFT
from loguru import logger
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import time
import pandas as pd

pretrain_location = "/teamspace/studios/this_studio/Deepest-Adjuvants/experiments/pretrain2024-09-05_04-51-59/model/0.pt"
pretrain = True

def train(model, trainLoader, smiVoc, device):
    model.train()
    batch = len(trainLoader)
    totalLoss = 0.0
    totalAcc = 0.0
    
    # Iterate over the training data.
    for protein, smile, label, proMask, smiMask in tqdm(trainLoader):
        protein = torch.as_tensor(protein).to(device)
        smile = torch.as_tensor(smile).to(device)
        proMask = torch.as_tensor(proMask).to(device)
        smiMask = torch.as_tensor(smiMask).to(device)
        label = torch.as_tensor(label, dtype=torch.long).to(device)

        # Target mask for transformer.
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(smile.shape[1]).tolist()
        tgt_mask = [tgt_mask] * 1
        tgt_mask = torch.as_tensor(tgt_mask).to(device)
        
        # Forward pass through the model.
        out,_ = model(protein, smile, smiMask, proMask, tgt_mask)
        
        # Calculate accuracy.
        cacc = ((torch.eq(torch.argmax(out, dim=-1) * smiMask, label * smiMask).sum() - (smiMask.shape[0] * smiMask.shape[1] - (smiMask).sum())) / (smiMask).sum().float()).item()
        totalAcc += cacc
        
        # Calculate loss, ignoring padding token. 
        loss = F.nll_loss(out.permute(0, 2, 1), label, ignore_index=smiVoc.index('^')) # mask padding
        totalLoss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        logger.info(f"Train Loss: {loss.item()}")
        logger.info(f"Train Acc: {cacc}")
        optimizer.step()


    # Calculate average loss and accuracy.
    avgLoss = round(totalLoss / batch, 3)
    avgAcc = round(totalAcc / batch, 3)

    return [avgAcc, avgLoss]


@torch.no_grad()
def valid(model, validLoader, smiVoc, device):
    model.eval()
    
    batch = len(validLoader)
    totalLoss = 0
    totalAcc = 0
    # Iterate over the validation data.
    for protein, smile, label, proMask, smiMask in tqdm(validLoader):
        protein = torch.as_tensor(protein).to(device)
        smile = torch.as_tensor(smile).to(device)
        proMask = torch.as_tensor(proMask).to(device)
        smiMask = torch.as_tensor(smiMask).to(device)
        label = torch.as_tensor(label, dtype= torch.long).to(device)
        
        # Generate target mask for transformer.
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(smile.shape[1]).tolist()
        tgt_mask = [tgt_mask] * 1
        tgt_mask = torch.as_tensor(tgt_mask).to(device)
        # Forward pass through the model.
        out, _ = model(protein, smile, smiMask, proMask, tgt_mask)
        
        # Calculate accuracy.
        cacc = ((torch.eq(torch.argmax(out, dim=-1) * smiMask, label * smiMask).sum() - (smiMask.shape[0] * smiMask.shape[1] - (smiMask).sum())) / (smiMask).sum().float()).item()
        totalAcc += cacc

        # Calculate loss. 
        loss = F.nll_loss(out.permute(0, 2, 1), label, ignore_index=smiVoc.index('^')) # mask padding
        totalLoss += loss.item()

        logger.info(f"Valid Loss: {loss.item()}")
        logger.info(f"Valid Acc: {cacc}")
        
    #Calculate average loss and accuracy.
    avgLoss = round(totalLoss / batch, 3)
    avgAcc = round(totalAcc / batch, 3)

    return [avgAcc, avgLoss]
    


         
if __name__ == '__main__':

    with open('/teamspace/studios/this_studio/Deepest-Adjuvants/configs/model_config.json') as f: 
        configs = json.load(f)
    
    with open('/teamspace/studios/this_studio/Deepest-Adjuvants/configs/train_settings.json') as f:
        trainsettings = json.load(f)
        
    trainsettings = EasyDict(trainsettings)
    configs = EasyDict(configs)

    model_folder, vis_folder, log_folder = prepareFolder("pretrain")
    
    with open(model_folder+'/model_settings.json', 'w') as f:
        json.dump(configs, f)

    with open(model_folder+'/training_settings.json', 'w') as f:
        json.dump(trainsettings, f)

    device = configs['device']
    
    batchsize = trainsettings.batchsize
    
    vocab = getVocFromCache()
    # try:
    trainloader, validloader = getPretrainDataFromCache(trainsettings) # change to getPretrain_datafromCache if already used
    # except:
    # trainloader, validloader = getPretrainData(trainsettings, vocab) # change to getPretrain_datafromCache if already used
    # Intialize the model.
    model = TMOE(vocab,  **configs).to(device)

    state_dict = torch.load(pretrain_location)

    model.load_state_dict(state_dict)


    # model = torch.nn.DataParallel(model, device_ids=[0])
    # Load pretrained model if specified.
    # if pretrain:
    # for key in state_dict.keys():
    #     print(key
    # Initalize optimizer and learning rate scheduler.
    optimizer = torch.optim.Adam(model.parameters(), lr = trainsettings.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = .99, last_epoch= -1)
    
    epoch_times = []
    avg_epoch_time = 0.0

    # Initalize log dataframe with training and validation metrics.
    propertys = ['accuracy', 'loss',]
    prefixs = ['training', 'validation']
    columns = [' '.join([pre, pro]) for pre in prefixs for pro in propertys]
    logdf = pd.DataFrame({}, columns=columns)
     
    # Training loop for 1000 epochs.
    for i in range(trainsettings.epochs):
                

        logger.info("Epoch: {}".format(i))
       
        # Train model and log results.
        d1 = train(model, trainloader, vocab.smiVoc, device)
        logger.info(f"Train Loss: {d1[1]}")
        logger.info(f"Train Acc: {d1[0]}")
        
        # Validate the model and log results.
        d2 = valid(model, validloader, vocab.smiVoc, device)
        
        # Save the model checkpoint.
        torch.save(model.state_dict(), model_folder+'/{}.pt'.format(i))
        
        # Add the traing/validation results to the log dataframe and save to CSV.
        logdf = logdf._append(pd.DataFrame([d1+d2], columns=columns), ignore_index=True)
        logdf.to_csv(log_folder+'/logdata.csv')
        
        # Step the learning rate scheduler.
        scheduler.step()
    
