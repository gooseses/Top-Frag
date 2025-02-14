import torch
from configs.ModelConfigs import ModelConfigs
from configs.TrainConfigs import TrainConfigs
from data.util import get_dataloaders
from model.MolSeek import TopFrag
from loguru import logger
from tqdm import tqdm_gui
from torch import nn
import argparse

def calculate_maxvio(expert_counts):
    avg_count = expert_counts.float().mean()
    min_violation = torch.min(expert_counts.float()) / avg_count
    max_violation = torch.max(expert_counts.float()) / avg_count
    return [min_violation.item(), max_violation.item()]

def train(model: TopFrag, train_loader, configs: ModelConfigs, train_settings: TrainConfigs, optimizer):
    model.train()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for pro, label in tqdm_gui(train_loader):
        out, gate_idx = model(pro, label[:, :-1])

        loss = nn.functional.nll_loss(out, label[:, 1:], ignore_index=configs.pad_idx)
        total_loss += loss.item()

        preds = torch.argmax(out, dim=-1)  
        mask = (label != configs.pad_idx)  
        correct = torch.sum((preds == label) & mask)  
        total_correct += correct.item()
        total_samples += torch.sum(mask).item()  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i, layer in enumerate(model.layers):
            if not layer.isMoe:
                continue
            e_expert_counts = torch.bincount(gate_idx[i][0], minlength=configs.num_experts)
            d_expert_counts = torch.bincount(gate_idx[i][1], minlength=configs.num_experts)

            avg_e_count = e_expert_counts.float().mean()
            avg_d_count = d_expert_counts.float().mean()

            for j, count in enumerate(e_expert_counts):
                error = avg_e_count - count.float()
                layer.encoder.feed_forward.gate.expert_bias[j] += train_settings.update_rate * torch.sign(error).detach()

            for k, count in enumerate(d_expert_counts):
                error = avg_d_count - count.float()
                layer.decoder.feed_forward.gate.expert_bias[k] += train_settings.update_rate * torch.sign(error).detach()

    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / len(train_loader)
    return accuracy, avg_loss

def valid(model: TopFrag, valid_loader, configs: ModelConfigs):
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for pro, label in tqdm_gui(valid_loader):
        out, = model(pro, label[:-1])

        loss = nn.functional.nll_loss(out, label, ignore_index=configs.pad_idx)
        total_loss += loss.item()

        preds = torch.argmax(out, dim=-1)
        mask = (label != configs.pad_idx) 
        correct = torch.sum((preds == label) & mask)
        total_correct += correct.item()
        total_samples += torch.sum(mask).item()  

    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / len(valid_loader)
    return accuracy, avg_loss

@torch.no_grad()
def calculate_metrics(expert_counts, avg_count):
    min_violation = torch.min(expert_counts.float()) / avg_count
    max_violation = torch.max(expert_counts.float()) / avg_count
    return min_violation.item(), max_violation.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', default=False, type=bool)
    parser.add_argument('--useCache', default=True, type=bool)

    args = parser.parse_args()

    configs = ModelConfigs()
    train_settings = TrainConfigs()

    model = TopFrag(configs)

    if(args.pretrain):
        state_dict = torch.load(f'{PRETRAIN}')
        model.load_state_dict(state_dict)
    
    trainloader, validloader = get_dataloaders(train_settings, args.useCache)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)

    # Training loop for 1000 epochs
    for epoch in range(train_settings.epochs):
        logger.info(f"Epoch: {epoch}")

        train_accuracy, train_loss = train(model, trainloader, configs, train_settings, optimizer)
        logger.info(f"Training Accuracy: {train_accuracy:.2f}% | Loss: {train_loss:.4f}")

        valid_accuracy, valid_loss = valid(model, validloader, configs, train_settings)
        logger.info(f"Validation Accuracy: {valid_accuracy:.2f}% | Loss: {valid_loss:.4f}")

        torch.save(model.state_dict(), f'{MODEL}/epoch_{epoch}.pt')

        scheduler.step()
