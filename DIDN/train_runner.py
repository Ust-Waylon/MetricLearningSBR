import time
from collections import defaultdict
import torch
from torch import nn, optim
from tqdm import tqdm

def prepare_batch(batch, device):
    inputs, lengths, labels = batch
    inputs_gpu = inputs.to(device)
    # lengths can stay on CPU or be list, DIDN converts it to tensor inside
    labels_gpu = labels.to(device)
    return inputs_gpu, lengths, labels_gpu

def train(model, train_loader, optimizer, criterion, device, epoch, log_interval=100):
    model.train()
    for i, batch in tqdm(enumerate(train_loader)):
        inputs, lengths, labels = prepare_batch(batch, device)
        optimizer.zero_grad()
        
        scores = model(inputs, lengths)
        loss = criterion(scores, labels)
        
        loss.backward()
        optimizer.step()
        
        if i % log_interval == 0:
            print(f'Epoch {epoch} | Batch {i} | Loss {loss.item():.4f}')

def print_results(results):
    print('Metric\t' + '\t'.join(results.keys()))
    print('Value\t' + '\t'.join([f'{round(val * 100, 2):.2f}' for val in results.values()]))

def evaluate(model, data_loader, device, Ks=[20]):
    model.eval()
    num_samples = 0
    results = defaultdict(float)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs, lengths, labels = prepare_batch(batch, device)
            
            logits = model(inputs, lengths)
            
            topk_ids = torch.topk(logits, max(Ks), dim=1)[1]
            batch_size = topk_ids.shape[0]
            num_samples += batch_size
            
            for K in Ks:
                hit_ranks = torch.where(topk_ids[:, :K] == labels.unsqueeze(1))[1] + 1
                hit_ranks = hit_ranks.float().cpu()
                results[f'HR@{K}'] += hit_ranks.numel()
                results[f'MRR@{K}'] += hit_ranks.reciprocal().sum().item()
                results[f'NDCG@{K}'] += torch.log2(1 + hit_ranks).reciprocal().sum().item()
                
    for metric in results:
        results[metric] /= num_samples
    print_results(results)
    
    return results

