import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

def prepare_batch(batch, device):
    inputs, labels = batch
    inputs_gpu = torch.stack([torch.LongTensor(x) for x in inputs]).to(device)
    labels_gpu = labels.to(device)
    return inputs_gpu, labels_gpu

def train(model, train_loader, optimizer, criterion, device, epoch, log_interval=100):
    model.train()
    # train for one epoch
    for i, batch in tqdm(enumerate(train_loader)):
        inputs, labels = prepare_batch(batch, device)
        optimizer.zero_grad()

        logits = model.forward(inputs)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        loss_val = loss.item()

        if i % log_interval == 0:
            print(f'Epoch {epoch} | Batch {i} | Loss {loss_val:.4f}')


def print_results(results):
    print('Metric\t' + '\t'.join(results.keys()))
    print('Value\t' + '\t'.join([f'{round(val * 100, 2):.2f}' for val in results.values()]))

def evaluate(model, data_loader, device, Ks=[20]):
    model.eval()
    num_samples = 0
    results = defaultdict(float)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            item_seq, labels = batch
            item_seq = item_seq.to(device)
            labels = labels.to(device)
            logits = model.forward(item_seq)
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