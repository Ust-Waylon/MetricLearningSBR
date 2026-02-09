import os
from datetime import datetime
from time import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models import SessionGraphAttn
from dataset import SessionData

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica',
                    help='dataset name: diginetica/gowalla/lastfm')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=256, help='hidden state size')
parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]0.001

parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', type=str2bool, default=False, help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
parser.add_argument('--alpha', type=float, default=0.75, help='parameter for beta distribution')
parser.add_argument('--norm', type=str2bool, default=True, help='adapt NISER, l2 norm over item and session embedding')
parser.add_argument('--scale', default=16, type=float, help='scaling factor sigma')
parser.add_argument('--heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--use_lp_pool', type=str, default="True")
parser.add_argument('--train_flag', type=str, default="True")
parser.add_argument('--lr_dc', type=float, default=0.1)
parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--softmax', type=str2bool, default=True)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--dot', default=True, action='store_true')
parser.add_argument('--last_k', type=int, default=7)
parser.add_argument('--l_p', type=int, default=4)

parser.add_argument('--logit_type', type=str, default='dot', choices=['dot', 'euclidean'])
parser.add_argument_group()

opt = parser.parse_args()
print(opt)

hyperparameter_defaults = vars(opt)
config = hyperparameter_defaults

class AreaAttnModel(nn.Module):

    def __init__(self, opt, n_node):
        super().__init__()

        self.opt = opt
        self.cnt = 0
        self.best_res = [0, 0]
        self.model = SessionGraphAttn(opt, n_node)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch):
        alias_inputs, A, items, mask, mask1, targets, n_node = batch

        alias_inputs = alias_inputs.squeeze().to(self.device)
        A = A.squeeze().to(self.device)
        items = items.squeeze().to(self.device)
        mask = mask.squeeze().to(self.device)
        mask1 = mask1.squeeze().to(self.device)
        targets = targets.squeeze().to(self.device)
        n_node = n_node.squeeze().to(self.device)

        hidden = self(items)

        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask)
        loss = self.model.loss_function(scores, targets - 1)

        return loss

    def validation_step(self, batch):
        alias_inputs, A, items, mask, mask1, targets, n_node = batch
        
        alias_inputs = alias_inputs.squeeze().to(self.device)
        A = A.squeeze().to(self.device)
        items = items.squeeze().to(self.device)
        mask = mask.squeeze().to(self.device)
        mask1 = mask1.squeeze().to(self.device)
        targets = targets.squeeze().to(self.device)
        n_node = n_node.squeeze().to(self.device)
        
        hidden = self(items)
        assert not torch.isnan(hidden).any()
        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12  # l2 norm over session embedding
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask)
        targets = targets.cpu().detach().numpy()
        sub_scores = scores.topk(20)[1]
        sub_scores = sub_scores.cpu().detach().numpy()
        res = []
        for score, target in zip(sub_scores, targets):
            hit = float(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr = 0
            else:
                mrr = 1 / (np.where(score == target - 1)[0][0] + 1)
            phi = 0
            res.append([hit, mrr, phi / 20])

        return torch.tensor(res)

    def test_step(self, batch):
        alias_inputs, A, items, mask, mask1, targets, n_node = batch
        
        alias_inputs = alias_inputs.squeeze().to(self.device)
        A = A.squeeze().to(self.device)
        items = items.squeeze().to(self.device)
        mask = mask.squeeze().to(self.device)
        mask1 = mask1.squeeze().to(self.device)
        targets = targets.squeeze().to(self.device)
        n_node = n_node.squeeze().to(self.device)
        
        hidden = self(items)
        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12  # l2 norm over session embedding
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask)
        targets = targets.cpu().detach().numpy()
        sub_scores = scores.topk(20)[1]
        sub_scores = sub_scores.cpu().detach().numpy()
        res = []
        for score, target in zip(sub_scores, targets):
            hit = float(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr = 0
            else:
                mrr = 1 / (np.where(score == target - 1)[0][0] + 1)
            res.append([hit, mrr])

        return torch.tensor(res)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return int(np.argmax(memory_available))

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    start_time = time()
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    end_time = time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, device):
    model.eval()
    all_results = []
    start_time = time()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            results = model.validation_step(batch)
            all_results.append(results)
    end_time = time()
    print(f"Validation time: {end_time - start_time:.2f} seconds")

    output = torch.cat(all_results, dim=0)
    hit = torch.mean(output[:, 0]) * 100
    mrr = torch.mean(output[:, 1]) * 100
    arp = torch.sum(output[:, 2]) / len(output)
    
    return hit.item(), mrr.item(), arp.item()

def test_epoch(model, test_loader, device):
    model.eval()
    all_results = []
    start_time = time()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            results = model.test_step(batch)
            all_results.append(results)
    end_time = time()
    print(f"Testing time: {end_time - start_time:.2f} seconds")
    
    output = torch.cat(all_results, dim=0)
    hit = torch.mean(output[:, 0]) * 100
    mrr = torch.mean(output[:, 1]) * 100
    
    return hit.item(), mrr.item()

def main():
    # Set random seeds for reproducibility
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set dataset parameters
    data_path = f'../data/{opt.dataset}/item_count.txt'
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            item_counts = pickle.load(f)
        n_node = max(item_counts.keys()) + 1
    else:
        n_node = 310 # default fallback or raise error

    # Setup device
    if torch.cuda.is_available():
        # device = torch.device(f'cuda:{get_freer_gpu()}')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f'../checkpoint/Atten-Mixer_{opt.dataset}_{time_str}.pt'

    # Load data
    session_data = SessionData(name=opt.dataset, batch_size=opt.batchSize)
    train_loader = session_data.train_dataloader()
    val_loader = session_data.val_dataloader()
    test_loader = session_data.test_dataloader()

    # Initialize model
    model = AreaAttnModel(opt=opt, n_node=n_node)
    model.to(device)

    if opt.train_flag == "True":
        # Setup optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        # Training loop
        best_mrr = 0
        patience_counter = 0
        
        for epoch in range(opt.epoch):
            print(f"\nEpoch {epoch + 1}/{opt.epoch}")
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device)
            
            # Validate
            hit, mrr, arp = validate_epoch(model, val_loader, device)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation - Hit@20: {hit:.2f}, MRR@20: {mrr:.2f}, ARP@20: {arp:.4f}")
            
            # Update best results
            if mrr > model.best_res[1]:
                model.best_res[1] = mrr
                best_mrr = mrr
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), checkpoint_path)
                print(f"New best MRR: {mrr:.2f} - Model saved!")
            else:
                patience_counter += 1
            
            if hit > model.best_res[0]:
                model.best_res[0] = hit
            
            # Early stopping
            if patience_counter >= opt.patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            # Update learning rate
            scheduler.step()
        
        print(f"\nTraining completed. Best Hit@20: {model.best_res[0]:.2f}, Best MRR@20: {model.best_res[1]:.2f}")

        # Load model for testing
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        # Test
        hit, mrr = test_epoch(model, test_loader, device)
        print(f"Test Results - Hit@20: {hit:.2f}, MRR@20: {mrr:.2f}")
        
    else:
        # Load model for testing
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        # Test
        hit, mrr = test_epoch(model, test_loader, device)
        print(f"Test Results - Hit@20: {hit:.2f}, MRR@20: {mrr:.2f}")

if __name__ == "__main__":
    main() 