import argparse
from collections import defaultdict
import torch as th
from torch import nn, optim
from tqdm import tqdm
import time
import sys
import os
from pathlib import Path

# Add sasrec directory to path to import dataset utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../sasrec'))
from data.dataset import read_dataset, AugmentedDataset

from torch.utils.data import DataLoader
from didn import DIDN
from train_runner import train, evaluate, print_results

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset-dir', 
    # default='../datasets/tmall/', 
    default='../datasets/retailrocket/', 
    # default='../datasets/lastfm/',
    help='the dataset directory'
)
parser.add_argument(
    '--embedding-dim', 
    type=int, 
    default=64, 
    help='the embedding size'
)
parser.add_argument(
    '--hidden-size', 
    type=int, 
    default=64, 
    help='the hidden size'
)
parser.add_argument(
    '--num-layers', 
    type=int, 
    default=1, 
    help='the number of layers'
)
parser.add_argument(
    '--lr', 
    type=float, 
    default=1e-3, 
    help='the learning rate'
)
parser.add_argument(
    '--batch-size', 
    type=int, 
    default=512, 
    help='the batch size for training'
)
parser.add_argument(
    '--epochs', 
    type=int, 
    default=100, 
    help='the number of training epochs'
)
parser.add_argument(
    '--weight-decay',
    type=float,
    default=1e-4,
    help='the parameter for L2 regularization',
)
parser.add_argument(
    '--Ks',
    default='10,20',
    help='the values of K in evaluation metrics, separated by commas',
)
parser.add_argument(
    '--patience',
    type=int,
    default=5,
    help='the number of epochs that the performance does not improves after which the training stops',
)
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='the number of processes to load the input graphs',
)
parser.add_argument(
    '--valid-split',
    type=float,
    default=None,
    help='the fraction for the validation set',
)
parser.add_argument(
    '--log-interval',
    type=int,
    default=100,
    help='print the loss after this number of iterations',
)
parser.add_argument(
    '--max-session-len',
    type=int,
    default=19,
    help='maximum length of a session',
)

# DIDN specific arguments
parser.add_argument('--alpha1', type=float, default=0.1, help='alpha1')
parser.add_argument('--alpha2', type=float, default=0.1, help='alpha2')
parser.add_argument('--alpha3', type=float, default=0.1, help='alpha3')
parser.add_argument('--neighbor-num', type=int, default=5, help='neighbor number')
parser.add_argument('--position-embed-dim', type=int, default=64, help='position embedding dimension')

parser.add_argument('--logit_type', type=str, default='dot', help='the type of logit function', choices=['dot', 'cos', 'euclidean'])
parser.add_argument('--scale', type=float, default=1.0, help='the scale of the logit function')

args = parser.parse_args()
print(args)

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

dataset_dir = Path(args.dataset_dir)
args.Ks = [int(K) for K in args.Ks.split(',')]
print(f'reading dataset from {dataset_dir}')
train_sessions, valid_sessions, test_sessions, num_items = read_dataset(dataset_dir)

if args.valid_split is not None:
    num_valid = int(len(train_sessions) * args.valid_split)
    test_sessions = train_sessions[-num_valid:]
    train_sessions = train_sessions[:-num_valid]

train_set = AugmentedDataset(train_sessions)
valid_set = AugmentedDataset(valid_sessions)
test_set = AugmentedDataset(test_sessions)

def collate_fn(samples):
    sessions, labels = zip(*samples)
    # Calculate lengths (number of non-zero elements)
    # Assuming 0 is padding and it's padded from the front
    lengths = []
    for seq in sessions:
        l = sum(1 for x in seq if x != 0)
        if l == 0: l = 1 # safeguard
        lengths.append(l)
        
    sessions = th.LongTensor(sessions)
    labels = th.LongTensor(labels)
    lengths = th.LongTensor(lengths)
    return sessions, lengths, labels

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
)

test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
)

# DIDN Initialization
# n_items, hidden_size, embedding_dim, batch_size, max_len, position_embed_dim, alpha1,  alpha2, alpha3, pos_num, neighbor_num, n_layers=1
model = DIDN(
    n_items=num_items,
    hidden_size=args.hidden_size,
    embedding_dim=args.embedding_dim,
    batch_size=args.batch_size,
    max_len=args.max_session_len,
    position_embed_dim=args.position_embed_dim,
    alpha1=args.alpha1,
    alpha2=args.alpha2,
    alpha3=args.alpha3,
    pos_num=2000, # Sufficiently large
    neighbor_num=args.neighbor_num,
    n_layers=args.num_layers,
    logit_type=args.logit_type,
    scale=args.scale
)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

patience = args.patience
best_results = defaultdict(float)
best_epochs = defaultdict(int)

timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
if not os.path.exists('best_model_weight'):
    os.makedirs('best_model_weight')
best_model_path = f'best_model_weight/best_model_didn_{timestamp}.pth'
print(f"Best model path: {best_model_path}")

# train the model
for epoch in tqdm(range(args.epochs)):
    train(model, train_loader, optimizer, criterion, device, epoch, args.log_interval)
    # validate
    results = evaluate(model, valid_loader, device, Ks=args.Ks)
    if results['HR@20'] > best_results['HR@20']:
        best_results = results
        best_epochs = epoch
        patience = args.patience
        th.save(model.state_dict(), best_model_path)
    else:
        patience -= 1
        print(f"patience remaining: {patience}")
        if patience == 0:
            break

print("Best validation results:")
print_results(best_results)

# test the model
print("Test results:")
model.load_state_dict(th.load(best_model_path))
test_results = evaluate(model, test_loader, device, Ks=args.Ks)

