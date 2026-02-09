import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset-dir', 
    default='../datasets/tmall/',
    # default='../datasets/retailrocket/',
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
    '--num-layers', 
    type=int, 
    default=3, 
    help='the number of layers'
)
parser.add_argument(
    '--num-heads',
    type=int,
    default=2,
    help='the number of attention heads'
)
parser.add_argument(
    '--feat-drop', 
    type=float, 
    default=0.2, 
    help='the dropout ratio for features'
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
parser.add_argument(
    '--logit_type',
    type=str,
    default='dot',
    help='the type of logit function',
    choices=['dot', 'euclidean'],
)
parser.add_argument(
    '--scale',
    type=float,
    default=1.0,
    help='the scale of the logit function',
)
parser.add_argument(
    '--batchnorm',
    action='store_true',
    help='use batch normalization',
)
args = parser.parse_args()
print(args)

# import sys
# sys.path.append('..')
from pathlib import Path
import torch as th
from torch import nn, optim
from tqdm import tqdm
import time
# th.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
from data.dataset import read_dataset, AugmentedDataset
from sasrec import SelfAttentiveSessionEncoder
from train_runner import train, evaluate, evaluate_with_XBOX, print_results

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
    sessions = th.LongTensor(sessions)
    labels = th.LongTensor(labels)
    return sessions, labels

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

model = SelfAttentiveSessionEncoder(num_items=num_items, 
                                    n_layers=args.num_layers,
                                    hidden_size=args.embedding_dim,
                                    hidden_dropout_prob=args.feat_drop,
                                    max_session_length=args.max_session_len,
                                    n_head=args.num_heads,
                                    logit_type=args.logit_type,
                                    scale=args.scale,
                                    batchnorm=args.batchnorm)
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()

if args.logit_type == 'euclidean':
    # Separate item_embedding parameters to exclude them from weight decay
    item_emb_param_ids = {id(p) for p in model.item_embedding.parameters()}
    base_params = [p for p in model.parameters() if id(p) not in item_emb_param_ids]
    optimizer_params = [
        {'params': base_params, 'weight_decay': args.weight_decay},
        {'params': model.item_embedding.parameters(), 'weight_decay': 0.0}
    ]
else:
    optimizer_params = [
        {'params': model.parameters(), 'weight_decay': args.weight_decay}
    ]

# if criterion has parameters, we need to add them to the optimizer
if hasattr(criterion, 'parameters'):
    optimizer_params.append({'params': criterion.parameters(), 'weight_decay': args.weight_decay})

optimizer = optim.AdamW(optimizer_params, lr=args.lr)

patience = args.patience
best_results = defaultdict(float)
best_epochs = defaultdict(int)

timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
best_model_path = f'best_model_weight/best_model_{timestamp}.pth'
print(f"Best model path: {best_model_path}")

# train the model
for epoch in tqdm(range(args.epochs)):
    train(model, train_loader, optimizer, criterion, device, epoch, args.log_interval)
    # validate
    start_time = time.time()
    results = evaluate(model, valid_loader, device, Ks=args.Ks)
    end_time = time.time()
    print(f"Time taken for evaluation: {end_time - start_time} seconds")

    start_time = time.time()
    results_with_XBOX = evaluate_with_XBOX(model, valid_loader, device, Ks=args.Ks)
    end_time = time.time()
    print(f"Time taken for evaluation with XBOX: {end_time - start_time} seconds")

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