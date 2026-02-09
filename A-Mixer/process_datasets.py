import os
import pickle
import argparse
import collections
from pathlib import Path

# Configuration
SOURCE_BASE = '/rwproject/kdd-db/students/wtanae/research/decoder/datasets'
DEST_BASE = '/rwproject/kdd-db/students/wtanae/research/decoder/Atten-Mixer-torch/data'

def load_sessions(file_path):
    """Reads comma-separated sessions from a text file."""
    sessions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse integers
            try:
                items = list(map(int, line.split(',')))
                sessions.append(items)
            except ValueError:
                print(f"Warning: Could not parse line in {file_path}: {line}")
    return sessions

def augment_session(session):
    """
    Splits a session [i1, i2, ..., in] into multiple (input, target) pairs:
    ([i1], i2)
    ([i1, i2], i3)
    ...
    ([i1, ..., in-1], in)
    """
    inputs = []
    targets = []
    if len(session) < 2:
        return inputs, targets
    
    for i in range(1, len(session)):
        inputs.append(session[:i])
        targets.append(session[i])
    
    return inputs, targets

def process_dataset(dataset_name):
    print(f"Processing {dataset_name}...")
    
    src_dir = os.path.join(SOURCE_BASE, dataset_name)
    dest_dir = os.path.join(DEST_BASE, dataset_name)
    
    if not os.path.exists(src_dir):
        print(f"Error: Source directory {src_dir} does not exist.")
        return

    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Process Training Data
    train_src = os.path.join(src_dir, 'train.txt')
    if os.path.exists(train_src):
        print(f"  Loading {train_src}...")
        train_sessions = load_sessions(train_src)
        
        # Save all_train_seq.txt (original sequences)
        print("  Saving all_train_seq.txt...")
        with open(os.path.join(dest_dir, 'all_train_seq.txt'), 'wb') as f:
            pickle.dump(train_sessions, f)
            
        # Compute item counts
        print("  Computing item counts...")
        item_counts = collections.defaultdict(int)
        for seq in train_sessions:
            for item in seq:
                item_counts[item] += 1
        
        with open(os.path.join(dest_dir, 'item_count.txt'), 'wb') as f:
            pickle.dump(item_counts, f)
            
        # Generate augmented training data
        print("  Augmenting training data...")
        train_inputs = []
        train_targets = []
        for seq in train_sessions:
            inps, tgts = augment_session(seq)
            train_inputs.extend(inps)
            train_targets.extend(tgts)
            
        print(f"  Generated {len(train_inputs)} training samples.")
        with open(os.path.join(dest_dir, 'train.txt'), 'wb') as f:
            pickle.dump([train_inputs, train_targets], f)
    else:
        print(f"Warning: {train_src} not found.")

    # 2. Process Test Data
    test_src = os.path.join(src_dir, 'test.txt')
    if os.path.exists(test_src):
        print(f"  Loading {test_src}...")
        test_sessions = load_sessions(test_src)
        
        # Generate augmented test data (as observed in diginetica sample)
        print("  Augmenting test data...")
        test_inputs = []
        test_targets = []
        for seq in test_sessions:
            inps, tgts = augment_session(seq)
            test_inputs.extend(inps)
            test_targets.extend(tgts)
            
        print(f"  Generated {len(test_inputs)} test samples.")
        with open(os.path.join(dest_dir, 'test.txt'), 'wb') as f:
            pickle.dump([test_inputs, test_targets], f)
    else:
        print(f"Warning: {test_src} not found.")
        
    print(f"Done processing {dataset_name}.\n")

def find_datasets(base_path):
    """Recursively finds directories containing train.txt."""
    datasets = []
    for root, dirs, files in os.walk(base_path):
        if 'train.txt' in files:
            # Get path relative to base
            rel_path = os.path.relpath(root, base_path)
            datasets.append(rel_path)
    return sorted(datasets)

def main():
    parser = argparse.ArgumentParser(description="Adapt datasets to Atten-Mixer format.")
    parser.add_argument('datasets', nargs='*', help='List of dataset names/paths to process. If empty, process all found in source folder.')
    args = parser.parse_args()
    
    if args.datasets:
        # If user provides arguments, check if they are datasets or containers
        targets = []
        for d in args.datasets:
            path = os.path.join(SOURCE_BASE, d)
            if os.path.exists(os.path.join(path, 'train.txt')):
                targets.append(d)
            else:
                # Try to find sub-datasets
                found = find_datasets(path)
                if found:
                    targets.extend([os.path.join(d, sub) for sub in found])
                else:
                    print(f"Warning: No datasets found in {d}")
        datasets = sorted(list(set(targets)))
    else:
        # Auto-discover all datasets
        datasets = find_datasets(SOURCE_BASE)
    
    if not datasets:
        print("No datasets found to process.")
        return

    print(f"Found datasets to process: {datasets}")
    
    for dataset in datasets:
        process_dataset(dataset)

if __name__ == "__main__":
    main()

