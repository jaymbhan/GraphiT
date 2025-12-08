import subprocess
import csv
import re
import os
from datetime import datetime

DATASETS = ['CLIQUE', 'BIPARTITE', 'CYCLE']
POS_ENCS = ['shortest_path', 'graphlet', 'coloring', 'no', 'estrada', 'inverse_cosine', 'clustering_coefficient', 'betweenness_centrality']

def extract_test_accuracy(output):
    match = re.search(r'test Acc ([0-9.]+)', output)
    if match:
        return float(match.group(1))
    return None

def extract_first_val_acc_1(output):
    """Extract the first epoch where validation accuracy reaches 1.0"""
    lines = output.split('\n')
    for i, line in enumerate(lines):
        match = re.search(r'Val loss: [0-9.]+ Acc: 1\.0000', line)
        if match:
            for j in range(i, max(0, i-10), -1):
                epoch_match = re.search(r'Epoch (\d+)/', lines[j])
                if epoch_match:
                    return int(epoch_match.group(1))
            return i
    return None

def run_experiment(dataset, pos_enc, fold_idx=1, beta=1.0):
    cmd = ['python', 'run_transformer_cv.py', '--dataset', dataset, '--fold-idx', str(fold_idx), '--pos-enc', pos_enc, '--beta', str(beta)]
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,timeout=3600)
        test_acc = extract_test_accuracy(result.stdout)
        first_val_acc_1_epoch = extract_first_val_acc_1(result.stdout)
        return test_acc, first_val_acc_1_epoch
    except subprocess.TimeoutExpired:
        return None, None

def main():
    results_file = 'experiment_results.csv'
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['dataset', 'pos_enc', 'test_accuracy', 'first_val_acc_1_epoch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    for dataset in DATASETS:
        for pos_enc in POS_ENCS:
            print(f"Dataset: {dataset}, Pos Enc: {pos_enc}")
            test_acc, first_val_acc_1_epoch = run_experiment(dataset, pos_enc, fold_idx=1, beta=1.0)
            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'dataset': dataset, 'pos_enc': pos_enc, 'test_accuracy': test_acc if test_acc is not None else 'FAILED', 'first_val_acc_1_epoch': first_val_acc_1_epoch if first_val_acc_1_epoch is not None else 'N/A'})

if __name__ == "__main__":
    main()
