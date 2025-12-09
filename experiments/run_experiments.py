import subprocess
import csv
import re
import os
import sys
from datetime import datetime

DATASETS = ['CLIQUE', 'BIPARTITE', 'CYCLE']
POS_ENCS = ['diffusion', 'pstep', 'adj', 'shortest_path', 'graphlet', 'coloring', 'no', 'estrada', 'cosine', 'clustering_coefficient', 'betweenness_centrality']

def extract_test_accuracy(output):
    match = re.search(r'test Acc ([0-9.]+)', output)
    if match:
        return float(match.group(1))
    return None

def extract_val_acc_every_20_epochs(output):
    """Extract validation accuracy at epochs 20, 40, 60, etc."""
    lines = output.split('\n')
    val_accs = []
    train_accs = []

    for i, line in enumerate(lines):
        epoch_match = re.search(r'Epoch (\d+)/', line)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            if epoch_num % 20 == 0 or epoch_num == 1:
                for j in range(i, min(len(lines), i + 10)):
                    val_match = re.search(r'Val loss: [0-9.]+ Acc: ([0-9.]+)', lines[j])
                    if val_match:
                        val_accs.append(float(val_match.group(1)))
                        break
                for j in range(i, min(len(lines), i + 10)):
                    train_match = re.search(r'Train loss: ([0-9.]+)', lines[j])
                    if train_match:
                        train_accs.append(float(train_match.group(1)))
                        break
    return val_accs, train_accs

def run_experiment(dataset, pos_enc, fold_idx=1, beta=1.0):
    cmd = ['python', '-u', 'run_transformer_cv.py', '--dataset', dataset, '--fold-idx', str(fold_idx), '--pos-enc', pos_enc, '--beta', str(beta)]
    print(f"\nRunning: {' '.join(cmd)}")
    print("="*80)
    sys.stdout.flush()

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        output = []
        for line in process.stdout:
            print(line, end='')
            output.append(line)

        process.wait(timeout=3600)

        full_output = ''.join(output)
        test_acc = extract_test_accuracy(full_output)
        val_accs_every_20, train_accs_every_20 = extract_val_acc_every_20_epochs(full_output)
        return test_acc, val_accs_every_20, train_accs_every_20
    except subprocess.TimeoutExpired:
        return None, [], []

def main():
    results_file = 'experiment_results2.csv'

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['dataset', 'pos_enc', 'test_accuracy', 'val_accs', 'train_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    for dataset in DATASETS:
        for pos_enc in POS_ENCS:
            print(f"Dataset: {dataset}, Pos Enc: {pos_enc}")
            test_acc, val_accs_every_20, train_accs_every_20 = run_experiment(dataset, pos_enc, fold_idx=1, beta=1.0)

            row_data = {
                'dataset': dataset,
                'pos_enc': pos_enc,
                'test_accuracy': test_acc if test_acc is not None else 'FAILED',
                'val_accs': val_accs_every_20 if val_accs_every_20 is not None else 'FAILED',
                'train_loss': train_accs_every_20 if train_accs_every_20 is not None else 'FAILED'
            }


            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row_data)

if __name__ == "__main__":
    main()
