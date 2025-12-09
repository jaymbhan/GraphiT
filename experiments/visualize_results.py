import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_val_acc_by_dataset(results_file='experiment_results2.csv', output_dir='figures'):
    df = pd.read_csv(results_file)
    datasets = df['dataset'].unique()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract val_accs column and parse the lists
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    additional_colors = plt.cm.Set3(np.linspace(0, 1, 5))
    all_colors = list(colors) + list(additional_colors)

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(12, 7))
        dataset_df = df[df['dataset'] == dataset]
        for idx, (_, row) in enumerate(dataset_df.iterrows()):
            pos_enc = row['pos_enc']
            val_accs_str = row['val_accs']
            # Parse the string representation of list
            if pd.notna(val_accs_str) and val_accs_str != '':
                # Remove brackets and split by comma
                val_accs = [float(x.strip()) for x in val_accs_str.strip('[]').split(',')]
                # Epochs are logged at 1, 20, 40, 60, ..., 300
                epochs = [1] + list(range(20, 20 * len(val_accs), 20))
                ax.plot(epochs, val_accs, marker='o', linewidth=2, label=pos_enc, color=all_colors[idx % len(all_colors)], markersize=4)

        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
        ax.set_title(f'Validation Accuracy over Epochs - {dataset}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, 310])
        ax.tick_params(axis='both', labelsize=11)
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'val_acc_{dataset.lower()}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f'\nAll plots saved to {output_dir}/')

def plot_train_loss_by_dataset(results_file='experiment_results2.csv', output_dir='figures'):
    df = pd.read_csv(results_file)
    datasets = df['dataset'].unique()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract train_loss column and parse the lists
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    additional_colors = plt.cm.Set3(np.linspace(0, 1, 5))
    all_colors = list(colors) + list(additional_colors)

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(12, 7))
        dataset_df = df[df['dataset'] == dataset]
        for idx, (_, row) in enumerate(dataset_df.iterrows()):
            pos_enc = row['pos_enc']
            train_loss_str = row['train_loss']
            # Parse the string representation of list
            if pd.notna(train_loss_str) and train_loss_str != '':
                # Remove brackets and split by comma
                train_losses = [float(x.strip()) for x in train_loss_str.strip('[]').split(',')]
                # Epochs are logged at 1, 20, 40, 60, ..., 300
                epochs = [1] + list(range(20, 20 * len(train_losses), 20))
                ax.plot(epochs, train_losses, marker='o', linewidth=2, label=pos_enc, color=all_colors[idx % len(all_colors)], markersize=4)

        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
        ax.set_title(f'Training Loss over Epochs - {dataset}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.set_xlim([0, 310])
        ax.tick_params(axis='both', labelsize=11)
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'train_loss_{dataset.lower()}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f'\nAll plots saved to {output_dir}/')

def main():
    results_file = 'experiment_results2.csv'
    if not os.path.exists(results_file):
        return
    plot_val_acc_by_dataset(results_file=results_file, output_dir='figures')
    plot_train_loss_by_dataset(results_file=results_file, output_dir='figures')

if __name__ == "__main__":
    main()
