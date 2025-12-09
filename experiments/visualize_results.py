import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_val_acc_by_dataset(results_file='experiment_results2.csv', output_dir='figures'):
    df = pd.read_csv(results_file)
    datasets = df['dataset'].unique()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    epoch_cols = [col for col in df.columns if col.startswith('val_acc_epoch_')]
    epochs = sorted([int(col.split('_')[-1]) for col in epoch_cols])
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    additional_colors = plt.cm.Set3(np.linspace(0, 1, 5))
    all_colors = list(colors) + list(additional_colors)
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(12, 7))
        dataset_df = df[df['dataset'] == dataset]
        for idx, (_, row) in enumerate(dataset_df.iterrows()):
            pos_enc = row['pos_enc']
            val_accs = []
            for epoch in epochs:
                col_name = f'val_acc_epoch_{epoch}'
                val_acc = row[col_name]
                if pd.notna(val_acc) and val_acc != '':
                    val_accs.append(float(val_acc))
                else:
                    val_accs.append(np.nan)
            ax.plot(epochs, val_accs, marker='o', linewidth=2, label=pos_enc, color=all_colors[idx % len(all_colors)], markersize=4)
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
        ax.set_title(f'Validation Accuracy over Epochs - {dataset}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.set_ylim([0, 1.05])
        ax.set_xticks(epochs[::2])
        ax.tick_params(axis='both', labelsize=11)
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'val_acc_{dataset.lower()}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f'\nAll plots saved to {output_dir}/')

def main():
    results_file = 'experiment_results2.csv'
    if not os.path.exists(results_file):
        return
    plot_val_acc_by_dataset(results_file=results_file, output_dir='figures')

if __name__ == "__main__":
    main()
