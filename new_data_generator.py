import itertools
import numpy as np
import networkx as nx
import random
import os

def generate_random_graph(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for u, v in itertools.combinations(G.nodes, 2):
        if random.random()<0.5:
            G.add_edge(u,v)

    return G

def get_clique_size(G):
    return max(len(c) for c in nx.find_cliques(G))

def generate_clique_dataset(num_graphs, min_nodes, max_nodes, output_dir, dataset_name):
    """
    Generate a dataset of random graphs with their clique sizes as labels in TUDataset format.

    Args:
        num_graphs: Number of graphs to generate
        min_nodes: Minimum number of nodes per graph
        max_nodes: Maximum number of nodes per graph
        output_dir: Directory to save the dataset files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate random graphs and compute their clique sizes
    graphs = []
    labels = []
    labels_dict = {4: 0, 5: 0, 6: 0, 7: 0}
    target_per_size = 1000

    total_generated = 0
    while any(count < target_per_size for count in labels_dict.values()):
        n_nodes = random.randint(min_nodes, max_nodes)

        G = generate_random_graph(n_nodes)
        clique_size = get_clique_size(G)

        if clique_size in labels_dict and labels_dict[clique_size] < target_per_size:
            graphs.append(G)
            labels.append(clique_size)
            labels_dict[clique_size] += 1
            print(f"Added graph with clique size {clique_size}. Progress: {labels_dict}")

        total_generated += 1

    # Node numbering is global across all graphs
    node_counter = 0
    edge_list = []
    graph_indicators = []
    node_labels = []

    for graph_id, G in enumerate(graphs, start=1):
        mapping = {old: node_counter + i for i, old in enumerate(G.nodes())}
        G_relabeled = nx.relabel_nodes(G, mapping)

        for u, v in G_relabeled.edges():
            edge_list.append((u + 1, v + 1))
            edge_list.append((v + 1, u + 1))

        for node in G_relabeled.nodes():
            graph_indicators.append(graph_id)
            node_labels.append(1)

        node_counter += len(G.nodes())

    with open(os.path.join(output_dir, f'{dataset_name}_A.txt'), 'w') as f:
        for u, v in sorted(edge_list):
            f.write(f'{u}, {v}\n')

    with open(os.path.join(output_dir, f'{dataset_name}_graph_indicator.txt'), 'w') as f:
        for indicator in graph_indicators:
            f.write(f'{indicator}\n')

    with open(os.path.join(output_dir, f'{dataset_name}_graph_labels.txt'), 'w') as f:
        for label in labels:
            f.write(f'{label}\n')

    with open(os.path.join(output_dir, f'{dataset_name}_node_labels.txt'), 'w') as f:
        for label in node_labels:
            f.write(f'{label}\n')

    print(f"Dataset saved!")
    return graphs, labels


def create_fold_indices(num_graphs, dataset_name, num_folds=10, train_ratio=0.8, val_ratio=0.1):
    """
    Create train/val/test split indices for cross-validation.
    """
    fold_dir = f'dataset/fold-idx/{dataset_name}'
    inner_fold_dir = os.path.join(fold_dir, 'inner_folds')
    os.makedirs(inner_fold_dir, exist_ok=True)

    all_indices = np.arange(num_graphs)

    test_size = int(num_graphs * (1 - train_ratio - val_ratio))
    val_size = int(num_graphs * val_ratio)
    train_size = num_graphs - test_size - val_size

    for fold in range(1, num_folds + 1):
        np.random.shuffle(all_indices)
        train_idx = all_indices[:train_size]
        val_idx = all_indices[train_size:train_size + val_size]
        test_idx = all_indices[train_size + val_size:]
        test_file = os.path.join(fold_dir, f'test_idx-{fold}.txt')
        np.savetxt(test_file, test_idx, fmt='%d')
        train_file = os.path.join(inner_fold_dir, f'train_idx-{fold}-1.txt')
        val_file = os.path.join(inner_fold_dir, f'val_idx-{fold}-1.txt')
        np.savetxt(train_file, train_idx, fmt='%d')
        np.savetxt(val_file, val_idx, fmt='%d')

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    dataset_name = "CLIQUE"
    num_graphs = 4000
    min_nodes = 20
    max_nodes = 20

    output_directory = f"dataset/TUDataset/{dataset_name}"
    graphs, labels = generate_clique_dataset(
        num_graphs=num_graphs,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        output_dir=os.path.join(output_directory, "raw"),
        dataset_name=dataset_name
    )

    create_fold_indices(num_graphs=num_graphs, dataset_name=dataset_name, num_folds=10, train_ratio=0.8, val_ratio=0.1)

    print("Dataset generated!")
