import itertools
import numpy as np
import networkx as nx
import random
import os

def generate_random_graph(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for u, v in itertools.combinations(G.nodes, 2):
        if random.random()<0.1:
            G.add_edge(u,v)

    print(f"Random graph with {G.number_of_edges()} edges")
    return G

def is_bipartite(G):
    """
    Check if a graph is bipartite.

    Args:
        G: NetworkX graph

    Returns:
        1 if the graph is bipartite, 0 otherwise
    """
    return 1 if nx.is_bipartite(G) else 0

def generate_random_bipartite_graph(n):
    """
    Generate a random bipartite graph with n nodes.

    Args:
        n: Total number of nodes

    Returns:
        A random bipartite NetworkX graph
    """
    # Randomly split nodes into two partitions
    split = random.randint(1, n - 1)
    partition_a = list(range(split))
    partition_b = list(range(split, n))

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Only add edges between partitions (guarantees bipartite)
    for u in partition_a:
        for v in partition_b:
            if random.random() < 0.5:
                G.add_edge(u, v)

    print(f"Random bipartite graph with {G.number_of_edges()} edges")

    return G

def generate_random_tripartite_graph(n):
    """
    Generate a random tripartite graph with n nodes.

    Args:
        n: Total number of nodes

    Returns:
        A random tripartite NetworkX graph
    """
    # Randomly split nodes into three partitions
    split1 = random.randint(1, n - 1)
    split2 = random.randint(1, n - 1)
    partition_a = list(range(split1))
    partition_b = list(range(split1, split2))
    partition_c = list(range(split2, n))

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Only add edges between partitions (guarantees bipartite)
    for u in partition_a:
        for v in partition_b:
            if random.random() < 0.5:
                G.add_edge(u, v)
        for v in partition_c:
            if random.random() < 0.5:
                G.add_edge(u, v)

    for u in partition_b:
        for v in partition_c:
            if random.random() < 0.5:
                G.add_edge(u, v)

    print(f"Random bipartite graph with {G.number_of_edges()} edges")

    return G


def generate_bipartite_dataset(min_nodes, max_nodes, output_dir, dataset_name):
    """
    Generate a dataset of random graphs with bipartite labels (1 if bipartite, 0 otherwise)
    in TUDataset format.

    Args:
        num_graphs: Number of graphs to generate (will be split evenly between classes)
        min_nodes: Minimum number of nodes per graph
        max_nodes: Maximum number of nodes per graph
        output_dir: Directory to save the dataset files
        dataset_name: Name of the dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    graphs = []
    labels = []
    labels_dict = {0: 0, 1: 0}  # 0: not bipartite, 1: bipartite
    target_per_class = 2000

    # Generate bipartite graphs (label=1)
    while labels_dict[1] < target_per_class:
        n_nodes = random.randint(min_nodes, max_nodes)
        G = generate_random_bipartite_graph(n_nodes)

        # Verify it's actually bipartite and has at least one edge
        if nx.is_bipartite(G) and G.number_of_edges() > 0:
            graphs.append(G)
            labels.append(1)
            labels_dict[1] += 1
            print(f"Added bipartite graph. Progress: {labels_dict[1]}")

    # Generate non-bipartite graphs (label=0)
    while labels_dict[0] < target_per_class:
        n_nodes = random.randint(min_nodes, max_nodes)
        G = generate_random_tripartite_graph(n_nodes)

        # Only keep if not bipartite and has at least one edge
        if not nx.is_bipartite(G) and G.number_of_edges() > 0:
            graphs.append(G)
            labels.append(0)
            labels_dict[0] += 1
            print(f"Added non-bipartite graph. Progress: {labels_dict[0]}")

    # Shuffle the dataset to mix bipartite and non-bipartite graphs
    combined = list(zip(graphs, labels))
    random.shuffle(combined)
    graphs, labels = zip(*combined)
    graphs = list(graphs)
    labels = list(labels)

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

    print(f"Bipartite dataset saved! Class distribution: {labels_dict}")
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

    dataset_name = "BIPARTITE"
    num_graphs = 4000
    min_nodes = 20
    max_nodes = 20

    output_directory = f"dataset/TUDataset/{dataset_name}"
    graphs, labels = generate_bipartite_dataset(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        output_dir=os.path.join(output_directory, "raw"),
        dataset_name=dataset_name
    )

    create_fold_indices(num_graphs=num_graphs, dataset_name=dataset_name, num_folds=10, train_ratio=0.8, val_ratio=0.1)

    print("Dataset generated!")
