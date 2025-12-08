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

def generate_random_tree(n):
    """
    Generate a random tree (acyclic graph) with n nodes.

    Args:
        n: Number of nodes

    Returns:
        A random tree NetworkX graph
    """
    return nx.random_labeled_tree(n)

def generate_random_sparse_graph(n, edge_prob=0.15):
    """
    Generate a random sparse graph with n nodes.
    Sparse graphs are more likely to have smaller or no cycles.

    Args:
        n: Number of nodes
        edge_prob: Probability of edge between any two nodes

    Returns:
        A random sparse NetworkX graph
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for u, v in itertools.combinations(G.nodes, 2):
        if random.random() < edge_prob:
            G.add_edge(u, v)

    return G

def get_longest_cycle_length(G):
    """
    Find the length of the longest cycle in an undirected graph.

    Args:
        G: NetworkX graph

    Returns:
        Length of the longest cycle, or 0 if the graph is acyclic
    """
    if G.number_of_edges() == 0:
        return 0

    # Convert to directed graph and find all simple cycles
    # Filter out 2-cycles (which are just back-and-forth on edges)
    DG = G.to_directed()

    max_length = 0
    try:
        for cycle in nx.simple_cycles(DG):
            if len(cycle) >= 3:  # Ignore 2-cycles
                max_length = max(max_length, len(cycle))
    except nx.NetworkXError:
        return 0

    return max_length

def has_cycle(G):
    """
    Check if a graph has any cycles.

    Args:
        G: NetworkX graph

    Returns:
        1 if the graph has a cycle, 0 otherwise
    """
    try:
        nx.find_cycle(G)
        return 1
    except nx.NetworkXNoCycle:
        return 0

def generate_cycle_dataset(min_nodes, max_nodes, output_dir, dataset_name):
    """
    Generate a dataset of random graphs with their longest cycle size as labels
    in TUDataset format.

    Args:
        min_nodes: Minimum number of nodes per graph
        max_nodes: Maximum number of nodes per graph
        output_dir: Directory to save the dataset files
        dataset_name: Name of the dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    graphs = []
    labels = []
    # Target cycle sizes: 0 (acyclic), 3, 4, 5, 6, 7
    labels_dict = {9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}
    target_per_size = 500

    total_generated = 0
    while any(count < target_per_size for count in labels_dict.values()):
        n_nodes = random.randint(min_nodes, max_nodes)
        # For acyclic graphs, generate trees
        """
        if labels_dict[0] < target_per_size:
            G = generate_random_tree(n_nodes)
            cycle_len = get_longest_cycle_length(G)
            if cycle_len == 0:
                graphs.append(G)
                labels.append(0)
                labels_dict[0] += 1
                print(f"Added acyclic graph. Progress: {labels_dict}")
                total_generated += 1
                continue
        """

        # For graphs with small cycles, use sparse graphs
        G = generate_random_sparse_graph(n_nodes, edge_prob=0.1)

        cycle_len = get_longest_cycle_length(G)
        print(f"Cycle length: {cycle_len}")

        if cycle_len in labels_dict and labels_dict[cycle_len] < target_per_size:
            graphs.append(G)
            labels.append(cycle_len)
            labels_dict[cycle_len] += 1
            print(f"Added graph with longest cycle {cycle_len}. Progress: {labels_dict}")

        total_generated += 1

        # Safety check to avoid infinite loops
        if total_generated % 10000 == 0:
            print(f"Generated {total_generated} graphs so far. Current distribution: {labels_dict}")

    # Shuffle the dataset

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

    print(f"Cycle dataset saved! Class distribution: {labels_dict}")
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

    dataset_name = "CYCLE"
    min_nodes = 20
    max_nodes = 20

    output_directory = f"dataset/TUDataset/{dataset_name}"
    graphs, labels = generate_cycle_dataset(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        output_dir=os.path.join(output_directory, "raw"),
        dataset_name=dataset_name
    )

    num_graphs = len(graphs)
    create_fold_indices(num_graphs=num_graphs, dataset_name=dataset_name, num_folds=10, train_ratio=0.8, val_ratio=0.1)

    print("Dataset generated!")
