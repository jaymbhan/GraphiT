import os
import pickle
import hashlib
import torch
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm
import networkx as nx
from networkx.algorithms import coloring
import networkx.algorithms.coloring as coloring


class PositionEncoding(object):
    def __init__(self, savepath=None, zero_diag=False):
        self.savepath = savepath
        self.zero_diag = zero_diag

    def apply_to(self, dataset, split='train'):
        saved_pos_enc = self.load(split)
        all_pe = []
        dataset.pe_list = []
        for i, g in enumerate(dataset):
            if saved_pos_enc is None:
                pe = self.compute_pe(g)
                all_pe.append(pe)
            else:
                pe = saved_pos_enc[i]
            if self.zero_diag:
                pe = pe.clone()
                pe.diagonal()[:] = 0
            dataset.pe_list.append(pe)

        self.save(all_pe, split)

        return dataset

    def save(self, pos_enc, split):
        if self.savepath is None:
            return
        if not os.path.isfile(self.savepath + "." + split):
            with open(self.savepath + "." + split, 'wb') as handle:
                pickle.dump(pos_enc, handle)

    def load(self, split):
        if self.savepath is None:
            return None
        if not os.path.isfile(self.savepath + "." + split):
            return None
        with open(self.savepath + "." + split, 'rb') as handle:
            pos_enc = pickle.load(handle)
        return pos_enc

    def compute_pe(self, graph):
        pass


class DiffusionEncoding(PositionEncoding):
    def __init__(self, savepath, beta=1., use_edge_attr=False, normalization=None, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
                graph.edge_index, edge_attr, normalization=self.normalization,
                num_nodes=graph.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        L = expm(-self.beta * L)
        return torch.from_numpy(L.toarray())


class PStepRWEncoding(PositionEncoding):
    def __init__(self, savepath, p=2, beta=0.5, use_edge_attr=False, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.p = p
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization,
            num_nodes=graph.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        L = sp.identity(L.shape[0], dtype=L.dtype) - self.beta * L
        tmp = L
        for _ in range(self.p - 1):
            tmp = tmp.dot(L)
        return torch.from_numpy(tmp.toarray())


class AdjEncoding(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def compute_pe(self, graph):
        return to_dense_adj(graph.edge_index)

class FullEncoding(PositionEncoding):
    def __init__(self, savepath, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)

    def compute_pe(self, graph):
        return torch.ones((graph.num_nodes, graph.num_nodes))

## Absolute position encoding
class LapEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim+1]).float()

    def apply_to(self, dataset):
        dataset.lap_pe_list = []
        for i, g in enumerate(dataset):
            pe = self.compute_pe(g)
            dataset.lap_pe_list.append(pe)

        return dataset

class ShortestPathEncoding(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def compute_pe(self, graph):
        G = nx.from_edgelist(graph.edge_index.t().tolist())
        spl_dict = dict(nx.all_pairs_shortest_path_length(G))
        num_nodes = graph.num_nodes
        spl_array = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            if i in spl_dict:
                for j, length in spl_dict[i].items():
                    if i == j:
                        spl_array[i, j] = 1
                    else:
                        spl_array[i, j] = 2.718 ** (-length)
        return torch.from_numpy(spl_array)

class GraphletEncoding(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def count_triangles(self, graph, node):
        """Count triangles involving a node."""
        return nx.triangles(graph, node)

    def count_3_paths(self, graph, node):
        """Count 3-paths (simple paths of length 3) with node as an endpoint."""
        count = 0
        paths_at_depth = {0: [[node]]}

        for depth in range(1, 4):
            paths_at_depth[depth] = []
            for path in paths_at_depth[depth - 1]:
                last_node = path[-1]
                for neighbor in graph.neighbors(last_node):
                    if neighbor not in path:
                        new_path = path + [neighbor]
                        paths_at_depth[depth].append(new_path)
                        if depth == 3:
                            count += 1
        return count

    def count_4_cliques(self, graph, node):
        """Count 4-cliques (K4) containing the node."""
        neighbors = list(graph.neighbors(node))
        count = 0
        if len(neighbors) < 3:
            return 0
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    if (graph.has_edge(neighbors[i], neighbors[j]) and
                        graph.has_edge(neighbors[j], neighbors[k]) and
                        graph.has_edge(neighbors[i], neighbors[k])):
                        count += 1
        return count

    def count_4_cycles(self, graph, node):
        """Count 4-cycles containing the node (avoiding double-counting)."""
        neighbors = list(graph.neighbors(node))
        count = 0
        if len(neighbors) < 2:
            return 0
        for n1 in neighbors:
            for n2 in neighbors:
                if n1 == n2:
                    continue
                if graph.has_edge(n1, n2):
                    continue
                common = (set(graph.neighbors(n1)) & set(graph.neighbors(n2))) - {node}
                count += len(common)
        return count // 2

    def vector(self, graph, node):
        """Create feature vector from graphlet counts."""
        return np.array([
            self.count_triangles(graph, node),
            self.count_4_cycles(graph, node),
            self.count_3_paths(graph, node),
            self.count_4_cliques(graph, node)
        ])

    def compute_pe(self, graph):
        G = nx.from_edgelist(graph.edge_index.t().tolist())
        V = np.stack([self.vector(G, node) for node in range(graph.num_nodes)])
        K = V @ V.T
        K = K / (K.max() + 1e-8)
        return torch.from_numpy(K).float()


class ColoringEncoding(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def compute_pe(self, graph):
        G = nx.from_edgelist(graph.edge_index.t().tolist())
        coloring_result = nx.coloring.greedy_color(G, strategy='largest_first')
        arr = np.zeros((graph.num_nodes, graph.num_nodes))
        for i in range(graph.num_nodes):
            for j in range(graph.num_nodes):
                if coloring_result[i] == coloring_result[j]:
                    arr[i, j] = 1
        return torch.from_numpy(arr).float()


class EstradaEncoding(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def compute_pe(self, graph):
        adj = to_dense_adj(graph.edge_index).squeeze(0)
        N = adj.size(0)
        try:
            print("success!")
            eigvals, eigvecs = torch.linalg.eig(adj)
        except RuntimeError:
            # fallback: return zeros
            print("failed!")
            return torch.zeros(N, N)

        # Real part (adjacency of real graph should be symmetric, so imaginary is zero)
        eigvals = eigvals.real
        eigvecs = eigvecs.real

        #If not enough eigenvals, pad
        if eigvals.size(0) < N:
            needed = N - eigvals.size(0)
            eigvals = torch.cat([eigvals, torch.zeros(needed)])
            eigvecs = torch.cat([eigvecs, torch.zeros(N, needed)], dim=1)

        # Exponential of eigenvalues
        exp_vals = torch.diag(torch.exp(eigvals))
        try:
            print("success!")
            inv_eigvecs = torch.linalg.inv(eigvecs)
        except RuntimeError:
            print("failed!")
            inv_eigvecs = torch.linalg.pinv(eigvecs)

        K = eigvecs @ exp_vals @ inv_eigvecs

        return K.real

class CosineEncoding(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def compute_pe(self, graph, epsilon = 1e-8, r_max = 1e4):

        adj = to_dense_adj(graph.edge_index).squeeze(0)

        #get e-vects and e-vals
        eigvals, eigvecs = torch.linalg.eigh(adj)
        eigvals = eigvals.real
        eigvals*=2/max(eigvals)
        eigvecs = eigvecs.real

        #cosine values, capped so that the subsequent fraction calculation
        #doesn't return too big of a value
        cos_vals = torch.cos(torch.pi/4 * eigvals)
        cos_vals = torch.clamp(cos_vals, min=epsilon, max=-epsilon)

        r_vals = cos_vals
        r_vals = torch.clamp(r_vals, -r_max, r_max)

        R = torch.diag(r_vals)
        K = eigvecs @ R @ eigvecs.T

        return K.real

class ClusteringCoefficientEncoding(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def compute_pe(self, graph):
        G = nx.from_edgelist(graph.edge_index.t().tolist())
        num_nodes = graph.num_nodes
        spl_array = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            count = 0
            for neighbor1 in G.neighbors(i):
                for neighbor2 in G.neighbors(i):
                    if G.has_edge(neighbor1, neighbor2):
                        count += 1
            count /= 2

            k = G.degree(i)

            if k <= 1:
                spl_array[i, i] = 2*count
            else:
                spl_array[i, i] = 2*count / (k * (k-1))

        return torch.from_numpy(spl_array)

class BetweennessCentrality(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def compute_pe(self, graph):
        G = nx.from_edgelist(graph.edge_index.t().tolist())
        num_nodes = graph.num_nodes
        spl_array = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):

            sum = 0
            for s in range(num_nodes):
                if s != i:
                    for t in range(num_nodes):
                        if t != i:
                            if not nx.has_path(G, s, t):
                                continue
                            shortest_paths_s_t = nx.all_shortest_paths(G, source=s, target=t)
                            num_shortest_paths = 0
                            num_shortest_paths_intermed = 0
                            for path in shortest_paths_s_t:
                                num_shortest_paths += 1
                                if i in path:
                                    num_shortest_paths_intermed += 1
                            sum += num_shortest_paths_intermed/num_shortest_paths

            spl_array[i, i] = sum

        return torch.from_numpy(spl_array)

class NoEncoding(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def compute_pe(self, graph):
        arr = np.ones((graph.num_nodes, graph.num_nodes))
        return torch.from_numpy(arr).float()



POSENCODINGS = {
    "diffusion": DiffusionEncoding,
    "pstep": PStepRWEncoding,
    "adj": AdjEncoding,
    "shortest_path": ShortestPathEncoding,
    "graphlet": GraphletEncoding,
    "coloring": ColoringEncoding,
    "estrada": EstradaEncoding,
    "cosine": CosineEncoding,
    "clustering_coefficient": ClusteringCoefficientEncoding,
    "betweenness_centrality": BetweennessCentrality,
    "no": NoEncoding
}
