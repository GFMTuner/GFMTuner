
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph, to_networkx
from typing import Dict, Any, List, Tuple
import networkx as nx


def sample_subgraph(data, method: str = 'random', k: int = 100) -> Data:
    num_nodes = data.num_nodes
    k = min(k, num_nodes)
    
    if method == 'random':
        sampled_nodes = torch.randperm(num_nodes)[:k]
    elif method == 'bfs':
        start_node = torch.randint(0, num_nodes, (1,)).item()
        G = to_networkx(data, to_undirected=True)
        bfs_nodes = list(nx.bfs_tree(G, start_node).nodes())[:k]
        sampled_nodes = torch.tensor(bfs_nodes)
    elif method == 'dfs':
        start_node = torch.randint(0, num_nodes, (1,)).item()
        G = to_networkx(data, to_undirected=True)
        dfs_nodes = list(nx.dfs_tree(G, start_node).nodes())[:k]
        sampled_nodes = torch.tensor(dfs_nodes)
    elif method == 'pagerank':
        G = to_networkx(data, to_undirected=True)
        pr = nx.pagerank(G)
        top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k]
        sampled_nodes = torch.tensor([n for n, _ in top_nodes])
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    edge_index, _ = subgraph(sampled_nodes, data.edge_index, relabel_nodes=True)
    
    sub_data = Data(edge_index=edge_index)
    if hasattr(data, 'x') and data.x is not None:
        sub_data.x = data.x[sampled_nodes]
    if hasattr(data, 'y') and data.y is not None:
        sub_data.y = data.y
    
    return sub_data


def find_motifs(data, size: int = 3) -> Dict[str, Any]:
    G = to_networkx(data, to_undirected=True)
    
    if size == 3:
        triangles = sum(nx.triangles(G).values()) // 3
        return {
            'triangles': triangles,
            'triangle_density': triangles / max(1, G.number_of_nodes())
        }
    elif size == 4:
        squares = sum(1 for _ in nx.enumerate_all_cliques(G) if len(list(_)) == 4)
        return {
            'squares': squares,
            'square_density': squares / max(1, G.number_of_nodes())
        }
    else:
        cliques = list(nx.find_cliques(G))
        motifs_of_size = [c for c in cliques if len(c) == size]
        return {
            f'motifs_size_{size}': len(motifs_of_size),
            f'motif_density_size_{size}': len(motifs_of_size) / max(1, G.number_of_nodes())
        }


def extract_ego_graphs(data, nodes: List[int], radius: int = 2) -> List[Data]:
    ego_graphs = []
    
    for center_node in nodes:
        subset, edge_index, mapping, _ = k_hop_subgraph(
            center_node, 
            radius, 
            data.edge_index, 
            relabel_nodes=True
        )
        
        ego_data = Data(edge_index=edge_index)
        if hasattr(data, 'x') and data.x is not None:
            ego_data.x = data.x[subset]
        if hasattr(data, 'y') and data.y is not None:
            ego_data.y = data.y
        
        ego_graphs.append(ego_data)
    
    return ego_graphs


def compute_ego_statistics(ego_graphs: List[Data]) -> Dict[str, Any]:
    if not ego_graphs:
        return {}
    
    sizes = [g.x.shape[0] if hasattr(g, 'x') else 0 for g in ego_graphs]
    edge_counts = [g.edge_index.shape[1] for g in ego_graphs]
    
    return {
        'num_ego_graphs': len(ego_graphs),
        'avg_ego_size': np.mean(sizes),
        'std_ego_size': np.std(sizes),
        'avg_ego_edges': np.mean(edge_counts),
        'min_ego_size': min(sizes) if sizes else 0,
        'max_ego_size': max(sizes) if sizes else 0
    }
