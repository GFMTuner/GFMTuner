import torch
import numpy as np
from torch_geometric.utils import degree, to_networkx
from typing import Dict, Any
import networkx as nx


def get_basic_stats(data) -> Dict[str, Any]:
    G = to_networkx(data, to_undirected=True)
    

    degree_cent = nx.degree_centrality(G)
    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_k]
    

    if len(G) > 1000:
        k_sample = min(100, len(G))
        between_cent = nx.betweenness_centrality(G, k=k_sample)
    else:
        between_cent = nx.betweenness_centrality(G)
    top_between = sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:top_k]
    

    pagerank_cent = nx.pagerank(G)
    top_pagerank = sorted(pagerank_cent.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return {
        'degree_centrality': degree_cent,
        'betweenness_centrality': between_cent,
        'pagerank_centrality': pagerank_cent,
        'top_degree': top_degree,
        'top_betweenness': top_between,
        'top_pagerank': top_pagerank
    }
