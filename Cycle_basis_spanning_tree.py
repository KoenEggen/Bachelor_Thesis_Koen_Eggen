# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 11:48:40 2025

@author: koene
"""

import networkx as nx

# Function to construct a cycle basis using the Spanning Tree algorithm
def cycle_basis_spanning_tree(G):
    T = nx.minimum_spanning_tree(G)
    fundamental_edges = set(G.edges()) - set(T.edges())
    cycle_basis = []
    
    for u, v in fundamental_edges:
        path = nx.shortest_path(T, source=u, target=v)
        cycle = path + [u]
        cycle_basis.append(cycle)
    
    return cycle_basis
