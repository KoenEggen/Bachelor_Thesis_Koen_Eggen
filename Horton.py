# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 15:20:21 2025

@author: koene
"""

import networkx as nx
import numpy as np


# Function to check whether a given vector is linearly independent from the existing row vectors in a given matrix
def is_linearly_independent(cycle_matrix, new_cycle_vector):
    augmented_matrix = np.vstack([cycle_matrix, new_cycle_vector]) if cycle_matrix.size else np.array([new_cycle_vector])
    _, independent_cols = np.linalg.qr(augmented_matrix.T, mode='reduced')
    return independent_cols.shape[1] > cycle_matrix.shape[0]


# Function which implements Horton's algorithm
def Horton(G):
    cycles = set()
    
    # For each vertex calculate the shortest path to all other vertices
    shortest_paths = {v: nx.single_source_shortest_path(G, v) for v in G.nodes()}

    # Create the cycles
    for v in G.nodes():
        for (x, y) in G.edges():
            if x == y or v in (x, y):
                continue
            
            if x in shortest_paths[v] and y in shortest_paths[v]:
                path_x = shortest_paths[v][x]
                path_y = shortest_paths[v][y]
                
                cycle = path_x + path_y[::-1] + [x,y]
                cycle_nodes = tuple(sorted(set(cycle)))
                
                cycles.add(cycle_nodes)
    
    # Sort the cycles by length
    sorted_cycles = sorted(cycles, key=len)
    min_cycle_basis = []
    all_edges = set()

    # Add all edges from the cycles to a set    
    for cycle in sorted_cycles:
        edges = {(min(u, v), max(u, v)) for u, v in zip(cycle, cycle[1:])}
        all_edges.update(edges)
    
    # Apply the Greedy algorithm
    num_edges = len(all_edges)
    edge_to_index = {edge: i for i, edge in enumerate(all_edges)}
    cycle_matrix = np.zeros((0, num_edges), dtype=int)
    
    for cycle in sorted_cycles:
        cycle_vector = np.zeros(num_edges, dtype=int)
        
        edge_indices = [edge_to_index[(min(u, v), max(u, v))] for u, v in zip(cycle, cycle[1:]) if (min(u, v), max(u, v)) in edge_to_index]
        cycle_vector[edge_indices] = 1
        
        if is_linearly_independent(cycle_matrix, cycle_vector):
            min_cycle_basis.append(cycle)
            cycle_matrix = np.vstack([cycle_matrix, cycle_vector]) if cycle_matrix.size else np.array([cycle_vector])
    
    return min_cycle_basis

