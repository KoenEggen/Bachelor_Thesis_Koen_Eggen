# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:55:06 2025

@author: koene
"""

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stack_graphs_3D(G,cycle_basis):
    num_layers = len(cycle_basis) + 1

    stacked_G = nx.Graph()  # Graph containing all layers
    pos = {}  # Store 3D positions

    # Create multiple copies of G, shifting them in z-direction
    z_spacing = 3  # Space layers apart
    for layer in range(num_layers):
        for node in G.nodes():
            new_node = (node, layer)  # Unique node identifier
            if node in G.nodes():
                x, y = G.nodes[node].get("pos", (0, 0))
            else:
                x, y = (0, 0)  # Default position for missing nodes

            pos[new_node] = (x, y, layer * z_spacing)  # Store 3D position
            stacked_G.add_node(new_node)

        # Add intra-layer edges (within each copy of G)
        for edge in G.edges():
            stacked_G.add_edge((edge[0], layer), (edge[1], layer))  

    # Connect corresponding nodes between layers
    for layer in range(num_layers - 1):
        for node in G.nodes():
            stacked_G.add_edge((node, layer), (node, layer + 1))  # Inter-layer edges

    return stacked_G, pos
