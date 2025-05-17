# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:13:02 2025

@author: koene
"""

import networkx as nx
from itertools import product
import random
import matplotlib.pyplot as plt
import os


# Function for creating the graph for the sparse SYK Hamiltonian
def create_graph_SYK(L, p):

    G = nx.Graph()
    G.add_nodes_from(range(L))

    terms = []

    for term in product(range(L), repeat=4):
        if random.random() < p:
            terms.append(term)
    
    for term in terms:
        j, k, l, m = term
        if j != k:
            G.add_edge(j, k)
        if l != m:
            G.add_edge(l, m)
    
    return G


# Function to create the Erdos-Renyi graph
def create_Erdos_Renyi_p(n, p):
    nodes = list(range(n))
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G.add_edge(i, j)

    return G


# Visualize a realization of a Erdos-Renyi graph
# # Parameters
# n = 20
# p = 10 * 1/n

# # Create the graph
# G = create_Erdos_Renyi_p(n, p)

# # Directory to save the figure
# save_dir = r"C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Figuren"
# os.makedirs(save_dir, exist_ok=True)
# file_path = os.path.join(save_dir, "cover.eps")

# # Generate layout and scale to elliptical shape (vertical ellipse)
# pos = nx.circular_layout(G)
# for node in pos:
#     x, y = pos[node]
#     pos[node] = (x * 0.6, y * 1.2)  # Compress horizontally, stretch vertically

# # Visualization
# plt.figure(figsize=(8, 8))
# nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=200)
# nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
# # nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# plt.axis('off')
# plt.tight_layout()

# # Save the figure to the specified directory
# plt.savefig(file_path, format='eps')

# # Display the figure
# plt.show()





