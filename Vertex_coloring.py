# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 11:17:18 2025

@author: koene
"""

import networkx as nx
from Cycle_basis_stacked import Stacked

from Freedman_Hastings import Freedman_Hastings
import matplotlib.pyplot as plt
import random
import itertools


# Function to get the vertices of to a certain cycle in cycle_basis
def get_vertices_from_cycle(cycle_edges):

    vertices = set()
    for edge in cycle_edges:
        vertices.add(edge[0])
        vertices.add(edge[1])

    return vertices


# Function to find the vertex in G_prime corresponding to a given cycle in cycle_basis
def find_vertex_by_cycle(graph_dict, given_cycle):

    for vertex, cycle in graph_dict.items():
        if sorted(cycle) == sorted(given_cycle):
            return vertex

    return None  # In case no match is found


# Function that implements the Vertex Coloring algorithm
def Vertex_coloring_alg(G, L, cycle_basis):
    
    # Create a graph G_prime with the vertices corresponding to cycles in cycle_basis    
    G_prime = nx.Graph()
    num_cycles = len(cycle_basis)
    G_prime.add_nodes_from(range(num_cycles))
    
    # Keep track of which vertex in G_prime belongs to which cycle in cycle_basis
    cycle_to_vertex = {}
    for i in range(num_cycles):
        cycle_to_vertex[i] = cycle_basis[i]
    
    # Add edges to G_prime between vertices i and j if their corresponding cycles share at least one vertex
    for i in range(num_cycles):
        for j in range(num_cycles):
            if i != j:
                # Get the set of vertices for both cycles
                vertices_i = set(cycle_to_vertex[i])
                vertices_j = set(cycle_to_vertex[j])
                
                # Check if there is any overlap between the two cycles and draw an edge between vertices if there is some overlap
                if len(vertices_i & vertices_j) > 0: # 0 if no overlap is allowed, 1/2/etc. if some overlap is allowed
                    if not G_prime.has_edge(i, j):
                        G_prime.add_edge(i, j)
    
    
    # Create a partition of non-overlapping cycles using the Greedy Algorithm    
    partition = {}
    cycles = cycle_basis[:]
    
    # Repeat until we have dealt with each cycle in cycle_basis
    i = 0
    while len(cycles) > 0:
        partition[i] = []
        
        # Iterate backwards over all cycles in R
        for current_cycle in cycles[::-1]:
            current_vertex = find_vertex_by_cycle(cycle_to_vertex, current_cycle)
            
            mark = True
            if partition[i]:
                for other_cycle in partition[i]:
                    other_vertex = find_vertex_by_cycle(cycle_to_vertex, other_cycle)
                    if G_prime.has_edge(current_vertex, other_vertex):
                        mark = False
                
                # If the current_cycle does not overlap with any of the other cycles in Ri, add it to Ri
                if mark:
                    partition[i].append(current_cycle)
            
            # If Ri is empty, add current_cycle to it
            else:
                partition[i].append(current_cycle)
        
        # Remove all cycles in Ri from R and repeat
        for cycle_to_remove in partition[i]:
            cycles.remove(cycle_to_remove)
        
        i += 1
    
    # Create a dictionary cycles_to_sew similar to partition, but which is fitted to give as an argument to the Stacked function
    cycles_to_sew = {}
    
    for color, cycles in partition.items():

        cycles_to_sew[color] = []
        for cycle in cycles:
            # cycle_nodes = [node for node, _ in cycle]

            # if cycle_nodes[0] != cycle_nodes[-1]:
            #     cycle_nodes.append(cycle_nodes[0])  # Close the cycle by repeating the first node at the end

            cycles_to_sew[color].append(cycle)
    
    # Call the Stacked function with the original graph G and the cycles_to_sew dictionary
    expected_size, actual_size, G_VC, CB_VC, num_layers = Stacked(G, L, cycle_basis = None, cycles_to_sew = cycles_to_sew)
    
    return expected_size, actual_size, G_VC, CB_VC, num_layers, cycles_to_sew





# # Function to visualize the vertex coloring
# def visualize_coloring(G_prime, coloring):
#     # Assign a color to each vertex based on its assigned color in the coloring dictionary
#     color_map = []
#     for node in G_prime.nodes():
#         color = coloring.get(node, 'grey')  # Default to grey if no color is assigned
#         color_map.append(color)

#     # Draw the graph with the colors
#     plt.figure(figsize=(10, 8))
#     pos = nx.spring_layout(G_prime, seed=42)  # Use a layout for positioning nodes
#     nx.draw(G_prime, pos, node_color=color_map, with_labels=True, node_size=500, font_size=12, font_weight='bold')
#     plt.title("Vertex Coloring Visualization")
#     plt.show()

# # Set the L value and run the algorithm
# L = 5  # Example number of nodes
# d = 4
# beta = d / 4

# G = nx.Graph()
# G.add_nodes_from(range(L))

# for i in range(L):
#     for j in range(i + 1, L):
#         if random.random() < 1:  # chance to add an edge
#             G.add_edge(i, j)

# # Get the cycle basis for the graph
# cycle_basis = Freedman_Hastings(G.copy()) #nx.cycle_basis(G)

# print("Graph created:", G.edges())  # Check edges here

# # Run the vertex coloring algorithm and visualize
# expected_size, actual_size, G_VC, CB_VC, num_layers = Vertex_coloring_alg(G, L, cycle_basis)

# print("Edges G_VC:",G_VC.edges())

# # Now, let's visualize the graph G_VC in 3D
# # Generate positions for nodes in 3D space
# pos_3d = {node: [random.random(), random.random(), random.random()] for node in G_VC.nodes()}

# # Create a 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Draw nodes
# x_vals = [pos_3d[node][0] for node in G_VC.nodes()]
# y_vals = [pos_3d[node][1] for node in G_VC.nodes()]
# z_vals = [pos_3d[node][2] for node in G_VC.nodes()]
# ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o', s=100)  # Scatter nodes

# # Draw edges
# for edge in G_VC.edges():
#     x_start, y_start, z_start = pos_3d[edge[0]]
#     x_end, y_end, z_end = pos_3d[edge[1]]
#     ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], color="g", linewidth=1)

# # Set axis labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title(f"3D Visualization of Graph G_VC with {num_layers} Layers")

# plt.show()
