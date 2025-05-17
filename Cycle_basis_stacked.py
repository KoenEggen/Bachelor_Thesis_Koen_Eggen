# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 12:28:46 2025

@author: koene
"""

import networkx as nx

from Sewing_methods import sew_cycles_straight
from Sewing_methods import sew_cycles_triangular
from Sewing_methods import sew_cycles_skew
from Sewing_methods import sew_cycles_straight_6

import matplotlib.pyplot as plt
import numpy as np
from Freedman_Hastings import Freedman_Hastings
# from Vertex_coloring import Vertex_coloring_alg
# from Creating_graphs import create_bipartite_expander


# Function to implement the Stacked graph algorithm
def Stacked(G, L, cycle_basis=None, cycles_to_sew=None):
    full_cycle_basis = set()
    node_mapping = {}
    next_node_id = 0
    seen_cycles = set()
    vertical_cycles = []
    
    G_prime = nx.Graph()
    
    # Sew multiple cycles per layer if cycles_to_sew is not None
    if cycles_to_sew is not None:
        
        layer_keys = list(cycles_to_sew.keys())
        num_layers = len(layer_keys)
        
        # Step 1: Node mapping for all layers
        for layer in layer_keys:
            for node in G.nodes():
                if (layer, node) not in node_mapping:
                    node_mapping[(layer, node)] = next_node_id
                    G_prime.add_node(next_node_id)
                    next_node_id += 1
        
        # Step 2: Sew all cycles per layer
        for layer in layer_keys:

            # Add edges between nodes that are part of the same layer's cycle
            for u, v in G.edges():
                u_map = node_mapping[(layer, u)]
                v_map = node_mapping[(layer, v)]

                if not G_prime.has_edge(u_map, v_map):
                    G_prime.add_edge(u_map, v_map)

            # Now sew the cycles that belong to the current layer
            for cycle in cycles_to_sew[layer]:

                sewn_cycles, _, G_prime = sew_cycles_straight(cycle, G_prime)
                # sewn_cycles, _, G_prime = sew_cycles_triangular(cycle, G_prime)
                # sewn_cycles, _, G_prime = sew_cycles_skew(cycle, G_prime)
                # sewn_cycles, _, G_prime = sew_cycles_straight_6(cycle, G_prime)

                for sewn_cycle in sewn_cycles:                
                    sewn_cycle_nodes = [node for node, _ in sewn_cycle]
                    
                    mapped_cycle = tuple(node_mapping[(layer, node)] for node in sewn_cycle_nodes)
                    full_cycle_basis.add(mapped_cycle)
        
        # Step 3: Inter-layer connections and vertical cycles
        for i in range(len(layer_keys) - 1):
            layer = layer_keys[i]
            next_layer = layer_keys[i + 1]
            
            for node in G.nodes():

                u_map = node_mapping[(layer, node)]
                v_map = node_mapping[(next_layer, node)]
                if not G_prime.has_edge(u_map, v_map):
                    G_prime.add_edge(u_map, v_map)
            
            for u, v in G.edges():
                vertical_cycle = [
                    node_mapping[(layer, u)],
                    node_mapping[(next_layer, u)],
                    node_mapping[(next_layer, v)],
                    node_mapping[(layer, v)]
                ]
                
                # Create edges and sort them to ensure consistency
                edges_in_cycle = [
                    tuple(sorted([vertical_cycle[i], vertical_cycle[(i+1) % len(vertical_cycle)]]))
                    for i in range(len(vertical_cycle))
                ]
                
                # Sort the edges themselves to avoid inconsistent ordering
                check_cycle = tuple(sorted(edges_in_cycle))
                
                # Check if this cycle has already been added
                if check_cycle not in seen_cycles:
                    # vertical_cycles.append(tuple(vertical_cycle))
                    seen_cycles.add(check_cycle)
                    full_cycle_basis.add(tuple(vertical_cycle))
                
        # for cycle in vertical_cycles:                
        #     sewn_vertical_cycles, _, G_prime = sew_cycles_triangular(cycle, G_prime)
            
        #     for sewn_vertical_cycle in sewn_vertical_cycles:
        #         sewn_vertical_cycle_nodes = [node for node, _ in sewn_vertical_cycle]
        #         full_cycle_basis.add(tuple(sewn_vertical_cycle_nodes))
        
    # If cycle_basis is provided sew one cycle per layer
    elif cycle_basis is not None:
        
        num_layers = len(cycle_basis)
        
        for layer in range(num_layers):
            for node in G.nodes():
                if (layer, node) not in node_mapping:
                    node_mapping[(layer, node)] = next_node_id
                    G_prime.add_node(next_node_id)
                    next_node_id += 1
        
        for layer, cycle in enumerate(cycle_basis):

            for u, v in G.edges():
                u_map = node_mapping[(layer, u)]
                v_map = node_mapping[(layer, v)]

                if not G_prime.has_edge(u_map, v_map):
                    G_prime.add_edge(u_map, v_map)
        
        for layer, cycle in enumerate(cycle_basis):
            
            sewn_cycles, _, G_prime = sew_cycles_straight(cycle, G_prime)
            # sewn_cycles, _, G_prime = sew_cycles_triangular(cycle, G_prime)
            # sewn_cycles, _, G_prime = sew_cycles_skew(cycle, G_prime)
            # sewn_cycles, _, G_prime = sew_cycles_straight_6(cycle, G_prime)

            for sewn_cycle in sewn_cycles:
                sewn_cycle_nodes = [node for node, _ in sewn_cycle]
                
                # Map the cycle nodes to the new nodes in G_prime
                mapped_cycle = tuple(node_mapping[(layer, node)] for node in sewn_cycle_nodes)                
                full_cycle_basis.add(mapped_cycle)
        
        # After finishing all cycles, handle inter-layer connections
        for layer in range(num_layers - 1):
            for node in G.nodes():

                u_map = node_mapping[(layer, node)]
                v_map = node_mapping[(layer + 1, node)]

                if not G_prime.has_edge(u_map, v_map):
                    G_prime.add_edge(u_map, v_map)
        
            for u, v in G.edges():
                vertical_cycle = [
                    node_mapping[(layer, u)],
                    node_mapping[(layer + 1, u)],
                    node_mapping[(layer + 1, v)],
                    node_mapping[(layer, v)]
                ]
                
                # Create edges and sort them to ensure consistency
                edges_in_cycle = [
                    tuple(sorted([vertical_cycle[i], vertical_cycle[(i+1) % len(vertical_cycle)]]))
                    for i in range(len(vertical_cycle))
                ]
                
                # Sort the edges themselves to avoid inconsistent ordering
                check_cycle = tuple(edges_in_cycle)
                # print("Original vertical cycle:", vertical_cycle)
                # print("Check cycle:", check_cycle)
                
                # Check if this cycle has already been added
                if check_cycle not in seen_cycles:
                    # vertical_cycles.append(tuple(vertical_cycle))
                    seen_cycles.add(check_cycle)
                    full_cycle_basis.add(tuple(vertical_cycle))
                
        # for cycle in vertical_cycles:                
        #     sewn_vertical_cycles, _, G_prime = sew_cycles_triangular(cycle, G_prime)
            
        #     for sewn_vertical_cycle in sewn_vertical_cycles:
        #         sewn_vertical_cycle_nodes = [node for node, _ in sewn_vertical_cycle]
        #         full_cycle_basis.add(tuple(sewn_vertical_cycle_nodes))
    
    else:
        raise ValueError("Either cycle_basis or cycles_to_sew must be provided.")
    
    expected_size = L**2
    actual_size = len(G_prime.edges())

    return expected_size, actual_size, G_prime, [list(cycle) for cycle in full_cycle_basis], num_layers




# def visualize_stacked_graph(G_prime, full_cycle_basis, num_layers, cycles_to_sew=None):
#     # Set up the 3D plot
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Set axis labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
    
#     # Define the color map for layers
#     color_map = plt.cm.get_cmap('tab20', num_layers)

#     # Generate 3D coordinates for each node
#     pos = {}
#     for i, node in enumerate(G_prime.nodes()):
#         # Distribute nodes in layers across the Z-axis
#         layer = i % num_layers
#         angle = 2 * np.pi * (i // num_layers) / num_layers  # Circular distribution within layers
#         radius = 10  # radius for node placement
#         z_pos = layer * 10  # Each layer is placed along the Z-axis with a step of 10
#         x_pos = radius * np.cos(angle)
#         y_pos = radius * np.sin(angle)
        
#         pos[node] = (x_pos, y_pos, z_pos)
        
#     # Plot nodes in 3D space
#     for node, (x, y, z) in pos.items():
#         layer = node % num_layers  # Determine which layer the node belongs to
#         ax.scatter(x, y, z, color=color_map(layer), s=50, label=f'Layer {layer}' if node == 0 else "")

#     # Plot edges between nodes
#     for u, v in G_prime.edges():
#         x_u, y_u, z_u = pos[u]
#         x_v, y_v, z_v = pos[v]
#         ax.plot([x_u, x_v], [y_u, y_v], [z_u, z_v], color='black', alpha=0.5)

#     # Add cycle edges in each layer, with different colors for each cycle
#     if cycles_to_sew is not None:
#         # If cycles_to_sew is provided, plot cycles for each color (layer)
#         for layer in range(num_layers):
#             cycle_color = color_map(layer)  # Color corresponding to this layer
#             for cycle in cycles_to_sew[layer]:
#                 # Draw edges for each cycle in this layer
#                 for i in range(len(cycle)):
#                     u, v = cycle[i], cycle[(i + 1) % len(cycle)]
#                     x_u, y_u, z_u = pos[u]
#                     x_v, y_v, z_v = pos[v]
#                     ax.plot([x_u, x_v], [y_u, y_v], [z_u, z_v], color=cycle_color, linewidth=2)

#     # Show legend for the first node in each layer
#     ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Layers", frameon=False)

#     # Display the 3D plot
#     plt.tight_layout()
#     plt.show()


# # Create a simple graph (cycle of 6 nodes)
# L = 4
# d = 4
# beta = d/4

# G = create_bipartite_expander(L, d, beta)

# # Define a cycle_basis (for this example, just the graph itself)
# cycle_basis = Freedman_Hastings(G.copy())

# # Use the Vertex coloring algorithm
# expected_edges_VC, actual_edges_VC, G_VC, CB_VC, tot_layers_VC, cycles_to_sew = Vertex_coloring_alg(G.copy(), L, cycle_basis)

# # # Call the Stacked function
# # expected_size, actual_size, G_prime, full_cycle_basis, num_layers = Stacked(G, L, cycle_basis = None, cycle_to_sew = cycles_to_sew)

# visualize_stacked_graph(G_VC, CB_VC, tot_layers_VC, cycles_to_sew)

# # Output results
# print("Actual size:", actual_edges_VC)
# print("Length cycle basis:", len(cycle_basis))
# print("Number of layers:", tot_layers_VC)
# # print("Full cycle basis:", CB_VC)



