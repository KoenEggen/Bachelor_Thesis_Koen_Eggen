# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:15:54 2025

@author: koene
"""

import matplotlib.pyplot as plt
import networkx as nx
import random
import math

# Function to find short cycles
def find_short_cycle(G, max_length, node):
    short_cycle = None

    # Get all edges connected to the specified node
    edges = list(G.edges(node))

    # Iterate over each edge connected to the specified node
    for edge in edges:
        u, v = edge

        if u != node:
            u, v = v, u  # Swap if necessary so that u is the specified node
        
        # Temporarily remove the edge to find a shortest path between u and v
        G.remove_edge(u, v)

        try:
            # Find the shortest path between u and v in the graph without the edge
            path = nx.shortest_path(G, source=u, target=v)

            cycle = path + [u]

            short_cycle = cycle            
            G.add_edge(u, v)

            # If the cycle length is smaller than max_length, return the cycle immediately
            if len(short_cycle) < max_length:
                return short_cycle

        except nx.NetworkXNoPath:
            G.add_edge(u, v)
            # If no path exists between u and v, we simply skip this edge
            pass

    return short_cycle


# Function to visualize a graph
def visualize_graph(G, title):
    plt.figure(figsize=(8, 6))
    
    pos = nx.spring_layout(G)
    
    # If the graph is a MultiGraph or MultiDiGraph, handle multi-edges separately
    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue")
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

        for (u, v, data) in G.edges(data=True):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='gray')

    else:
        nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=12, font_weight="bold", edge_color="gray")
    
    # Add title
    plt.title(title)
    plt.show()
    

# Function to visualize a given cycle basis
def visualize_cycle_basis(cycle_basis):
    for idx, cycle in enumerate(cycle_basis):
        # Create a new graph for each cycle
        G_cycle = nx.Graph()
        
        # Add the edges for the current cycle to the graph
        for edge in cycle:
            G_cycle.add_edge(edge[0], edge[1])
        
        # Draw the graph for this cycle
        pos = nx.spring_layout(G_cycle)
        plt.figure(figsize=(6, 6))
        
        # Draw the cycle graph
        nx.draw(G_cycle, pos, with_labels=True, node_color="skyblue", node_size=1000, font_size=12, font_weight='bold', edge_color="gray")
        plt.title(f"Cycle {idx + 1}")
        plt.show()


# Function implementing the algoritm of Freedman-Hastings
def Freedman_Hastings(G):
    
    if not isinstance(G, nx.MultiGraph):
        G = nx.MultiGraph(G)
    
    nodes = list(G.nodes())
    n = len(nodes)
    CB = []  # To store the cycle basis
    
    removed_nodes = []  # To track removed edges (like (x, y) or (node, y))
    removed_edges = {}
    added_edges = []    # To track added edges (like (x, node) and (node, y))
    added_cycles = []
    
    # Step 1: Remove self-edges and replace multi-edges with a single edge
    def process_graph(G):

        self_loops = [(node, node) for node in G.nodes() if G.has_edge(node, node)]
        G.remove_edges_from(self_loops)

        # Replace multi-edges with single edges
        for u, v, data in list(G.edges(data=True)):
            if G.number_of_edges(u, v) > 1:
                # Replace multi-edges with a single edge
                G.remove_edges_from(list(G.edges(u, v))[1:])
        return G
    
    # Apply the processing step to remove self-loops and replace multi-edges
    G = process_graph(G)
    
    # Visualize the processed graph
    # visualize_graph(G, title="Processed Graph")
    
    def recursive_A(G):
        # print("We start with the recursive function")
        
        if not G.edges:
            # print("No more edges, base case reached.")
            return []  # Base case, stop recursion if no edges left
        
        node_degrees = dict(G.degree())
        
        # Step 1: Remove vertices with degree 1
        degree_one_nodes = [node for node, deg in node_degrees.items() if deg == 1]
        if degree_one_nodes:
            # print("We start with step 1")
            G.remove_nodes_from(degree_one_nodes)
            
            # visualize_graph(G, title="After removing degree-1 nodes")
            return recursive_A(G)  # Continue recursion after removing degree-1 nodes

        
        # Step 2: Handle vertices with degree 2
        degree_two_nodes = [node for node, deg in node_degrees.items() if deg == 2]
        if degree_two_nodes:
            # print("We start with step 2")
            for node in degree_two_nodes:
                # print("We consider node - deg:", str(node) + " - " + str(node_degrees[node]))
                
                # print("Node has degree 2")
                neighbors = list(G.neighbors(node))
                # print("Node has neighbors", neighbors)

                # Step 2A: Handle self-loops
                if G.has_edge(node, node):
                    # print("We start with step 2A")
                    G.remove_edge(node, node)
                    CB.append([node])
                    
                    # visualize_graph(G, title="After removing self-loops")
                    return recursive_A(G)
                
                # Step 2B: Handle degree-2 node with exactly two neighbors
                elif len(neighbors) == 2:
                    
                    # print("We start with step 2B")
                    
                    x, y = min(neighbors), max(neighbors)
                    
                    # Replace (x, y) with (x, node) and (node, y)
                    G.add_edge(x, y)
                    G.remove_node(node)
                    # print("We have removed the node and added edge:", (x, y))
                    
                    # Track the removed edges
                    removed_nodes.append(node)  # Store edges as a tuple of two edges
                    
                    removed_edges[(x, y), node] = [(x, node), (node, y)]
                    
                    # print("We have added these edges to removed_edges:", [(x, node), (node, y)])
                    # print("Check:", removed_edges)
                    added_edges.append((x, y))  # Store the added edge with the node as key

                    # visualize_graph(G, title="After handling degree-2 nodes with 2 neighbors")
                    
                    return recursive_A(G)
                
                # Handle nodes with exactly one neighbor while having degree 2
                elif len(neighbors) == 1:
                    # print("Node has only 1 neighbor")
                    
                    x = neighbors[0]
                    
                    # For degree 1 nodes, simply add to the cycle basis (or remove the node)
                    cycle = [(node, x), (x, node)]
                    # print()
                    # print("The cycle looks like:", cycle)
                    CB.append(cycle)
                    # print("The cycle now looks like:", cycle)
                    # print()
                    added_cycles.append(cycle)
                    
                    G.remove_node(node)  # Remove the node
                    
                    # visualize_graph(G, title="After handling degree-2 nodes with 1 neighbor")
                    return recursive_A(G)
        
        # Step 3: Handle nodes with degree greater than 2
        higher_degree_nodes = [node for node, deg in node_degrees.items() if deg > 2]
        
        k = 200
        max_length = int(k * math.log(n))
        
        if higher_degree_nodes:
            for node in higher_degree_nodes:
                # print("We consider the node:", node)
                # Step 3: Try to find the shortest simple cycle
                short_cycle = find_short_cycle(G, max_length, node)
                # print("We have found the short cycle:", short_cycle)
                
                if short_cycle:
                    # Correctly form the cycle edges without duplication
                    cycle_edges = [(short_cycle[i], short_cycle[i + 1]) for i in range(len(short_cycle) - 1)]
                    # print("This cycle has edges:", cycle_edges)
                    
                    CB.append(cycle_edges)
                    # Select a random edge to remove from the cycle
                    edge_to_remove = random.choice(cycle_edges)
                    # print("We have removed edge:", edge_to_remove)
                    
                    # Directly remove the edge from the graph (no check for existence needed)
                    G.remove_edge(*edge_to_remove)
                    
                    # visualize_graph(G, title="After handling degree-3 nodes")
                    
                    return recursive_A(G)
        
        # Stop recursion
        return []
    
    # Start the recursion
    recursive_A(G)

    # After completing the recursion, replace the edges added in step 2B with the edges removed there    
    temporary_removed_edges = removed_edges.copy()
    temporary_removed_nodes = removed_nodes.copy()
    
    modified_cycles = []
    
    edge_to_consider = False
    
    if len(set(map(tuple, CB))) == 1:
        
        # Current cycle is the first one in CB
        current_cycle = CB[0]
        
        # Edge to consider is the first edge of the current cycle
        edge_to_consider = current_cycle[1]
        
        # Extract the nodes to consider from the edge
        node_to_consider_1, node_to_consider_2 = edge_to_consider

    for cycle in CB:
        
        if cycle in added_cycles:
            
            modified = True  # Start with 'True' to enter the while loop
    
            # While loop will keep checking the cycle as long as it's modified
            while modified:
                modified = False
    
                # Iterate over each edge in the cycle
                for index, edge in enumerate(cycle):
                    # print("Edge to replace:", edge)
    
                    # Iterate over nodes in temporary_removed_nodes
                    for node in temporary_removed_nodes:
                        # print("We consider node:", node)
    
                        if (edge, node) in list(temporary_removed_edges.keys()):
                            # print("The key exists")
    
                            # Get the removed edges
                            edge_to_add_1 = temporary_removed_edges[(edge, node)][0]
                            edge_to_add_2 = temporary_removed_edges[(edge, node)][1]
    
                            # print("The edges to add are:", str(edge_to_add_1) + " and " + str(edge_to_add_2))
    
                            # Replace the added edge with the removed edges
                            cycle.remove(edge)
    
                            # Insert the removed edges
                            cycle.insert(index, edge_to_add_1)  # Add the first removed edge
                            cycle.insert(index + 1, edge_to_add_2)  # Add the second removed edge
    
                            # print("The modified cycle is:", cycle)
    
                            # Delete the node and edge from temporary lists
                            del temporary_removed_edges[(edge, node)]

                            temporary_removed_nodes.remove(node)
    
                            modified_cycles.append(cycle)

                            # Mark that the cycle was modified
                            modified = True  # Set modified to True to continue checking this cycle
                            break
        
                    # If the cycle was modified, break the edge loop and check again
                    if modified:
                        break

    # Make sure that the cycles are placed correctly in the cycle basis if there was at some point a multi-edge between two vertices
    if edge_to_consider:

        lower_cycles = {}
        higher_cycles = {}
        
        # Populate the lower_cycles and higher_cycles dictionaries
        for cycle in CB:

            if cycle in modified_cycles and edge_to_consider in cycle:
                
                temp_cycle_nodes = [node for node, _ in cycle][:]
    
                temp_cycle_nodes.remove(node_to_consider_1)            
                temp_cycle_nodes.remove(node_to_consider_2)

                lower = False
                higher = False
                for node in temp_cycle_nodes:
                    if node < node_to_consider_2:
                        lower = True
                        break
                    elif node > node_to_consider_2:
                        higher = True
                        break
        
                # Convert cycle to tuple for the dictionary key
                cycle_key = tuple(cycle)
        
                if lower:
                    lower_cycles[cycle_key] = cycle
                elif higher:
                    higher_cycles[cycle_key] = cycle    
        
        if len(lower_cycles.values()) > 0 and len(higher_cycles.values()) > 0:
            
            lower_cycles = dict(reversed(list(lower_cycles.items())))
                        
            # Function to calculate the symmetric difference
            def calculate_symmetric_difference(list1, list2):
                return list(set(list1) ^ set(list2))
            
            # Function to update the dictionary based on symmetric difference
            def update_cycles(cycles_dict):

                keys = list(cycles_dict.keys())
                for i in range(len(keys) - 1):
                    key_i = keys[i]
                    key_next = keys[i + 1]
            
                    # Get the values for the i-th and i+1-th keys
                    value_next = cycles_dict[key_next]
            
                    # Calculate the symmetric difference between key_i and value_next
                    symmetric_diff = calculate_symmetric_difference(list(key_i), value_next)
            
                    # Replace the value for the i-th key with the symmetric difference
                    cycles_dict[key_i] = symmetric_diff
            
            # Update both lower_cycles and higher_cycles
            update_cycles(lower_cycles)
            update_cycles(higher_cycles)
                
            # Update the cycle basis based on the changes in lower_cycles and higher_cycles
            def update_CB_with_cycles(cycles_dict):
                for i, cycle in enumerate(CB):
                    cycle_tuple = tuple(cycle)
                    if cycle_tuple in cycles_dict.keys():
                        # Replace the current cycle with the two cycles from cycles_dict
                        CB[i] = cycles_dict[cycle_tuple]
            
            # Apply updates to both lower_cycles and higher_cycles in CB
            update_CB_with_cycles(lower_cycles)
            update_CB_with_cycles(higher_cycles)
            
            # Step 3: Handle removal and addition of the first and last cycle
            def handle_first_and_last_cycle(cycles_dict):
                # Get the first and last cycle
                first_cycle = list(cycles_dict.keys())[0]
                last_key = list(cycles_dict.keys())[-1]
            
                last_cycle = cycles_dict[last_key]
            
                # Remove the cycle from CB if it exists
                if last_cycle in CB:
                    CB.remove(last_cycle)
            
                # If the first cycle is not in CB, append it
                if first_cycle not in CB:
                    CB.append(first_cycle)
            
            # Handle both lower_cycles and higher_cycles for first and last cycles
            handle_first_and_last_cycle(lower_cycles)
            handle_first_and_last_cycle(higher_cycles)
    
    # visualize_cycle_basis(CB)
    
    # Make sure that the cycles are displayed correctly in the cycle basis
    final_CB = []
    for cycle in CB:
        cycle_nodes = [node for node, _ in cycle]
        if cycle_nodes[0] != cycle_nodes[-1]:
            cycle_nodes.append(cycle_nodes[0])  # Close the cycle by repeating the first node at the end

        final_CB.append(cycle_nodes)        
    
    return final_CB


# Code to visualize an example of the algorithm of Freedman-Hastings
# Create a cycle graph of length 8
# G = nx.cycle_graph(8)

# # Create the other type of graph
# G = nx.Graph()

# # Manually add nodes and edges based on the given structure
# edges = [
#     (1, 3), (1, 10), (2, 3), (2, 10), (3, 11), (3, 10), (3, 4), (4, 5), (5, 6), (6, 7),
#     (7, 8), (8, 9), (9, 10), (10, 11)
# ]

# # Add edges to the graph
# G.add_edges_from(edges)

# # Define the positions of the nodes manually to match the structure of the graph
# pos = {
#     1: (0, 0),
#     2: (1, 0),
#     3: (2, 1),
#     4: (3, 2),
#     5: (4, 2),
#     6: (5, 1),
#     7: (5, -1),
#     8: (4, -2),
#     9: (3, -2),
#     10: (2, -1),
#     11: (3, 0),
# }



# # Visualize the graph with the custom node positions
# plt.figure(figsize=(8, 6))
# nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=12, font_weight="bold", edge_color="black")

# # Show the plot
# plt.title("Custom Graph Based on Image")
# plt.show()


# # Visualize initial graph
# visualize_graph(G, title="Initial Graph")

# # Run the Freedman-Hastings algorithm
# cycle_basis = Freedman_Hastings(G.copy())









