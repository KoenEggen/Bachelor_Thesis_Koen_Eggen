# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:52:15 2025

@author: koene
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# Function to sew a cycle connecting opposite vertices (straight sewing)
def sew_cycles_straight(cycle, G):
    if cycle[0] == cycle[-1]:
        cycle = cycle[:-1]

    n = len(cycle)

    # If cycle is too short, return its edges and an unchanged graph
    if n <= 4:
        edges = [(cycle[i], cycle[(i + 1) % n]) for i in range(n)]
        return [edges], [], G

    half_size = n // 2
    top_vertices = cycle[1:half_size]
    bottom_vertices = cycle[1 + half_size:][::-1]

    new_edges = [(top_vertices[i], bottom_vertices[i]) for i in range(len(top_vertices))]
    G.add_edges_from(new_edges)

    def create_quad(top, bottom, next_top, next_bottom):
        return [(top, next_top), (next_top, next_bottom), (next_bottom, bottom), (bottom, top)]

    new_cycles = []

    for i, (top, bottom) in enumerate(new_edges):
        if i == 0:
            new_cycles.append([(cycle[0], top), (top, bottom), (bottom, cycle[0])])

            if len(new_edges) == 1:
                if n % 2 == 1:
                    middle1 = cycle[half_size]
                    middle2 = cycle[half_size + 1]
                    new_cycles.append([(top, middle1), (middle1, middle2), (middle2, bottom), (bottom, top)])
            else:
                next_top = top_vertices[i + 1]
                next_bottom = bottom_vertices[i + 1]
                new_cycles.append(create_quad(top, bottom, next_top, next_bottom))

        elif i == len(new_edges) - 1:
            if n % 2 == 0:
                middle = cycle[half_size]
                new_cycles.append([(top, middle), (middle, bottom), (bottom, top)])
            else:
                middle1 = cycle[half_size]
                middle2 = cycle[half_size + 1]
                new_cycles.append([(top, middle1), (middle1, middle2), (middle2, bottom), (bottom, top)])
        else:
            next_top = top_vertices[i + 1]
            next_bottom = bottom_vertices[i + 1]
            new_cycle = create_quad(top, bottom, next_top, next_bottom)
            new_cycles.append(new_cycle)

    return new_cycles, new_edges, G



# Function to implement the traingular sewing
def sew_cycles_triangular(cycle, G):
    if cycle[0] == cycle[-1]:
        cycle = cycle[:-1]

    n = len(cycle)

    # Handle cycles of length 3 or less
    if n <= 3:
        cycle_edges = [(cycle[i], cycle[(i + 1) % n]) for i in range(n)]
        return [cycle_edges], [], G

    # Special case for square (n == 4)
    if n == 4:
        a, b, c, d = cycle  # Assume cycle is [a, b, c, d]
        new_edges = [(b, d)]
        G.add_edge(b, d)

        new_cycles = [
            [(a, b), (b, d), (d, a)],
            [(b, c), (c, d), (d, b)]
        ]
        return new_cycles, new_edges, G

    # General case for n > 4
    half_size = n // 2
    even = n % 2 == 0

    top_vertices = cycle[1:half_size + (0 if even else 1)]
    bottom_vertices = cycle[half_size + 1:][::-1]

    new_edges = []
    new_cycles = []

    for i in range(len(top_vertices) - 1):
        top = top_vertices[i]
        bottom = bottom_vertices[i]

        edge1 = (top, bottom)
        G.add_edge(*edge1)
        new_edges.append(edge1)

        next_bottom = bottom_vertices[i + 1]

        edge2 = (top, next_bottom)
        G.add_edge(*edge2)
        new_edges.append(edge2)

    # Final unmatched pair if needed
    if even and len(top_vertices) == len(bottom_vertices):
        last_edge = (top_vertices[-1], bottom_vertices[-1])
        G.add_edge(*last_edge)
        new_edges.append(last_edge)

    for idx, (top, bottom) in enumerate(new_edges):
        if idx < len(new_edges) - 1:
            next_edge = new_edges[idx + 1][::-1]

            if idx % 2 == 0:
                new_cycles.append([(top, bottom), (bottom, next_edge[0]), next_edge])
            else:
                new_cycles.append([(top, bottom), next_edge, (next_edge[1], top)])

        if idx == 0:
            first_node = cycle[0]
            new_cycles.append([(first_node, top), (top, bottom), (bottom, first_node)])

        elif idx == len(new_edges) - 1:
            middle_node = cycle[half_size]
            new_cycles.append([(top, middle_node), (middle_node, bottom), (bottom, top)])

    return new_cycles, new_edges, G



# Function to implement the skew sewing
def sew_cycles_skew(cycle, G):
    if cycle[0] == cycle[-1]:
        cycle = cycle[:-1]

    n = len(cycle)

    if n <= 4:
        # Simply return the cycle edges and unchanged graph
        cycle_edges = [(cycle[i], cycle[(i + 1) % n]) for i in range(n)]
        return [cycle_edges], [], G

    half_size = n // 2
    is_even = (n % 2 == 0)

    top_vertices = cycle[1:half_size]
    bottom_vertices = cycle[1 + half_size:][::-1]

    new_edges = []
    new_cycles = []

    edge_range = range(half_size - 2) if is_even else range(half_size - 1)
    for i in edge_range:
        top = top_vertices[i]
        bottom = bottom_vertices[i + 1]
        edge = (top, bottom)
        G.add_edge(*edge)
        new_edges.append(edge)

    for idx, (top, bottom) in enumerate(new_edges):
        if idx == 0:
            first_node, last_node = cycle[0], cycle[-1]
            new_cycles.append([(first_node, top), (top, bottom), (bottom, last_node), (last_node, first_node)])

            index_top = top_vertices.index(top)
            index_bottom = bottom_vertices.index(bottom)

            next_top = top_vertices[index_top + 1] if index_top + 1 < len(top_vertices) else None
            next_bottom = bottom_vertices[index_bottom + 1] if index_bottom + 1 < len(bottom_vertices) else None

            if next_top and next_bottom:
                new_cycles.append([(top, next_top), (next_top, next_bottom), (next_bottom, bottom), (bottom, top)])
            else:
                if is_even:
                    half_node = cycle[half_size]
                    if next_top:
                        new_cycles.append([(top, next_top), (next_top, half_node), (half_node, bottom), (bottom, top)])
                    else:
                        new_cycles.append([(top, half_node), (half_node, bottom), (bottom, top)])
                else:
                    half_node_1 = cycle[half_size]
                    if half_size + 1 < n:
                        half_node_2 = cycle[half_size + 1]
                        # Return triangle if possible
                        new_cycles.append([(top, half_node_1), (half_node_1, half_node_2), (half_node_2, top)])
                    else:
                        new_cycles.append([(top, half_node_1), (half_node_1, bottom), (bottom, top)])

        elif idx == len(new_edges) - 1:
            if is_even:
                half_node_1 = cycle[half_size - 1]
                half_node_2 = cycle[half_size]
                new_cycles.append([(top, half_node_1), (half_node_1, half_node_2), (half_node_2, bottom), (bottom, top)])
            else:
                # Create a triangle cycle properly
                half_node = cycle[half_size]
                new_cycles.append([(top, half_node), (half_node, bottom), (bottom, top)])

        else:
            index_top = top_vertices.index(top)
            index_bottom = bottom_vertices.index(bottom)

            next_top = top_vertices[index_top + 1]
            next_bottom = bottom_vertices[index_bottom + 1]

            new_cycle = [(top, next_top), (next_top, next_bottom), (next_bottom, bottom), (bottom, top)]
            new_cycles.append(new_cycle)

    return new_cycles, new_edges, G


# Function to implement the straight sewing with cycles of length max 6
def sew_cycles_straight_6(cycle, G):

    if cycle[0] == cycle[-1]:
        cycle = cycle[:-1]

    n = len(cycle)
    
    # Only sew the cycle if its length is greater than 4
    if n > 6:
        
        # Split the cycle into two halves (top half and bottom half)
        half_size = n // 2
        
        top_vertices = cycle[1:half_size]
        bottom_vertices = cycle[1 + half_size:][::-1]
        
        # Add edges between corresponding opposite vertices
        new_edges = []
        new_cycles = []
        
        # Adding edges between the two halves
        for i in np.arange(0, half_size - 1, 2):
            if i + 1 < len(top_vertices):
                top = top_vertices[i + 1]
                bottom = bottom_vertices[i + 1]
                
                # Add the edge between the top half and the bottom half
                edge_to_add = (top, bottom)
                G.add_edge(*edge_to_add)
                new_edges.append(edge_to_add)
        
        # If the cycle length is even
        if n % 2 == 0:
            for (top, bottom) in new_edges:
                
                if (top, bottom) == new_edges[0]:
                    index_top = cycle.index(top)
                    index_bottom = cycle.index(bottom)

                    # print(cycle[:index_top+1])
                    # print(cycle[index_bottom:])
                    cycle_1_nodes = cycle[:index_top + 1] + cycle[index_bottom:]

                    cycle_1_edges = []
                    for i in range(len(cycle_1_nodes) - 1):
                        cycle_1_edges.append((cycle_1_nodes[i], cycle_1_nodes[i + 1]))
                    cycle_1_edges.append((cycle_1_nodes[-1], cycle_1_nodes[0]))
                    
                    index_top = cycle.index(top)
                    index_bottom = cycle.index(bottom)
                    
                    if n > 8:
                        first_next_top = cycle[index_top + 1]
                        second_next_top = cycle[index_top + 2]                    
                        first_next_bottom = cycle[index_bottom - 1]
                        second_next_bottom = cycle[index_bottom - 2]
                        
                        new_cycle_2 = [(top, first_next_top), (first_next_top, second_next_top), (second_next_top, second_next_bottom), (second_next_bottom, first_next_bottom), (first_next_bottom, bottom), (bottom, top)]
                        
                    else:
                        first_next_top = cycle[index_top + 1]
                        half_node = cycle[index_top + 2]                    
                        first_next_bottom = cycle[index_bottom - 1]
                        
                        new_cycle_2 = [(top, first_next_top), (first_next_top, half_node), (half_node, first_next_bottom), (first_next_bottom, bottom), (bottom, top)]

                    new_cycles.append(cycle_1_edges)
                    new_cycles.append(new_cycle_2)
                        
                elif (top, bottom) == new_edges[-1]:
                    
                    if len(cycle[top:bottom - 1]) == 1:                    
                        half_node = cycle[half_size]
                        new_cycle = [(top, half_node), (half_node, bottom), (bottom, top)]
                        new_cycles.append(new_cycle)
                        
                    elif len(cycle[top:bottom - 1]) == 3:
                        half_node_1 = cycle[top]
                        half_node_2 = cycle[top + 1]
                        half_node_3 = cycle[top + 2]
                        
                        new_cycle = [(top, half_node_1), (half_node_1, half_node_2), (half_node_2, half_node_3), (half_node_3, bottom), (bottom, top)]
                        new_cycles.append(new_cycle)

                else:
                    index_top = top_vertices.index(top)
                    index_bottom = bottom_vertices.index(bottom)
                    
                    first_next_top = top_vertices[index_top + 1]
                    second_next_top = top_vertices[index_top + 2]
                    first_next_bottom = bottom_vertices[index_bottom + 1]
                    second_next_bottom = bottom_vertices[index_bottom + 2]                    
                    new_cycle = [(top, first_next_top), (first_next_top, second_next_top), (second_next_top, second_next_bottom), (second_next_bottom, first_next_bottom), (first_next_bottom, bottom), (bottom, top)]
                    
                    if new_cycle not in new_cycles:
                        new_cycles.append(new_cycle)

            return new_cycles, new_edges, G

        # If the cycle length is odd
        elif n % 2 == 1:
            for (top, bottom) in new_edges:
                # print(new_edges)
                if (top, bottom) == new_edges[0]:
                    index_top = cycle.index(top)
                    index_bottom = cycle.index(bottom)
                    
                    # print(cycle[:index_top+1])
                    # print(cycle[index_bottom:])
                    cycle_1_nodes = cycle[:index_top + 1] + cycle[index_bottom:]

                    cycle_1_edges = []
                    for i in range(len(cycle_1_nodes) - 1):
                        cycle_1_edges.append((cycle_1_nodes[i], cycle_1_nodes[i + 1]))
                    cycle_1_edges.append((cycle_1_nodes[-1], cycle_1_nodes[0]))
                    
                    index_top = cycle.index(top)
                    index_bottom = cycle.index(bottom)
                    
                    if n > 7:
                        first_next_top = cycle[index_top + 1]
                        second_next_top = cycle[index_top + 2]                    
                        first_next_bottom = cycle[index_bottom - 1]
                        second_next_bottom = cycle[index_bottom - 2]
                        
                        new_cycle_2 = [(top, first_next_top), (first_next_top, second_next_top), (second_next_top, second_next_bottom), (second_next_bottom, first_next_bottom), (first_next_bottom, bottom), (bottom, top)]
                        
                    else:
                        first_next_top = cycle[index_top + 1]
                        first_next_bottom = cycle[index_bottom - 1]
                        
                        new_cycle_2 = [(top, first_next_top), (first_next_top, first_next_bottom), (first_next_bottom, bottom), (bottom, top)]

                    new_cycles.append(cycle_1_edges)
                    new_cycles.append(new_cycle_2)

                elif (top, bottom) == new_edges[-1]:
                    index_top = cycle.index(top)
                    index_bottom = cycle.index(bottom)

                    # print((top, bottom))
                    # print(cycle[index_top + 1:index_bottom])

                    nodes_between = cycle[index_top + 1:index_bottom]
                    # print(nodes_between)

                    new_cycle = [(top, nodes_between[0])]
                    for i in range(len(nodes_between) - 1):
                        node = nodes_between[i]
                        next_node = nodes_between[i + 1]
                        edge = (node, next_node)
                        new_cycle.append(edge)
                    
                    new_cycle.append((nodes_between[-1], bottom))
                    new_cycle.append((bottom, top))

                    new_cycles.append(new_cycle)
                
                else:
                    index_top = top_vertices.index(top)
                    index_bottom = bottom_vertices.index(bottom)

                    first_next_top = top_vertices[index_top + 1]
                    second_next_top = top_vertices[index_top + 2]
                    first_next_bottom = bottom_vertices[index_bottom + 1]
                    second_next_bottom = bottom_vertices[index_bottom + 2]                    
                    new_cycle = [(top, first_next_top), (first_next_top, second_next_top), (second_next_top, second_next_bottom), (second_next_bottom, first_next_bottom), (first_next_bottom, bottom), (bottom, top)]
                    
                    if new_cycle not in new_cycles:
                        new_cycles.append(new_cycle)

            return new_cycles, new_edges, G
    
    # Do not sew anything if the cycle has length <= 4
    else:
        cycle_edges = []
        for i in range(len(cycle) - 1):
            cycle_edges.append((cycle[i], cycle[i + 1]))
        cycle_edges.append((cycle[-1], cycle[0]))
        
        return [cycle_edges], [], G




# Code to create and visualize an example
def visualize_cycle_and_sewn_version(cycle, new_edges):
    # Create a new graph from the cycle and add the new edges
    G = nx.Graph()
    cycle_edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
    G.add_edges_from(cycle_edges)

    # Position nodes in a circular layout for clear visualization
    pos = nx.circular_layout(G)
    
    plt.figure(figsize=(8, 6))

    # Draw the original cycle in black
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold', edge_color='black')

    # Highlight the new edges (sewn edges) in blue
    if new_edges:
        nx.draw_networkx_edges(G, pos, edgelist=new_edges, edge_color='blue', width=2)

    plt.title("Original Cycle and Sewn Version (Highlighted Edges in Blue)")
    plt.axis("off")  # Turn off the axis for better appearance
    plt.show()


# # Example of a cycle with length 6
# cycle = [4, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 4]
# # cycle = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# G = nx.Graph()


# # Sew the cycle
# # new_cycles, new_edges, G = sew_cycles_straight(cycle, G)
# # new_cycles, new_edges, G = sew_cycles_triangular(cycle, G)
# # new_cycles, new_edges, G = sew_cycles_skew(cycle, G)
# new_cycles, new_edges, G = sew_cycles_straight_6(cycle, G)

# print("The obtained small cycles are:", new_cycles)

# visualize_cycle_and_sewn_version(cycle, new_edges=[])
# visualize_cycle_and_sewn_version(cycle, new_edges=new_edges)



# # Example of two cycles overlapping on 3 vertices
# # Shared vertices between the two cycles
# shared = [0, 1, 2]

# # Unique vertices for the first cycle (length 6)
# unique1 = [3, 4, 5]
# cycle1 = shared + unique1  # Total length = 6

# # Unique vertices for the second cycle (length 7)
# unique2 = [6, 7, 8, 9]
# cycle2 = shared + unique2  # Total length = 7

# # Create a base graph with both cycles included
# G = nx.Graph()

# # Apply sewing to cycle 1
# new_cycles_1, new_edges_1, G = sew_cycles_straight(cycle1, G)
# visualize_cycle_and_sewn_version(cycle1, new_edges_1)

# print("The obtained small cycles from cycle 1 are:", new_cycles_1)

# # Apply sewing to cycle 2 (must reuse the same G to keep shared nodes)
# new_cycles_2, new_edges_2, G = sew_cycles_straight(cycle2, G)
# visualize_cycle_and_sewn_version(cycle2, new_edges_2)

# print("The obtained small cycles from cycle 2 are:", new_cycles_2)
