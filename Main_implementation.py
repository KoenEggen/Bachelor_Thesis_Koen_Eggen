# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:43:26 2025

@author: koene
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from Stacked_graph_visualization import stack_graphs_3D
from collections import Counter
from multiprocessing import Pool
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from scipy.optimize import curve_fit
import random
import tempfile
import plotly.graph_objects as go
import mpld3
import io
import os
import sympy as sp
import pickle
from matplotlib.colors import to_rgb

from Creating_graphs import create_graph_SYK
from Creating_graphs import create_Erdos_Renyi_p
from SYK_Hamiltonians import syk_hamiltonian
from Vertex_coloring import Vertex_coloring_alg

from Cycle_basis_spanning_tree import cycle_basis_spanning_tree
from Horton import Horton
from Freedman_Hastings import Freedman_Hastings
from Cycle_basis_stacked import Stacked


# Values for the lattice size
# L = 6
L_range = np.arange(4, 151, 5)

# Specify the number of iterations and the confidence level
num_iterations = 50
confidence_level = 0.95

# Specify the threshold for the degree of the vertices in the graph and the maximum length a cycle in the basis must have
threshold = 0
threshold_degree = 4
threshold_length = 10
# p = 1/L**3 # p must scale as 1/n^3

# Parameters for the random bipartite expander graph
d = 4
beta = d/4

# Parameters for the Cayley graph and bipartite expander graph from Margulis
A = sp.Matrix([[1, 2], [0, 1]])  # Example matrix A
B = sp.Matrix([[1, 0], [2, 1]])  # Example matrix B
prime = 5

# Create all values for J and set each one to 0 with probability 1-p
# J = np.random.normal(0, 1, size=(L, L, L, L))

# mask = np.random.rand(L, L, L, L) < p
# J *= mask

# Step 1 - Create the SYK Hamiltonian ------------------------------------------------------------------------
# H_SYK = syk_hamiltonian(L, J, p)


# Step 2 - Create the fermionic interacion graph based on terms in the SYK Hamiltonian -----------------------

# Create the graph
# G = create_graph_SYK(L, J, p)
# G = create_graph_many_body(L, J, threshold)
# G = create_bipartite_expander(L, d, beta)


# Step 3 - Check whether or not the obtained graph and Hamiltonian are sparse and local and how they scale with L ------------
def check_graph_sparsity(G, L):
    max_edges = L * (L - 1) / 2
    actual_edges = G.number_of_edges()
    
    sparsity = 1 - (actual_edges / max_edges)
    return sparsity

def check_graph_locality(G):
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        avg_path_length = np.inf

    clustering_coeff = nx.average_clustering(G)  # Closer to 1 means highly local

    return avg_path_length, clustering_coeff

# spars_graph = check_graph_sparsity(G, L)
# avg_path_length, clustering_coeff = check_graph_locality(G)

# print(f"  - Sparsity: {spars_graph:.2f} {'Threshold = 0.5':>50}")
# print(f"  - Avg. Shortest Path Length: {avg_path_length:.2f}" if avg_path_length != np.inf else f"  - Avg. Shortest Path Length: Disconnected Graph{' ':>10}")
# print(f"  - Clustering Coefficient: {clustering_coeff:.2f} {'Threshold = 0.5':>36}")
# print()


# Step 4 - Create cycle bases based on different algorithms -----------------------------------------------

# cycle_basis_ST = cycle_basis_spanning_tree(G.copy())
# cycle_basis_Horton = Horton(G.copy())
# G_prime, cycle_basis_stacked, node_mapping = Stacked(G, cycle_basis_Horton, L)
# cycle_basis_FH = Freedman_Hastings(G.copy())

# print("Size of cycle basis Spanning Tree: ",len(cycle_basis_ST))
# print("Size of cycle basis Stacked graphs: ",len(cycle_basis_stacked))
# print("Size of cycle basis Horton: ",len(cycle_basis_Horton))
# print("Size of cycle basis FH: ",len(cycle_basis_FH))
# print()


# Step 5 - Map the SYK Hamiltonian to qubits by placing a qubit on every edge ----------------------------
# -


# Step 6 - Investigate how the sparsity and locality of the encoded Hamiltonian scales with L ------------

# Function to calculate the locality
def locality(cycle_basis):
    return max((len(cycle) for cycle in cycle_basis), default=0)

def min_locality(cycle_basis):
    return min((len(cycle) for cycle in cycle_basis), default=0)

# Function to calculate the sparsity
def sparsity(cycle_basis):
    edge_counts = Counter()
    
    for cycle in cycle_basis:
        for i in range(len(cycle)):
            # Create an undirected edge by sorting the tuple (u, v)
            edge = tuple(sorted((cycle[i], cycle[(i+1) % len(cycle)])))
            # Increment the counter for this edge
            edge_counts[edge] += 1

    # Return the maximum number of times any edge appears in the cycle basis
    return max(edge_counts.values())


# loc_FH = locality(cycle_basis_FH)
# spars_FH = sparsity(cycle_basis_FH)
# print("Sparsity of FH:",spars_FH)
# print("log(V)^2:",math.log(L)**2)
# print()


# Function to calculate the total number of edges (and thus qubits)
def total_edges(cycle_basis):
    edge_set = set()

    for cycle in cycle_basis:
        for i in range(len(cycle)):
            edge = tuple(sorted((cycle[i], cycle[(i+1) % len(cycle)])))
            edge_set.add(edge)

    return len(edge_set)


# Function to calculate the maximum degree for all vertices
def avg_degree(G):
    degrees = dict(G.degree())

    return np.mean(list(degrees.values()))


# Function to calculate the total weight of a cycle basis
def total_cycle_basis_weight(cycle_basis):

    return sum(len(cycle) for cycle in cycle_basis)


# Function to remove a vertex if the degree is too high
def remove_nodes(G, threshold):
    nodes_to_remove = [node for node, degree in G.degree() if degree > threshold]
    
    G.remove_nodes_from(nodes_to_remove)

    return G


# Function to remove edges of a vertex whose degree is too high until it is not anymore
def modify_graph_degree(G, threshold):
    nodes_to_modify = [node for node, degree in G.degree() if degree > threshold]
    
    for node in nodes_to_modify:
        while G.degree(node) > threshold:
            edges = list(G.edges(node))
            if edges:
                edge_to_remove = random.choice(edges)
                G.remove_edge(*edge_to_remove)
    
    return G


# Calculate how certain properties depend on the lattice size n for different algorithms
def L_dependence(L_values, num_iterations, confidence_level, threshold_degree, threshold_length, d, beta, prime):
    
    results = {}
    
    # Initialize empty lists to keep track of all final values calculated after averaging
    loc_ST = []
    min_loc_Horton = []
    loc_Stacked = []
    loc_FH = []
    loc_VC = []
    
    spars_ST = []
    spars_Horton = []
    spars_Stacked = []
    spars_FH = []
    spars_VC = []
    
    original_edges = []    
    edges_FH = []    
    expected_edges_Stacked = []
    actual_edges_Stacked = []

    expected_edges_VC = []
    actual_edges_VC = []
    
    tot_layers_VC = []
    
    weight_FH = []
    weight_Horton = []
    size_FH = []
    weight_VC = []
    size_VC = []
    
    mean_cycle_length_FH = []
    mean_cycle_length_VC = []

    degrees_G = []
    degrees_G_Stacked = []
    degrees_G_VC = []

    big_cycles_FH = []
    
    loc_ST_ci = []
    min_loc_Horton_ci = []
    loc_Stacked_ci = []
    loc_FH_ci = []
    loc_VC_ci = []
    
    spars_ST_ci = []
    spars_Horton_ci = []
    spars_Stacked_ci = []
    spars_FH_ci = []
    spars_VC_ci = []
    
    big_cycles_FH_ci = []
    size_FH_ci = []
    weight_FH_ci = []
    weight_Horton_ci = []

    mean_cycle_length_FH_ci = []
    weight_VC_ci = []
    mean_cycle_length_VC_ci = []
    size_VC_ci = []

    original_edges_ci = []
    edges_FH_ci = []
    actual_edges_Stacked_ci = []
    actual_edges_VC_ci = []
    tot_layers_VC_ci = []

    degrees_G_ci = []
    degrees_G_Stacked_ci = []    
    degrees_G_VC_ci = []
    
    loc_ST_values_all = []
    loc_FH_values_all = []
    spars_ST_values_all = []
    spars_FH_values_all = []
    
    for L in L_values:
        print(f"Processing L = {L}")
        
        p = 1 / L**3 # for the sparse SYK graph
        # p = 10 * 1 / L # for the Erdos-Renyi random graph
        
        # Initialize empty lists to keep track of all values for averaging        
        loc_ST_values = []
        min_loc_Horton_values = []
        loc_Stacked_values = []
        loc_FH_values = []
        loc_VC_values = []
        
        spars_ST_values = []
        spars_Horton_values = []
        spars_Stacked_values = []
        spars_FH_values = []
        spars_VC_values = []
        
        expected_edges_Stacked_values = []
        actual_edges_Stacked_values = []

        expected_edges_VC_values = []
        actual_edges_VC_values = []
        tot_layers_VC_values = []
        
        original_edges_values = []
        size_FH_values = []
        edges_FH_values = []
        weight_FH_values = []
        weight_Horton_values = []
        mean_cycle_length_FH_values = []
        
        size_VC_values = []
        weight_VC_values = []
        mean_cycle_length_VC_values = []
        
        degrees_G_values = []
        degrees_G_Stacked_values = []
        degrees_G_VC_values = []
        
        big_cycles_FH_values = [0] * num_iterations
        
        # Perform a number of iterations for each value of L
        for i in range(num_iterations):
            
            # Create the graph and make sure that the degree of each vertex is <= threshold_degree by removing edges
            # G = create_Erdos_Renyi_p(L, p)
            G = create_graph_SYK(L, p)
            
            G = modify_graph_degree(G, threshold_degree)
            
            # Calculate the number of edges and the degree of the vertices in the graph
            original_edges_values.append(len(G.edges()))
            
            degrees_G_values.append(avg_degree(G.copy()))
            
            # Calculate the cycle basis using a Spanning Tree, Horton, Freedman-Hastings (FH) and the stacked graphs algorithm
            CB_ST = cycle_basis_spanning_tree(G.copy())
            # CB_Horton = Horton(G.copy())
            CB_FH = Freedman_Hastings(G.copy())
            
            
            if len(CB_ST) > 0:
                expected_edges_Stacked_temp, actual_edges_Stacked_temp, G_Stacked, CB_Stacked, tot_layers_temp = Stacked(G.copy(), L, cycle_basis=CB_ST, cycles_to_sew=None)
            
                degrees_G_Stacked_values.append(avg_degree(G_Stacked))

                loc_ST_values.append(locality(CB_ST))
                spars_ST_values.append(sparsity(CB_ST))

                loc_Stacked_values.append(locality(CB_Stacked))    
                spars_Stacked_values.append(sparsity(CB_Stacked))
                
                expected_edges_Stacked_values.append(expected_edges_Stacked_temp)
                actual_edges_Stacked_values.append(actual_edges_Stacked_temp)

            # if len(CB_Horton) > 0:
                # min_loc_Horton_values.append(locality(CB_Horton))
                # spars_Horton_values.append(sparsity(CB_Horton))
                # weight_Horton_values.append(total_cycle_basis_weight(CB_Horton))
            
            if len(CB_FH) > 0:
                # Keep track of the cycles with length > threshold_length
                for cycle in CB_FH:
                    if len(cycle) > threshold_length:
                        big_cycles_FH_values[i] += 1
                
                mean_cycle_length_FH_temp = np.mean([len(cycle) for cycle in CB_FH])
                mean_cycle_length_FH_values.append(mean_cycle_length_FH_temp)
                                
                # Calculate and add all values to the lists for averaging
                loc_FH_values.append(locality(CB_FH))
                spars_FH_values.append(sparsity(CB_FH))
                
                size_FH_values.append(len(CB_FH))
                edges_FH_values.append(len(G.edges()))
                weight_FH_values.append(total_cycle_basis_weight(CB_FH))
                
                # Use the vertex coloring algorithm to calculate a cycle basis
                expected_edges_VC_temp, actual_edges_VC_temp, G_VC, CB_VC, tot_layers_VC_temp, _ = Vertex_coloring_alg(G.copy(), L, CB_FH)
            
                degrees_G_VC_values.append(avg_degree(G_VC))

                if len(CB_VC) > 0:
                    loc_VC_values.append(locality(CB_VC))
                    spars_VC_values.append(sparsity(CB_VC))
    
                    mean_cycle_length_VC_temp = np.mean([len(cycle) for cycle in CB_VC])
                    mean_cycle_length_VC_values.append(mean_cycle_length_VC_temp)
    
                    expected_edges_VC_values.append(expected_edges_VC_temp)
                    actual_edges_VC_values.append(actual_edges_VC_temp)
                    tot_layers_VC_values.append(tot_layers_VC_temp)
                    size_VC_values.append(len(CB_VC))
                    weight_VC_values.append(total_cycle_basis_weight(CB_VC))
            
                # elif len(CB_VC) == 0:
                #     mean_cycle_length_VC_values.append(0)

            # elif len(CB_FH) == 0:
            #     mean_cycle_length_FH_values.append(0)

        
        # Keep track of all values to calculate the distribution
        loc_ST_values_all.append(loc_ST_values)
        loc_FH_values_all.append(loc_FH_values)
        spars_ST_values_all.append(spars_ST_values)
        spars_FH_values_all.append(spars_FH_values)
        
        # Add the average values to all lists
        loc_ST.append(np.mean(loc_ST_values))
        min_loc_Horton.append(np.mean(min_loc_Horton_values))
        loc_Stacked.append(np.mean(loc_Stacked_values))
        loc_FH.append(np.mean(loc_FH_values))
        loc_VC.append(np.mean(loc_VC_values))
        
        spars_ST.append(np.mean(spars_ST_values))
        spars_Horton.append(np.mean(spars_Horton_values))
        spars_Stacked.append(np.mean(spars_Stacked_values))
        spars_FH.append(np.mean(spars_FH_values))
        spars_VC.append(np.mean(spars_VC_values))
        
        expected_edges_Stacked.append(np.mean(expected_edges_Stacked_values))
        actual_edges_Stacked.append(np.mean(actual_edges_Stacked_values))
        
        expected_edges_VC.append(np.mean(expected_edges_VC_values))
        actual_edges_VC.append(np.mean(actual_edges_VC_values))
        tot_layers_VC.append(np.mean(tot_layers_VC_values))
        
        original_edges.append(np.mean(original_edges_values))
        size_FH.append(np.mean(size_FH_values))
        edges_FH.append(np.mean(edges_FH_values)) 
        weight_FH.append(np.mean(weight_FH_values))
        mean_cycle_length_FH.append(np.mean(mean_cycle_length_FH_values))
        weight_Horton.append(np.mean(weight_Horton_values))
        
        size_VC.append(np.mean(size_VC_values))
        weight_VC.append(np.mean(weight_VC_values))
        mean_cycle_length_VC.append(np.mean(mean_cycle_length_VC_values))
        
        degrees_G.append(np.mean(degrees_G_values))
        degrees_G_Stacked.append(np.mean(degrees_G_Stacked_values))
        degrees_G_VC.append(np.mean(degrees_G_VC_values))
        
        big_cycles_FH.append(np.mean(big_cycles_FH_values))
        
        # Calculate the standard deviations and the confidence intervals
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                
        def compute_ci(values, z_score, num_iterations):
            mean = np.mean(values)
            std = np.std(values)
            margin = z_score * std / np.sqrt(num_iterations)
            return (mean - margin, mean + margin)
        
        # Compute and append confidence intervals
        loc_ST_ci.append(compute_ci(loc_ST_values, z_score, num_iterations))
        min_loc_Horton_ci.append(compute_ci(min_loc_Horton_values, z_score, num_iterations))
        loc_Stacked_ci.append(compute_ci(loc_Stacked_values, z_score, num_iterations))
        loc_FH_ci.append(compute_ci(loc_FH_values, z_score, num_iterations))
        loc_VC_ci.append(compute_ci(loc_VC_values, z_score, num_iterations))
        
        spars_ST_ci.append(compute_ci(spars_ST_values, z_score, num_iterations))
        spars_Horton_ci.append(compute_ci(spars_Horton_values, z_score, num_iterations))
        spars_Stacked_ci.append(compute_ci(spars_Stacked_values, z_score, num_iterations))
        spars_FH_ci.append(compute_ci(spars_FH_values, z_score, num_iterations))
        spars_VC_ci.append(compute_ci(spars_VC_values, z_score, num_iterations))

        big_cycles_FH_ci.append(compute_ci(big_cycles_FH_values, z_score, num_iterations))
        size_FH_ci.append(compute_ci(size_FH_values, z_score, num_iterations))
        weight_FH_ci.append(compute_ci(weight_FH_values, z_score, num_iterations))
        weight_Horton_ci.append(compute_ci(weight_Horton_values, z_score, num_iterations))

        mean_cycle_length_FH_ci.append(compute_ci(mean_cycle_length_FH_values, z_score, num_iterations))
        weight_VC_ci.append(compute_ci(weight_VC_values, z_score, num_iterations))
        mean_cycle_length_VC_ci.append(compute_ci(mean_cycle_length_VC_values, z_score, num_iterations))

        original_edges_ci.append(compute_ci(original_edges_values, z_score, num_iterations))
        edges_FH_ci.append(compute_ci(edges_FH_values, z_score, num_iterations))
        actual_edges_Stacked_ci.append(compute_ci(actual_edges_Stacked_values, z_score, num_iterations))
        actual_edges_VC_ci.append(compute_ci(actual_edges_VC_values, z_score, num_iterations))
        tot_layers_VC_ci.append(compute_ci(tot_layers_VC_values, z_score, num_iterations))
        size_VC_ci.append(compute_ci(size_VC_values, z_score, num_iterations))
        degrees_G_VC_ci.append(compute_ci(degrees_G_VC_values, z_score, num_iterations))
        degrees_G_Stacked_ci.append(compute_ci(degrees_G_Stacked_values, z_score, num_iterations))
        degrees_G_ci.append(compute_ci(degrees_G_values, z_score, num_iterations))
        
        
    # Store the calculated results into the results dictionary
    results['loc_ST'] = loc_ST
    results['min_loc_Horton'] = min_loc_Horton
    results['loc_Stacked'] = loc_Stacked
    results['loc_FH'] = loc_FH
    results['loc_VC'] = loc_VC
    
    results['spars_ST'] = spars_ST
    results['spars_Horton'] = spars_Horton
    results['spars_Stacked'] = spars_Stacked
    results['spars_FH'] = spars_FH
    results['spars_VC'] = spars_VC
    
    results['expected_edges_Stacked'] = expected_edges_Stacked
    results['actual_edges_Stacked'] = actual_edges_Stacked
    
    results['expected_edges_VC'] = expected_edges_VC
    results['actual_edges_VC'] = actual_edges_VC
    results['tot_layers_VC'] = tot_layers_VC
    
    results['loc_ST_ci'] = loc_ST_ci
    results['min_loc_Horton_ci'] = min_loc_Horton_ci
    results['loc_Stacked_ci'] = loc_Stacked_ci
    results['loc_FH_ci'] = loc_FH_ci
    results['loc_VC_ci'] = loc_VC_ci
    
    results['spars_ST_ci'] = spars_ST_ci
    results['spars_Horton_ci'] = spars_Horton_ci
    results['spars_Stacked_ci'] = spars_Stacked_ci
    results['spars_FH_ci'] = spars_FH_ci
    results['spars_VC_ci'] = spars_VC_ci

    results['original_edges'] = original_edges    
    results['size_FH'] = size_FH
    results['size_VC'] = size_VC
    results['edges_FH'] = edges_FH
    results['degrees_G'] = degrees_G
    results['degrees_G_Stacked'] = degrees_G_Stacked
    results['degrees_G_VC'] = degrees_G_VC
    
    results['big_cycles_FH'] = big_cycles_FH
    
    results['loc_ST_values_all'] = loc_ST_values_all
    results['loc_FH_values_all'] = loc_FH_values_all
    results['spars_ST_values_all'] = spars_ST_values_all
    results['spars_FH_values_all'] = spars_FH_values_all
    
    results['weight_Horton'] = weight_Horton
    results['weight_FH'] = weight_FH
    results['weight_VC'] = weight_VC
    results['mean_cycle_length_FH'] = mean_cycle_length_FH
    results['mean_cycle_length_VC'] = mean_cycle_length_VC
    
    results['big_cycles_FH_ci'] = big_cycles_FH_ci
    results['size_FH_ci'] = size_FH_ci
    results['weight_FH_ci'] = weight_FH_ci
    results['weight_Horton_ci'] = weight_Horton_ci
    results['weight_VC_ci'] = weight_VC_ci
    results['mean_cycle_length_FH_ci'] = mean_cycle_length_FH_ci
    results['mean_cycle_length_VC_ci'] = mean_cycle_length_VC_ci
    results['size_VC_ci'] = size_VC_ci
    
    results['original_edges_ci'] = original_edges_ci
    results['edges_FH_ci'] = edges_FH_ci
    results['actual_edges_Stacked_ci'] = actual_edges_Stacked_ci
    results['actual_edges_VC_ci'] = actual_edges_VC_ci
    results['tot_layers_VC_ci'] = tot_layers_VC_ci

    results['degrees_G_ci'] = degrees_G_ci
    results['degrees_G_Stacked_ci'] = degrees_G_Stacked_ci
    results['degrees_G_VC_ci'] = degrees_G_VC_ci
        
    return results


# Functions to plot the results

# Create the save directory
save_dir = r"C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Figuren\Data-analyse"
os.makedirs(save_dir, exist_ok=True)

# Helper functions
def lighten_color(color, factor=0.3):
    r, g, b = to_rgb(color)
    return ((1 - factor) + factor * r,
            (1 - factor) + factor * g,
            (1 - factor) + factor * b)

def plot_with_ci_custom(x, y, y_lower, y_upper, label, color, marker, linestyle, markersize=5):
    plt.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle, markersize=markersize)
    light_color = lighten_color(color, factor=0.3)
    plt.fill_between(x, y_lower, y_upper, color=light_color)

def extract_bounds(ci_list):
    lower = [ci[0] for ci in ci_list]
    upper = [ci[1] for ci in ci_list]
    return lower, upper

def compute_edges_per_n(edges, ci_list, n_values):
    edges_div_n = [e / n for e, n in zip(edges, n_values)]
    lower = [ci[0] / n for ci, n in zip(ci_list, n_values)]
    upper = [ci[1] / n for ci, n in zip(ci_list, n_values)]
    return edges_div_n, lower, upper

def customize_plot(xlabel, ylabel, filename, ylim=None):
    ax = plt.gca()
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', labelsize=16)

    # Optional: Apply y-limits
    if ylim:
        plt.ylim(ymin=ylim[0], ymax=ylim[1])

    # Remove top and right spines always
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    # Align origin **only** if 0 is visible in both axes
    x0_visible = plt.xlim()[0] <= 0 <= plt.xlim()[1]
    y0_visible = plt.ylim()[0] <= 0 <= plt.ylim()[1]

    if x0_visible:
        ax.spines['left'].set_position('zero')
    else:
        ax.spines['left'].set_position(('outward', 0))

    if y0_visible:
        ax.spines['bottom'].set_position('zero')
    else:
        ax.spines['bottom'].set_position(('outward', 0))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(save_dir, filename), format='eps')
    plt.show()


def finalize_plot(xlabel, ylabel, filename):
    ax = plt.gca()
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    # Only show legend if any artist has a label that doesn't start with "_"
    handles, labels = ax.get_legend_handles_labels()
    if any(label and not label.startswith('_') for label in labels):
        plt.legend(fontsize=16)

    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), format='eps')
    plt.show()
    

# Plot of locality & sparsity next to each other
def plot_L_dependence(
    L_range,
    loc_ST, loc_Stacked, loc_FH, loc_VC,
    spars_ST, spars_Stacked, spars_FH, spars_VC,
    loc_ST_ci, loc_Stacked_ci, loc_FH_ci, loc_VC_ci,
    spars_ST_ci, spars_Stacked_ci, spars_FH_ci, spars_VC_ci,
    num_iterations, name_graph,
    log_L, log_L_sqrd,
    loc_Horton, spars_Horton,
    loc_Horton_ci, spars_Horton_ci,
    filename,  # <- Moved here to match logical order
    bound_spars=None, bound_loc=None,
    save_path=save_dir,
    label_fontsize=30, tick_fontsize=26, legend_fontsize=26
):

    plt.figure(figsize=(28, 12))

    # ---- Locality Plot ----
    plt.subplot(1, 2, 1)
    plt.plot(L_range, log_L, label="log(n)", color="black")
    plt.plot(L_range, L_range, label="n", color="grey")
    if bound_spars is not None:
        plt.plot(L_range, bound_spars, label="log(n)^2", color="black", linestyle="--")
    if bound_loc is not None:
        plt.plot(L_range, L_range * bound_loc, label="nlog(n)", color="blue", linestyle="--")

    # Spanning Tree
    y_lower = [ci[0] for ci in loc_ST_ci]
    y_upper = [2 * avg - ci[0] for avg, ci in zip(loc_ST, loc_ST_ci)]
    plot_with_ci_custom(L_range, loc_ST, y_lower, y_upper, "Spanning tree", "red", "^", ":")

    # Stacked Graphs
    y_lower = [ci[0] for ci in loc_Stacked_ci]
    y_upper = [2 * avg - ci[0] for avg, ci in zip(loc_Stacked, loc_Stacked_ci)]
    plot_with_ci_custom(L_range, loc_Stacked, y_lower, y_upper, "Stacked graphs", "green", "o", "-.")

    # Freedman-Hastings
    y_lower = [ci[0] for ci in loc_FH_ci]
    y_upper = [2 * avg - ci[0] for avg, ci in zip(loc_FH, loc_FH_ci)]
    plot_with_ci_custom(L_range, loc_FH, y_lower, y_upper, "Freedman-Hastings", "orange", "s", "--")

    # # Vertex Coloring
    # y_lower = [ci[0] for ci in loc_VC_ci]
    # y_upper = [2 * avg - ci[0] for avg, ci in zip(loc_VC, loc_VC_ci)]
    # plot_with_ci_custom(L_range, loc_VC, y_lower, y_upper, "Vertex coloring", "blue", ".", "--")

    # Horton (unchanged)
    if loc_Horton is not None and loc_Horton_ci is not None:
        y_lower = [ci[0] for ci in loc_Horton_ci]
        y_upper = [2 * avg - ci[0] for avg, ci in zip(loc_Horton, loc_Horton_ci)]
        plot_with_ci_custom(L_range, loc_Horton, y_lower, y_upper, "Horton", "purple", ".", "--")


    plt.xlabel("Lattice size (n)", fontsize=label_fontsize)
    plt.ylabel("Locality", fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.ylim(0, 40)
    plt.grid()
    plt.legend(fontsize=legend_fontsize)

    # ---- Sparsity Plot ----
    plt.subplot(1, 2, 2)
    plt.plot(L_range, log_L_sqrd, label="log(n)^2", color="black")
    plt.plot(L_range, L_range, label="n", color="grey")
    if bound_loc is not None:
        plt.plot(L_range, bound_loc, label="log(n)", color="blue", linestyle="--")

    # Spanning Tree
    y_lower = [ci[0] for ci in spars_ST_ci]
    y_upper = [2 * avg - ci[0] for avg, ci in zip(spars_ST, spars_ST_ci)]
    plot_with_ci_custom(L_range, spars_ST, y_lower, y_upper, "Spanning tree", "red", "^", ":")

    # Stacked Graphs
    y_lower = [ci[0] for ci in spars_Stacked_ci]
    y_upper = [2 * avg - ci[0] for avg, ci in zip(spars_Stacked, spars_Stacked_ci)]
    plot_with_ci_custom(L_range, spars_Stacked, y_lower, y_upper, "Stacked graphs", "green", "o", "-.")

    # Freedman-Hastings
    y_lower = [ci[0] for ci in spars_FH_ci]
    y_upper = [2 * avg - ci[0] for avg, ci in zip(spars_FH, spars_FH_ci)]
    plot_with_ci_custom(L_range, spars_FH, y_lower, y_upper, "Freedman-Hastings", "orange", "s", "--")

    # # Vertex Coloring
    # y_lower = [ci[0] for ci in spars_VC_ci]
    # y_upper = [2 * avg - ci[0] for avg, ci in zip(spars_VC, spars_VC_ci)]
    # plot_with_ci_custom(L_range, spars_VC, y_lower, y_upper, "Vertex coloring", "blue", ".", "--")

    # Horton (unchanged)
    if spars_Horton is not None and spars_Horton_ci is not None:
        y_lower = [ci[0] for ci in spars_Horton_ci]
        y_upper = [2 * avg - ci[0] for avg, ci in zip(spars_Horton, spars_Horton_ci)]
        plot_with_ci_custom(L_range, spars_Horton, y_lower, y_upper, "Horton", "purple", ".", "--")

    plt.xlabel("Lattice size (n)", fontsize=label_fontsize)
    plt.ylabel("Sparsity", fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.ylim(0, 50)
    plt.grid()
    plt.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig(f"{save_path}\\{filename}", format='eps')
    plt.show()


# General plots with all results
def plot_all_with_errorbars(
    L_range,
    loc_data, loc_cis,
    spars_data, spars_cis, edges_div_n_FH, edges_div_n_Stacked,
    edges_div_n_data, edges_cis,
    degrees_data, degrees_cis
):
    global save_dir
    save_dir = r"C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Figuren\Data-analyse\Erdos-Renyi graph"
    os.makedirs(save_dir, exist_ok=True)

    styles = [
        {'label': "VC straight sewing, no overlap", 'color': "blue", 'marker': 'o', 'linestyle': "-"},
        {'label': "VC straight sewing, 1 overlap", 'color': "red", 'marker': '^', 'linestyle': "-."},
        {'label': "VC straight sewing, 2 overlap", 'color': "purple", 'marker': 'v', 'linestyle': ":"},
        {'label': "VC straight sewing, 3 overlap", 'color': "black", 'marker': 'h', 'linestyle': "-"},
        {'label': "VC triangular sewing, no overlap", 'color': "brown", 'marker': 'D', 'linestyle': "--"},
        {'label': "VC skew sewing, no overlap", 'color': "deeppink", 'marker': '*', 'linestyle': "-"},
        {'label': "VC straight-6 sewing, 1 overlap", 'color': "gray", 'marker': 'P', 'linestyle': "-."},
        {'label': "VC straight-6 sewing, 2 overlap", 'color': "olive", 'marker': 'X', 'linestyle': ":"},
        {'label': "VC straight-6 sewing, 3 overlap", 'color': "teal", 'marker': 's', 'linestyle': "-"},
    ]

    fh_style = {'label': "Freedman-Hastings", 'color': "orange", 'marker': 's', 'linestyle': "--"}
    stacked_style = {'label': "Stacking and sewing", 'color': "green", 'marker': '.', 'linestyle': "--"}

    # Locality
    plt.figure(figsize=(15, 8))
    plot_with_ci_custom(L_range, loc_FH, *extract_bounds(loc_FH_ci), **fh_style)
    plot_with_ci_custom(L_range, loc_Stacked, *extract_bounds(loc_Stacked_ci), **stacked_style)
    for data, ci, style in zip(loc_data, loc_cis, styles):
        plot_with_ci_custom(L_range, data, *extract_bounds(ci), **style)
    customize_plot("Lattice size (n)", "Maximum cycle length", "Locality (Erdos-Renyi graph, general).eps", ylim=(0, 15))

    # Sparsity
    plt.figure(figsize=(15, 8))
    plot_with_ci_custom(L_range, spars_FH, *extract_bounds(spars_FH_ci), **fh_style)
    plot_with_ci_custom(L_range, spars_Stacked, *extract_bounds(spars_Stacked_ci), **stacked_style)
    for data, ci, style in zip(spars_data, spars_cis, styles):
        plot_with_ci_custom(L_range, data, *extract_bounds(ci), **style)
    customize_plot("Lattice size (n)", "Sparsity", "Sparsity (Erdos-Renyi graph, general, zoomed).eps", ylim=(0, 16))

    # Num edges div n
    plt.figure(figsize=(15, 8))
    plot_with_ci_custom(L_range, edges_div_n_FH,
                        [ci[0] / n for ci, n in zip(edges_FH_ci, L_range)],
                        [ci[1] / n for ci, n in zip(edges_FH_ci, L_range)],
                        **fh_style)
    plot_with_ci_custom(L_range, edges_div_n_Stacked,
                        [ci[0] / n for ci, n in zip(actual_edges_Stacked_ci, L_range)],
                        [ci[1] / n for ci, n in zip(actual_edges_Stacked_ci, L_range)],
                        **stacked_style)
    for edges_div_n, ci, style in zip(edges_div_n_data, edges_cis, styles):
        lower = [ci[0] / n for ci, n in zip(ci, L_range)]
        upper = [ci[1] / n for ci, n in zip(ci, L_range)]
        plot_with_ci_custom(L_range, edges_div_n, lower, upper, **style)
    customize_plot("Lattice size (n)", "Number of edges divided by the lattice size n", "Num edges div n (Erdos-Renyi graph, general, zoomed).eps", ylim=(0, 80))

    # Average degree
    plt.figure(figsize=(15, 8))
    plot_with_ci_custom(L_range, degrees_G, *extract_bounds(degrees_G_ci), **fh_style)
    plot_with_ci_custom(L_range, degrees_G_Stacked, *extract_bounds(degrees_G_Stacked_ci), **stacked_style)
    for data, ci, style in zip(degrees_data, degrees_cis, styles):
        plot_with_ci_custom(L_range, data, *extract_bounds(ci), **style)
    customize_plot("Lattice size (n)", "Average degree of vertices in the obtained graph", "Average degree (Erdos-Renyi graph, general, zoomed).eps", ylim=(2, 6))





# # Run the function L_dependence and store the data using pickle
# results = L_dependence(L_range, num_iterations, confidence_level, threshold_degree, threshold_length, d, beta, prime)

# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\results_SYK_straight_sewing_no_overlap.pkl', 'wb') as f:
#     pickle.dump(results, f)




# # Load the data and assign the values to the corresponding variables
# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Erdos-Renyi graph\results_expander_Erdos_straight_sewing_no_overlap_with_max_Horton.pkl', 'rb') as f:
#     results_straight_sewing_no_overlap_with_max_Horton = pickle.load(f)

# # Load the data and assign the values to the corresponding variables
# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Erdos-Renyi graph\results_expander_Erdos_straight_sewing_no_overlap_with_size.pkl', 'rb') as f:
#     results_straight_sewing_no_overlap_with_size = pickle.load(f)

# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Erdos-Renyi graph\results_expander_Erdos_straight_sewing_1_overlap_with_size.pkl', 'rb') as f:
#     results_straight_sewing_1_overlap_with_size = pickle.load(f)

# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Erdos-Renyi graph\results_expander_Erdos_straight_sewing_2_overlap_with_size.pkl', 'rb') as f:
#     results_straight_sewing_2_overlap_with_size = pickle.load(f)

# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Erdos-Renyi graph\results_expander_Erdos_straight_sewing_3_overlap_with_size.pkl', 'rb') as f:
#     results_straight_sewing_3_overlap_with_size = pickle.load(f)

# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Erdos-Renyi graph\results_expander_Erdos_straight_6_sewing_1_overlap_with_size.pkl', 'rb') as f:
#     results_straight_6_sewing_1_overlap_with_size = pickle.load(f)

# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Erdos-Renyi graph\results_expander_Erdos_straight_6_sewing_2_overlap_with_size.pkl', 'rb') as f:
#     results_straight_6_sewing_2_overlap_with_size = pickle.load(f)

# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Erdos-Renyi graph\results_expander_Erdos_straight_6_sewing_3_overlap_with_size.pkl', 'rb') as f:
#     results_straight_6_sewing_3_overlap_with_size = pickle.load(f)

# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Erdos-Renyi graph\results_expander_Erdos_triangular_sewing_no_overlap_with_size.pkl', 'rb') as f:
#     results_triangular_sewing_no_overlap_with_size = pickle.load(f)

# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Erdos-Renyi graph\results_expander_Erdos_skew_sewing_no_overlap_with_size.pkl', 'rb') as f:
#     results_skew_sewing_no_overlap_with_size = pickle.load(f)


# # Load the data with max Horton
# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Erdos-Renyi graph\results_expander_Erdos_straight_sewing_no_overlap_with_max_Horton.pkl', 'rb') as f:
#     results_expander_Erdos_straight_sewing_no_overlap_with_max_Horton = pickle.load(f)


# # Load the data of the maximal/average sparsity of Freedman-Hastings
# with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Erdos-Renyi graph\results_Erdos_Renyi_straight_sewing_no_overlap_outlier_decongestion_mean_sparsity.pkl', 'rb') as f:
#     results_straight_no_overlap_outlier_decongestion_mean_sparsity = pickle.load(f)


# Load the data of the sparse SYK graph
with open(r'C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Numerieke implementatie\Data\Sparse SYK graph\results_SYK_straight_sewing_no_overlap.pkl', 'rb') as f:
    results_SYK_straight_sewing_no_overlap = pickle.load(f)



# Define the variables
def extract_results_vars(results):
    return {
        'loc_ST': results['loc_ST'],
        'min_loc_Horton': results['min_loc_Horton'],
        'loc_Stacked': results['loc_Stacked'],
        'loc_FH': results['loc_FH'],
        'loc_VC': results['loc_VC'],

        'spars_ST': results['spars_ST'],
        'spars_Horton': results['spars_Horton'],
        'spars_Stacked': results['spars_Stacked'],
        'spars_FH': results['spars_FH'],
        'spars_VC': results['spars_VC'],

        'edges_Stacked': results['actual_edges_Stacked'],
        'edges_VC': results['actual_edges_VC'],
        'expected_edges_VC': results['expected_edges_VC'],
        'tot_layers_VC': results['tot_layers_VC'],

        'loc_ST_ci': results['loc_ST_ci'],
        'min_loc_Horton_ci': results['min_loc_Horton_ci'],
        'loc_Stacked_ci': results['loc_Stacked_ci'],
        'loc_FH_ci': results['loc_FH_ci'],
        'loc_VC_ci': results['loc_VC_ci'],

        'spars_ST_ci': results['spars_ST_ci'],
        'spars_Horton_ci': results['spars_Horton_ci'],
        'spars_Stacked_ci': results['spars_Stacked_ci'],
        'spars_FH_ci': results['spars_FH_ci'],
        'spars_VC_ci': results['spars_VC_ci'],

        'original_edges': results['original_edges'],
        'size_FH': results['size_FH'],
        'size_VC': results['size_VC'],
        'edges_FH': results['edges_FH'],
        'degrees_G': results['degrees_G'],
        'degrees_G_Stacked': results['degrees_G_Stacked'],
        'degrees_G_VC': results['degrees_G_VC'],

        'big_cycles_FH': results['big_cycles_FH'],

        'loc_ST_values_all': results['loc_ST_values_all'],
        'loc_FH_values_all': results['loc_FH_values_all'],
        'spars_ST_values_all': results['spars_ST_values_all'],
        'spars_FH_values_all': results['spars_FH_values_all'],

        'weight_Horton': results['weight_Horton'],
        'weight_FH': results['weight_FH'],
        'weight_VC': results['weight_VC'],
        'mean_cycle_length_FH': results['mean_cycle_length_FH'],
        'mean_cycle_length_VC': results['mean_cycle_length_VC'],

        'big_cycles_FH_ci': results['big_cycles_FH_ci'],
        'size_FH_ci': results['size_FH_ci'],
        'weight_Horton_ci': results['weight_Horton_ci'],
        'weight_FH_ci': results['weight_FH_ci'],
        'weight_VC_ci': results['weight_VC_ci'],
        'mean_cycle_length_FH_ci': results['mean_cycle_length_FH_ci'],
        'mean_cycle_length_VC_ci': results['mean_cycle_length_VC_ci'],
        'size_VC_ci': results['size_VC_ci'],

        'original_edges_ci': results['original_edges_ci'],
        'edges_FH_ci': results['edges_FH_ci'],
        'actual_edges_Stacked_ci': results['actual_edges_Stacked_ci'],
        'edges_VC_ci': results['actual_edges_VC_ci'],
        'tot_layers_VC_ci': results['tot_layers_VC_ci'],

        'degrees_G_ci': results['degrees_G_ci'],
        'degrees_G_Stacked_ci': results['degrees_G_Stacked_ci'],
        'degrees_G_VC_ci': results['degrees_G_VC_ci'],
    }



variables = extract_results_vars(results_SYK_straight_sewing_no_overlap)
name_results = "(sparse SYK graph)"



loc_ST = variables['loc_ST']
min_loc_Horton = variables['min_loc_Horton']
loc_Stacked = variables['loc_Stacked']
loc_FH = variables['loc_FH']
loc_VC = variables['loc_VC']

spars_ST = variables['spars_ST']
spars_Horton = variables['spars_Horton']
spars_Stacked = variables['spars_Stacked']
spars_FH = variables['spars_FH']
spars_VC = variables['spars_VC']

edges_Stacked = variables['edges_Stacked']
edges_VC = variables['edges_VC']
expected_edges_VC = variables['expected_edges_VC']
tot_layers_VC = variables['tot_layers_VC']

loc_ST_ci = variables['loc_ST_ci']
min_loc_Horton_ci = variables['min_loc_Horton_ci']
loc_Stacked_ci = variables['loc_Stacked_ci']
loc_FH_ci = variables['loc_FH_ci']
loc_VC_ci = variables['loc_VC_ci']

spars_ST_ci = variables['spars_ST_ci']
spars_Horton_ci = variables['spars_Horton_ci']
spars_Stacked_ci = variables['spars_Stacked_ci']
spars_FH_ci = variables['spars_FH_ci']
spars_VC_ci = variables['spars_VC_ci']

original_edges = variables['original_edges']
size_FH = variables['size_FH']
size_VC = variables['size_VC']
edges_FH = variables['edges_FH']
degrees_G = variables['degrees_G']
degrees_G_Stacked = variables['degrees_G_Stacked']
degrees_G_VC = variables['degrees_G_VC']

big_cycles_FH = variables['big_cycles_FH']

loc_ST_values_all = variables['loc_ST_values_all']
loc_FH_values_all = variables['loc_FH_values_all']
spars_ST_values_all = variables['spars_ST_values_all']
spars_FH_values_all = variables['spars_FH_values_all']

weight_Horton = variables['weight_Horton']
weight_FH = variables['weight_FH']
weight_VC = variables['weight_VC']
mean_cycle_length_FH = variables['mean_cycle_length_FH']
mean_cycle_length_VC = variables['mean_cycle_length_VC']

big_cycles_FH_ci = variables['big_cycles_FH_ci']
size_FH_ci = variables['size_FH_ci']
weight_Horton_ci = variables['weight_Horton_ci']
weight_FH_ci = variables['weight_FH_ci']
weight_VC_ci = variables['weight_VC_ci']
mean_cycle_length_FH_ci = variables['mean_cycle_length_FH_ci']
mean_cycle_length_VC_ci = variables['mean_cycle_length_VC_ci']
size_VC_ci = variables['size_VC_ci']

original_edges_ci = variables['original_edges_ci']
edges_FH_ci = variables['edges_FH_ci']
actual_edges_Stacked_ci = variables['actual_edges_Stacked_ci']
edges_VC_ci = variables['edges_VC_ci']
tot_layers_VC_ci = variables['tot_layers_VC_ci']

degrees_G_ci = variables['degrees_G_ci']
degrees_G_Stacked_ci = variables['degrees_G_Stacked_ci']
degrees_G_VC_ci = variables['degrees_G_VC_ci']


# # Define the variables for the general plots: locality, sparsity, num edges div n, avg degree
# loc_VC_straight_sewing_no_overlap = results_straight_sewing_no_overlap['loc_VC']
# loc_VC_straight_sewing_1_overlap = results_straight_sewing_1_overlap['loc_VC']
# loc_VC_straight_sewing_2_overlap = results_straight_sewing_2_overlap['loc_VC']
# loc_VC_straight_sewing_3_overlap = results_straight_sewing_3_overlap['loc_VC']
# loc_VC_straight_6_sewing_1_overlap = results_straight_6_sewing_1_overlap['loc_VC']
# loc_VC_straight_6_sewing_2_overlap = results_straight_6_sewing_2_overlap['loc_VC']
# loc_VC_straight_6_sewing_3_overlap = results_straight_6_sewing_3_overlap['loc_VC']
# loc_VC_triangular_sewing_no_overlap = results_triangular_sewing_no_overlap['loc_VC']
# loc_VC_skew_sewing_no_overlap = results_skew_sewing_no_overlap['loc_VC']

# spars_VC_straight_sewing_no_overlap = results_straight_sewing_no_overlap['spars_VC']
# spars_VC_straight_sewing_1_overlap = results_straight_sewing_1_overlap['spars_VC']
# spars_VC_straight_sewing_2_overlap = results_straight_sewing_2_overlap['spars_VC']
# spars_VC_straight_sewing_3_overlap = results_straight_sewing_3_overlap['spars_VC']
# spars_VC_straight_6_sewing_1_overlap = results_straight_6_sewing_1_overlap['spars_VC']
# spars_VC_straight_6_sewing_2_overlap = results_straight_6_sewing_2_overlap['spars_VC']
# spars_VC_straight_6_sewing_3_overlap = results_straight_6_sewing_3_overlap['spars_VC']
# spars_VC_triangular_sewing_no_overlap = results_triangular_sewing_no_overlap['spars_VC']
# spars_VC_skew_sewing_no_overlap = results_skew_sewing_no_overlap['spars_VC']

# edges_VC_straight_sewing_no_overlap = results_straight_sewing_no_overlap['actual_edges_VC']
# edges_VC_straight_sewing_1_overlap = results_straight_sewing_1_overlap['actual_edges_VC']
# edges_VC_straight_sewing_2_overlap = results_straight_sewing_2_overlap['actual_edges_VC']
# edges_VC_straight_sewing_3_overlap = results_straight_sewing_3_overlap['actual_edges_VC']
# edges_VC_straight_6_sewing_1_overlap = results_straight_6_sewing_1_overlap['actual_edges_VC']
# edges_VC_straight_6_sewing_2_overlap = results_straight_6_sewing_2_overlap['actual_edges_VC']
# edges_VC_straight_6_sewing_3_overlap = results_straight_6_sewing_3_overlap['actual_edges_VC']
# edges_VC_triangular_sewing_no_overlap = results_triangular_sewing_no_overlap['actual_edges_VC']
# edges_VC_skew_sewing_no_overlap = results_skew_sewing_no_overlap['actual_edges_VC']

# degrees_G_VC_straight_sewing_no_overlap = results_straight_sewing_no_overlap['degrees_G_VC']
# degrees_G_VC_straight_sewing_1_overlap = results_straight_sewing_1_overlap['degrees_G_VC']
# degrees_G_VC_straight_sewing_2_overlap = results_straight_sewing_2_overlap['degrees_G_VC']
# degrees_G_VC_straight_sewing_3_overlap = results_straight_sewing_3_overlap['degrees_G_VC']
# degrees_G_VC_straight_6_sewing_1_overlap = results_straight_6_sewing_1_overlap['degrees_G_VC']
# degrees_G_VC_straight_6_sewing_2_overlap = results_straight_6_sewing_2_overlap['degrees_G_VC']
# degrees_G_VC_straight_6_sewing_3_overlap = results_straight_6_sewing_3_overlap['degrees_G_VC']
# degrees_G_VC_triangular_sewing_no_overlap = results_triangular_sewing_no_overlap['degrees_G_VC']
# degrees_G_VC_skew_sewing_no_overlap = results_skew_sewing_no_overlap['degrees_G_VC']

# loc_VC_ci_straight_sewing_no_overlap = results_straight_sewing_no_overlap['loc_VC_ci']
# loc_VC_ci_straight_sewing_1_overlap = results_straight_sewing_1_overlap['loc_VC_ci']
# loc_VC_ci_straight_sewing_2_overlap = results_straight_sewing_2_overlap['loc_VC_ci']
# loc_VC_ci_straight_sewing_3_overlap = results_straight_sewing_3_overlap['loc_VC_ci']
# loc_VC_ci_straight_6_sewing_1_overlap = results_straight_6_sewing_1_overlap['loc_VC_ci']
# loc_VC_ci_straight_6_sewing_2_overlap = results_straight_6_sewing_2_overlap['loc_VC_ci']
# loc_VC_ci_straight_6_sewing_3_overlap = results_straight_6_sewing_3_overlap['loc_VC_ci']
# loc_VC_ci_triangular_sewing_no_overlap = results_triangular_sewing_no_overlap['loc_VC_ci']
# loc_VC_ci_skew_sewing_no_overlap = results_skew_sewing_no_overlap['loc_VC_ci']

# spars_VC_ci_straight_sewing_no_overlap = results_straight_sewing_no_overlap['spars_VC_ci']
# spars_VC_ci_straight_sewing_1_overlap = results_straight_sewing_1_overlap['spars_VC_ci']
# spars_VC_ci_straight_sewing_2_overlap = results_straight_sewing_2_overlap['spars_VC_ci']
# spars_VC_ci_straight_sewing_3_overlap = results_straight_sewing_3_overlap['spars_VC_ci']
# spars_VC_ci_straight_6_sewing_1_overlap = results_straight_6_sewing_1_overlap['spars_VC_ci']
# spars_VC_ci_straight_6_sewing_2_overlap = results_straight_6_sewing_2_overlap['spars_VC_ci']
# spars_VC_ci_straight_6_sewing_3_overlap = results_straight_6_sewing_3_overlap['spars_VC_ci']
# spars_VC_ci_triangular_sewing_no_overlap = results_triangular_sewing_no_overlap['spars_VC_ci']
# spars_VC_ci_skew_sewing_no_overlap = results_skew_sewing_no_overlap['spars_VC_ci']

# edges_VC_ci_straight_sewing_no_overlap = results_straight_sewing_no_overlap['actual_edges_VC_ci']
# edges_VC_ci_straight_sewing_1_overlap = results_straight_sewing_1_overlap['actual_edges_VC_ci']
# edges_VC_ci_straight_sewing_2_overlap = results_straight_sewing_2_overlap['actual_edges_VC_ci']
# edges_VC_ci_straight_sewing_3_overlap = results_straight_sewing_3_overlap['actual_edges_VC_ci']
# edges_VC_ci_straight_6_sewing_1_overlap = results_straight_6_sewing_1_overlap['actual_edges_VC_ci']
# edges_VC_ci_straight_6_sewing_2_overlap = results_straight_6_sewing_2_overlap['actual_edges_VC_ci']
# edges_VC_ci_straight_6_sewing_3_overlap = results_straight_6_sewing_3_overlap['actual_edges_VC_ci']
# edges_VC_ci_triangular_sewing_no_overlap = results_triangular_sewing_no_overlap['actual_edges_VC_ci']
# edges_VC_ci_skew_sewing_no_overlap = results_skew_sewing_no_overlap['actual_edges_VC_ci']

# degrees_G_VC_ci_straight_sewing_no_overlap = results_straight_sewing_no_overlap['degrees_G_VC_ci']
# degrees_G_VC_ci_straight_sewing_1_overlap = results_straight_sewing_1_overlap['degrees_G_VC_ci']
# degrees_G_VC_ci_straight_sewing_2_overlap = results_straight_sewing_2_overlap['degrees_G_VC_ci']
# degrees_G_VC_ci_straight_sewing_3_overlap = results_straight_sewing_3_overlap['degrees_G_VC_ci']
# degrees_G_VC_ci_straight_6_sewing_1_overlap = results_straight_6_sewing_1_overlap['degrees_G_VC_ci']
# degrees_G_VC_ci_straight_6_sewing_2_overlap = results_straight_6_sewing_2_overlap['degrees_G_VC_ci']
# degrees_G_VC_ci_straight_6_sewing_3_overlap = results_straight_6_sewing_3_overlap['degrees_G_VC_ci']
# degrees_G_VC_ci_triangular_sewing_no_overlap = results_triangular_sewing_no_overlap['degrees_G_VC_ci']
# degrees_G_VC_ci_skew_sewing_no_overlap = results_skew_sewing_no_overlap['degrees_G_VC_ci']


# Make lists for log(n), log(n)^2 and log(n)^3
log_L = [math.log(x) for x in L_range]
log_L_sqrd = [math.log(x)**2 for x in L_range]
log_L_cube = [math.log(x)**3 for x in L_range]


# Define the name of the graph that is used
# name_graph = "random expander"
name_graph = "sparse SYK"
# name_graph = "Fermi-Hubbard"
# name_graph = "Cayley"
# name_graph = "Margulis bipartite expander"



# Make all plots

# edges_div_n_FH = [x / y for x, y in zip(edges_FH, L_range)]
# edges_div_n_Stacked = [x / y for x, y in zip(edges_Stacked, L_range)]

# edges_div_n_VC_straight_sewing_no_overlap = [x / y for x, y in zip(edges_VC_straight_sewing_no_overlap, L_range)]
# edges_div_n_VC_straight_sewing_1_overlap = [x / y for x, y in zip(edges_VC_straight_sewing_1_overlap, L_range)]
# edges_div_n_VC_straight_sewing_2_overlap = [x / y for x, y in zip(edges_VC_straight_sewing_2_overlap, L_range)]
# edges_div_n_VC_straight_sewing_3_overlap = [x / y for x, y in zip(edges_VC_straight_sewing_3_overlap, L_range)]
# edges_div_n_VC_triangular_sewing_no_overlap = [x / y for x, y in zip(edges_VC_triangular_sewing_no_overlap, L_range)]
# edges_div_n_VC_skew_sewing_no_overlap = [x / y for x, y in zip(edges_VC_skew_sewing_no_overlap, L_range)]
# edges_div_n_VC_straight_6_sewing_1_overlap = [x / y for x, y in zip(edges_VC_straight_6_sewing_1_overlap, L_range)]
# edges_div_n_VC_straight_6_sewing_2_overlap = [x / y for x, y in zip(edges_VC_straight_6_sewing_2_overlap, L_range)]
# edges_div_n_VC_straight_6_sewing_3_overlap = [x / y for x, y in zip(edges_VC_straight_6_sewing_3_overlap, L_range)]


# Make the general plots
# plot_all_with_errorbars(
#     L_range=L_range,
#     loc_data=[
#         loc_VC_straight_sewing_no_overlap,
#         loc_VC_straight_sewing_1_overlap,
#         loc_VC_straight_sewing_2_overlap,
#         loc_VC_straight_sewing_3_overlap,
#         loc_VC_triangular_sewing_no_overlap,
#         loc_VC_skew_sewing_no_overlap,
#         loc_VC_straight_6_sewing_1_overlap,
#         loc_VC_straight_6_sewing_2_overlap,
#         loc_VC_straight_6_sewing_3_overlap,
#     ],
#     loc_cis=[
#         loc_VC_ci_straight_sewing_no_overlap,
#         loc_VC_ci_straight_sewing_1_overlap,
#         loc_VC_ci_straight_sewing_2_overlap,
#         loc_VC_ci_straight_sewing_3_overlap,
#         loc_VC_ci_triangular_sewing_no_overlap,
#         loc_VC_ci_skew_sewing_no_overlap,
#         loc_VC_ci_straight_6_sewing_1_overlap,
#         loc_VC_ci_straight_6_sewing_2_overlap,
#         loc_VC_ci_straight_6_sewing_3_overlap,
#     ],
#     spars_data=[
#         spars_VC_straight_sewing_no_overlap,
#         spars_VC_straight_sewing_1_overlap,
#         spars_VC_straight_sewing_2_overlap,
#         spars_VC_straight_sewing_3_overlap,
#         spars_VC_triangular_sewing_no_overlap,
#         spars_VC_skew_sewing_no_overlap,
#         spars_VC_straight_6_sewing_1_overlap,
#         spars_VC_straight_6_sewing_2_overlap,
#         spars_VC_straight_6_sewing_3_overlap,
#     ],
#     spars_cis=[
#         spars_VC_ci_straight_sewing_no_overlap,
#         spars_VC_ci_straight_sewing_1_overlap,
#         spars_VC_ci_straight_sewing_2_overlap,
#         spars_VC_ci_straight_sewing_3_overlap,
#         spars_VC_ci_triangular_sewing_no_overlap,
#         spars_VC_ci_skew_sewing_no_overlap,
#         spars_VC_ci_straight_6_sewing_1_overlap,
#         spars_VC_ci_straight_6_sewing_2_overlap,
#         spars_VC_ci_straight_6_sewing_3_overlap,
#     ],
#     edges_div_n_FH, edges_div_n_Stacked,
#     edges_div_n_data=[
#         edges_div_n_VC_straight_sewing_no_overlap,
#         edges_div_n_VC_straight_sewing_1_overlap,
#         edges_div_n_VC_straight_sewing_2_overlap,
#         edges_div_n_VC_straight_sewing_3_overlap,
#         edges_div_n_VC_triangular_sewing_no_overlap,
#         edges_div_n_VC_skew_sewing_no_overlap,
#         edges_div_n_VC_straight_6_sewing_1_overlap,
#         edges_div_n_VC_straight_6_sewing_2_overlap,
#         edges_div_n_VC_straight_6_sewing_3_overlap,
#     ],
#     edges_cis=[
#         edges_VC_ci_straight_sewing_no_overlap,
#         edges_VC_ci_straight_sewing_1_overlap,
#         edges_VC_ci_straight_sewing_2_overlap,
#         edges_VC_ci_straight_sewing_3_overlap,
#         edges_VC_ci_triangular_sewing_no_overlap,
#         edges_VC_ci_skew_sewing_no_overlap,
#         edges_VC_ci_straight_6_sewing_1_overlap,
#         edges_VC_ci_straight_6_sewing_2_overlap,
#         edges_VC_ci_straight_6_sewing_3_overlap,
#     ],
#     degrees_data=[
#         degrees_G_VC_straight_sewing_no_overlap,
#         degrees_G_VC_straight_sewing_1_overlap,
#         degrees_G_VC_straight_sewing_2_overlap,
#         degrees_G_VC_straight_sewing_3_overlap,
#         degrees_G_VC_triangular_sewing_no_overlap,
#         degrees_G_VC_skew_sewing_no_overlap,
#         degrees_G_VC_straight_6_sewing_1_overlap,
#         degrees_G_VC_straight_6_sewing_2_overlap,
#         degrees_G_VC_straight_6_sewing_3_overlap,
#     ],
#     degrees_cis=[
#         degrees_G_VC_ci_straight_sewing_no_overlap,
#         degrees_G_VC_ci_straight_sewing_1_overlap,
#         degrees_G_VC_ci_straight_sewing_2_overlap,
#         degrees_G_VC_ci_straight_sewing_3_overlap,
#         degrees_G_VC_ci_triangular_sewing_no_overlap,
#         degrees_G_VC_ci_skew_sewing_no_overlap,
#         degrees_G_VC_ci_straight_6_sewing_1_overlap,
#         degrees_G_VC_ci_straight_6_sewing_2_overlap,
#         degrees_G_VC_ci_straight_6_sewing_3_overlap,
#     ]
# )



# Plot the locality & sparsity of all algorithms next to each other
plot_L_dependence(
    L_range,
    loc_ST, loc_Stacked, loc_FH, loc_VC,
    spars_ST, spars_Stacked, spars_FH, spars_VC,
    loc_ST_ci, loc_Stacked_ci, loc_FH_ci, loc_VC_ci,
    spars_ST_ci, spars_Stacked_ci, spars_FH_ci, spars_VC_ci,
    num_iterations, name_graph,
    log_L, log_L_sqrd,
    min_loc_Horton, spars_Horton,
    min_loc_Horton_ci, spars_Horton_ci,
    filename = f"Locality & sparsity {name_results}.eps"
)



# # Make the plots

# # diff_deg_G_Stacked = [a - b for a, b in zip(degrees_G_Stacked, degrees_G)]
# # diff_deg_G_VC = [a - b for a, b in zip(degrees_G_VC, degrees_G)]


# # Average degree
# plt.figure(figsize=(8, 6))
# plot_with_ci_custom(L_range, degrees_G, *extract_bounds(degrees_G_ci), label="Average degree of G", color="orange", marker="s", linestyle="--")
# plot_with_ci_custom(L_range, degrees_G_Stacked, *extract_bounds(degrees_G_Stacked_ci), label="Average degree of G_Stacked", color="green", marker="^", linestyle="--")
# # plot_with_ci_custom(L_range, degrees_G_VC, *extract_bounds(degrees_G_VC_ci), label="Average degree of G_VC", color="blue", marker=".", linestyle="--")
# # plot_with_ci_custom(L_range, diff_deg_G_Stacked, *extract_bounds(results_straight_sewing_no_overlap['diff_deg_G_Stacked_ci']), label="G_Stacked - G", color="black", marker="^", linestyle="--")
# # plot_with_ci_custom(L_range, diff_deg_G_VC, *extract_bounds(results_straight_sewing_no_overlap['diff_deg_G_VC_ci']), label="G_VC - G", color="grey", marker="s", linestyle="--")
# finalize_plot("Lattice size (n)", "Average degree of the obtained graph", f"Average degree {name_results}.eps")


# # Num edges
# plt.figure(figsize=(8, 6))
# # plot_with_ci_custom(L_range, edges_FH, *extract_bounds(edges_FH_ci), label=None, color="orange", marker="s", linestyle="--")
# plot_with_ci_custom(L_range**2, edges_Stacked, *extract_bounds(actual_edges_Stacked_ci), label=None, color="green", marker="s", linestyle="--")
# # plot_with_ci_custom(L_range * log_L_cube, edges_VC, *extract_bounds(edges_VC_ci), label=None, color="blue", marker=".", linestyle="--")
# finalize_plot("Lattice size (n)", "Number of edges", f"Num edges Stacked {name_results}.eps")


# # Num edges div n
# edges_div_n_VC, lower, upper = compute_edges_per_n(edges_VC, edges_VC_ci, L_range)
# plt.figure(figsize=(8, 6))
# plot_with_ci_custom(L_range, edges_div_n_VC, lower, upper, label=None, color="blue", marker=".", linestyle="--")
# finalize_plot("Lattice size (n)", "Number of edges divided by the lattice size n", f"Num edges div n VC {name_results}.eps")


# # Total weight
# plt.figure(figsize=(8, 6))
# plot_with_ci_custom(L_range*log_L, weight_FH, *extract_bounds(weight_FH_ci), label=None, color="orange", marker="s", linestyle="--")
# # plot_with_ci_custom(L_range*log_L, weight_Horton, *extract_bounds(weight_Horton_ci), label=None, color="purple", marker="o", linestyle="--")
# # plot_with_ci_custom(L_range*log_L_cube, weight_VC, *extract_bounds(weight_VC_ci), label=None, color="blue", marker=".", linestyle="--")
# finalize_plot("Lattice size (nlog(n))", "Weight of cycle basis", f"Total weight FH {name_results}.eps")


# # Average cycle length
# plt.figure(figsize=(8, 6))
# plot_with_ci_custom(log_L, mean_cycle_length_FH, *extract_bounds(mean_cycle_length_FH_ci), label=None, color="orange", marker="s", linestyle="--")
# # plot_with_ci_custom(L_range, mean_cycle_length_VC, *extract_bounds(mean_cycle_length_VC_ci), label=None, color="blue", marker=".", linestyle="--")
# finalize_plot("Lattice size (log(n))", "Average cycle length", f"Average cycle length FH {name_results}.eps")


# # Size of the cycle basis
# plt.figure(figsize=(8, 6))
# plot_with_ci_custom(L_range, size_FH, *extract_bounds(size_FH_ci), label=None, color="orange", marker="s", linestyle="--")
# # plot_with_ci_custom(L_range, size_VC, *extract_bounds(size_VC_ci), label=None, color="blue", marker=".", linestyle="--")
# finalize_plot("Lattice size (n)", "Number of cycles", f"Size FH {name_results}.eps")


# # Weight div size
# weight_div_size_FH = [x / y for x, y in zip(weight_FH, size_FH)]
# weight_div_size_VC = [x / y for x, y in zip(weight_VC, size_VC)]

# # Step 1: Extract the lower and upper bounds for weight and size confidence intervals (FH)
# weight_FH_lower, weight_FH_upper = extract_bounds(weight_FH_ci)
# size_FH_lower, size_FH_upper = extract_bounds(size_FH_ci)

# # Step 2: Perform element-wise division for the lower bounds and upper bounds (FH)
# weight_div_size_FH_lower = []
# weight_div_size_FH_upper = []

# for w_lower, w_upper, s_lower, s_upper in zip(weight_FH_lower, weight_FH_upper, size_FH_lower, size_FH_upper):
#     weight_div_size_FH_lower.append(w_lower / s_lower)
#     weight_div_size_FH_upper.append(w_upper / s_upper)

# # Step 3: Combine the results into the final confidence interval for weight/size (FH)
# weight_div_size_FH_ci = list(zip(weight_div_size_FH_lower, weight_div_size_FH_upper))

# # Step 4: Extract the lower and upper bounds for weight and size confidence intervals (VC)
# weight_VC_lower, weight_VC_upper = extract_bounds(weight_VC_ci)
# size_VC_lower, size_VC_upper = extract_bounds(size_VC_ci)

# # Step 5: Perform element-wise division for the lower bounds and upper bounds (VC)
# weight_div_size_VC_lower = []
# weight_div_size_VC_upper = []

# for w_lower, w_upper, s_lower, s_upper in zip(weight_VC_lower, weight_VC_upper, size_VC_lower, size_VC_upper):
#     weight_div_size_VC_lower.append(w_lower / s_lower)
#     weight_div_size_VC_upper.append(w_upper / s_upper)

# # Step 6: Combine the results into the final confidence interval for weight/size (VC)
# weight_div_size_VC_ci = list(zip(weight_div_size_VC_lower, weight_div_size_VC_upper))


# # Size of the cycle basis
# plt.figure(figsize=(8, 6))
# plot_with_ci_custom(L_range, weight_div_size_FH, *extract_bounds(weight_div_size_FH_ci), label="Weight div size", color="black", marker="s", linestyle="--")
# plot_with_ci_custom(L_range, mean_cycle_length_FH, *extract_bounds(mean_cycle_length_FH_ci), label="Average cycle length", color="orange", marker="s", linestyle="--")

# # plot_with_ci_custom(L_range, weight_div_size_VC, *extract_bounds(weight_div_size_VC_ci), label="Weight div size", color="black", marker=".", linestyle="--")
# # plot_with_ci_custom(L_range, mean_cycle_length_VC, *extract_bounds(mean_cycle_length_VC_ci), label="Average cycle length", color="blue", marker="s", linestyle="--")
# finalize_plot("Lattice size (n)", "Cycle length", f"Weight div size FH {name_results}.eps")








# Old plots

# # Plot the total weight of the cycle basis
# plt.figure(figsize=(8, 6))
# # plt.plot(L_range*log_L, weight_FH, label="Weight of CB FH", marker="s", linestyle="--", color="orange")
# plt.plot(L_range*log_L_cube, weight_VC, label="Weight of CB VC", marker=".", linestyle="--", color="blue")

# # plt.plot(L_range, L_range * log_L, label="nlog(n)", color="black")
# plt.xlabel("Lattice size (nlog(n)^3)")
# plt.ylabel("Weight of the cycle basis")
# # plt.title(f"Total weight of the cycle basis, using a {name_graph} graph")
# # plt.legend()
# plt.grid()
# plt.show()


# # Plot for each L the length of the average cycle in the cycle basis
# plt.figure(figsize=(8, 6))
# # plt.plot(log_L, mean_cycle_length_FH, label="Average cycle length for FH", marker="s", linestyle="--", color="orange")
# plt.plot(L_range, mean_cycle_length_VC, label="Average cycle length for VC", marker=".", linestyle="--", color="blue")

# # plt.plot(L_range, L_range, label="n", color="grey")
# # plt.plot(L_range, log_L, label="log(n)", color="black")
# # plt.ylim(ymin=2.9975, ymax=3.0005)
# plt.xlabel("Lattice size (n)")
# plt.ylabel("Cycle length")
# # plt.title(f"Average length of the cycles in the cycle basis, using a {name_graph} graph")
# # plt.legend()
# plt.grid()
# plt.show()


# # Plot the number of cycles with (non-)constant length
# plt.figure(figsize=(8, 6))
# # plt.plot(L_range, constant_cycle_count, marker="s", linestyle="--", label="constant length", color="blue")
# plt.plot(L_range, non_constant_cycle_count, marker="s", linestyle="--", label="cycles with non-constant length", color="red")
# # plt.plot(L_range, bound_loc, label="log(n)", color="black")
# plt.xlabel("Lattice size (n)")
# plt.ylabel("Number of cycles")
# plt.title(f"Number of cycles with non-constant length for FH, using a {name_graph} graph")
# plt.legend()
# plt.grid()
# plt.show()

# # Plot the number of cycles with length > threshold in the cycle basis obtained by Freedman-Hastings

# # Extract the lower and upper bounds of the confidence intervals
# big_cycles_FH_lower = [ci[0] for ci in big_cycles_FH_ci]
# big_cycles_FH_upper = [ci[1] for ci in big_cycles_FH_ci]

# # Plot the number of big cycles with the confidence interval as a shaded region
# plt.figure(figsize=(8, 6))
# plt.plot(L_range * log_L, big_cycles_FH, label="Number of big cycles", marker="s", linestyle="--", color="orange")
# # plt.fill_between(L_range * log_L, big_cycles_FH_lower, big_cycles_FH_upper, color="orange", alpha=0.3)

# # plt.plot(L_range*log_L, L_range*log_L, label="nlog(n)", color="black")
# # plt.plot(L_range*log_L, L_range, label="n", color="grey")

# # Customize the plot
# plt.xlabel("Lattice size (nlog(n))")
# plt.ylabel("Number of cycles")
# # plt.title(f"Number of cycles with length > {threshold_length} for FH, using a {name_graph} graph")
# # plt.legend()
# plt.grid()
# plt.show()



# Use curve fit to fit functions to the data

# Define the model functions
def func_n(n, a, b):
    return a*n + b

def func_log(n, a, b):
    return a * np.log(n) + b

def func_log_sqrd(n, a, b):
    return a * np.log(n)**2 + b

def func_nlog(n, a, b):
    return a * n* np.log(n) + b

def func_nlog3(n, a, b):
    return a * n* np.log(n)**3 + b

def func_nloga(n, a, b, c):
    return a * n* np.log(n)**c + b

def func_n2(n, a, b):
    return a * n**2 + b


# Create the mask for the fit
threshold_fit_L = 100

mask_L = L_range > threshold_fit_L

# Get the correct x-values for plotting and fitting
log_L_array = np.array(log_L)
log_L_sqrd_array = np.array(log_L_sqrd)
log_L_cube_array = np.array(log_L_cube)

L_full = L_range
log_L_full = log_L_array
log_L_sqrd_full = log_L_sqrd_array

L_fit = L_range[mask_L]
log_L_fit = log_L_array[mask_L]
log_L_sqrd_fit = log_L_sqrd_array[mask_L]
log_L_cube_fit = log_L_cube_array[mask_L]

x_fit = L_fit * log_L_fit
x_cube_fit = L_fit * log_L_cube_fit



# # Perform the fit for the big cycles
# big_cycles_FH_array = np.array(big_cycles_FH)
# big_cycles_FH_ci_array = np.array(big_cycles_FH_ci)

# big_cycles_FH_fit = big_cycles_FH_array[mask_L]
# big_cycles_FH_ci_fit = big_cycles_FH_ci_array[mask_L]

# big_cycles_FH_lower_fit = [ci[0] for ci in big_cycles_FH_ci_fit]
# big_cycles_FH_upper_fit = [ci[1] for ci in big_cycles_FH_ci_fit]


# # Perform the curve fit for L > 80
# params_big_cycles_n_FH, cov_big_cycles_n_FH = curve_fit(func_n, L_fit, big_cycles_FH_fit)  # big_cycles vs n
# params_big_cycles_nlog_FH, cov_big_cycles_nlog_FH = curve_fit(func_nlog, L_fit, big_cycles_FH_fit)  # big_cycles vs nlog(n)

# # Extract fitted values
# a_bigcycles_n_FH = params_big_cycles_n_FH[0]
# b_bigcycles_n_FH = params_big_cycles_n_FH[1]

# a_bigcycles_nlog_FH = params_big_cycles_nlog_FH[0]
# b_bigcycles_nlog_FH = params_big_cycles_nlog_FH[1]

# # Calculate standard deviations
# std_a_bigcycles_n = np.sqrt(np.diag(cov_big_cycles_n_FH))[0]
# std_b_bigcycles_n = np.sqrt(np.diag(cov_big_cycles_n_FH))[1]

# std_a_bigcycles_nlog = np.sqrt(np.diag(cov_big_cycles_nlog_FH))[0]
# std_b_bigcycles_nlog = np.sqrt(np.diag(cov_big_cycles_nlog_FH))[1]

# # Format the parameters and their standard deviations
# n_bigcycles_label = f"n: a = {a_bigcycles_n_FH:.1f} ± {std_a_bigcycles_n:.1f}, b = {b_bigcycles_n_FH:.1f} ± {std_b_bigcycles_n:.1f}"
# nlog_bigcycles_label = f"nlog: a = {a_bigcycles_nlog_FH:.1f} ± {std_a_bigcycles_nlog:.1f}, b = {b_bigcycles_nlog_FH:.1f} ± {std_b_bigcycles_nlog:.1f}"


# # Create the interactive plot for the big cycles
# fig1, ax1 = plt.subplots(figsize=(8, 6))

# # Plot big cycles vs nlog(n)
# ax1.plot(x_fit, big_cycles_FH_fit, label="Number of big cycles", marker="s", linestyle="--", color="orange")

# # Plot the confidence intervals
# ax1.fill_between(x_fit, big_cycles_FH_lower_fit, big_cycles_FH_upper_fit, color='orange', alpha=0.3)

# # Plot the fitted curves n and nlog(n)
# ax1.plot(x_fit, func_n(L_fit, a_bigcycles_n_FH, b_bigcycles_n_FH), label=f"Fitted curve {n_bigcycles_label}", color='blue')
# ax1.plot(x_fit, func_nlog(L_fit, a_bigcycles_nlog_FH, b_bigcycles_nlog_FH), label=f"Fitted curve {nlog_bigcycles_label}", color='black')

# ax1.set_xlabel('Lattice size (nlog(n))')
# ax1.set_ylabel('Number of cycles')
# ax1.set_title(f"Number of cycles with length > {threshold_length} of FH with fits for L > {threshold_fit_L}, using a {name_graph} graph")

# ax1.legend()
# ax1.grid()

# # Save the plot as an interactive HTML file
# save_path_big_cycles = r"C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Figuren\Data-analyse\Interactieve plots\interactive_big_cycles_FH_plot_expander_p_thresh_deg_4_straight_sewing.html"

# mpld3.save_html(fig1, save_path_big_cycles)

# plt.show()



# # Do the same for the total weight
# # # For FH
# # weight_FH_array = np.array(weight_FH)
# # weight_FH_ci_array = np.array(weight_FH_ci)

# # weight_FH_fit = weight_FH_array[mask_L]
# # weight_FH_ci_fit = weight_FH_ci_array[mask_L]

# # weight_FH_lower_fit = [ci[0] for ci in weight_FH_ci_fit]
# # weight_FH_upper_fit = [ci[1] for ci in weight_FH_ci_fit]

# # For VC
# weight_VC_array = np.array(weight_VC)
# weight_VC_ci_array = np.array(weight_VC_ci)

# weight_VC_fit = weight_VC_array[mask_L]
# weight_VC_ci_fit = weight_VC_ci_array[mask_L]

# weight_VC_lower_fit = [ci[0] for ci in weight_VC_ci_fit]
# weight_VC_upper_fit = [ci[1] for ci in weight_VC_ci_fit]

# # Perform curve fitting using nlog(n) and n^2
# # params_weight_nlog_FH, cov_weight_nlog_FH = curve_fit(func_nlog, L_fit, weight_FH_fit)
# # params_weight_n2_FH, cov_weight_n2_FH = curve_fit(func_n2, L_fit, weight_FH_fit)

# params_weight_nlog3_VC, cov_weight_nlog3_VC = curve_fit(func_nlog3, L_fit, weight_VC_fit)
# params_weight_n2_VC, cov_weight_n2_VC = curve_fit(func_n2, L_fit, weight_VC_fit)

# # Extract fitted values
# # a_weight_nlog_FH = params_weight_nlog_FH[0]
# # b_weight_nlog_FH = params_weight_nlog_FH[1]

# # a_weight_n2_FH = params_weight_n2_FH[0]
# # b_weight_n2_FH = params_weight_n2_FH[1]

# a_weight_nlog3_VC = params_weight_nlog3_VC[0]
# b_weight_nlog3_VC = params_weight_nlog3_VC[1]

# a_weight_n2_VC = params_weight_n2_VC[0]
# b_weight_n2_VC = params_weight_n2_VC[1]

# # Calculate standard deviations
# # std_a_weight_nlog_FH = np.sqrt(np.diag(cov_weight_nlog_FH))[0]
# # std_b_weight_nlog_FH = np.sqrt(np.diag(cov_weight_nlog_FH))[1]

# # std_a_weight_n2_FH = np.sqrt(np.diag(cov_weight_n2_FH))[0]
# # std_b_weight_n2_FH = np.sqrt(np.diag(cov_weight_n2_FH))[1]

# std_a_weight_nlog3_VC = np.sqrt(np.diag(cov_weight_nlog3_VC))[0]
# std_b_weight_nlog3_VC = np.sqrt(np.diag(cov_weight_nlog3_VC))[1]

# std_a_weight_n2_VC = np.sqrt(np.diag(cov_weight_n2_VC))[0]
# std_b_weight_n2_VC = np.sqrt(np.diag(cov_weight_n2_VC))[1]


# # Format the parameters and their standard deviations
# # nlog_weight_label_FH = f"nlog: a = {a_weight_nlog_FH:.1f} ± {std_a_weight_nlog_FH:.1f}, b = {b_weight_nlog_FH:.1f} ± {std_b_weight_nlog_FH:.1f}"
# # n2_weight_label_FH = f"n^2: a = {a_weight_n2_FH:.5f} ± {std_a_weight_n2_FH:.5f}, b = {b_weight_n2_FH:.5f} ± {std_b_weight_n2_FH:.5f}"

# nlog3_weight_label_VC = f"nlog(n)^3: a = {a_weight_nlog3_VC:.1f} ± {std_a_weight_nlog3_VC:.1f}, b = {b_weight_nlog3_VC:.1f} ± {std_b_weight_nlog3_VC:.1f}"
# n2_weight_label_VC = f"n^2: a = {a_weight_n2_VC:.1f} ± {std_a_weight_n2_VC:.1f}, b = {b_weight_n2_VC:.1f} ± {std_b_weight_n2_VC:.1f}"

# # Create the interactive plot
# fig2, ax2 = plt.subplots(figsize=(8, 6))

# # Plot the weight vs. nlog(n)
# # ax2.plot(x_fit, weight_FH_fit, label="Weight of FH", marker="s", linestyle="--", color="orange")
# ax2.plot(x_cube_fit, weight_VC_fit, label="Weight of VC", marker=".", linestyle="--", color="blue")

# # Plot the confidence intervals
# # ax2.fill_between(x_fit, weight_FH_lower_fit, weight_FH_upper_fit, color='orange', alpha=0.3)
# ax2.fill_between(x_cube_fit, weight_VC_lower_fit, weight_VC_upper_fit, color='blue', alpha=0.3)

# # Plot the fitted curves
# # ax2.plot(x_fit, func_nlog(L_fit, a_weight_nlog_FH, b_weight_nlog_FH), label=f"Fitted curve {nlog_weight_label_FH}", color='black')
# # ax2.plot(x_fit, func_n2(L_fit, a_weight_n2_FH, b_weight_n2_FH), label=f"Fitted curve {n2_weight_label_FH}", color='red')

# ax2.plot(x_cube_fit, func_nlog3(L_fit, a_weight_nlog3_VC, b_weight_nlog3_VC), label=f"Fitted curve {nlog3_weight_label_VC}", color='black')
# ax2.plot(x_cube_fit, func_n2(L_fit, a_weight_n2_VC, b_weight_n2_VC), label=f"Fitted curve {n2_weight_label_VC}", color='grey')

# ax2.set_xlabel('Lattice size (nlog(n)^3)')
# ax2.set_ylabel('Weight of the cycle basis')
# ax2.set_title(f"Weight of the cycle basis with fits for L > {threshold_fit_L}, using a {name_graph} graph")

# ax2.legend()
# ax2.grid()

# # Save the second plot as an interactive HTML file
# save_path_weight = r"C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Figuren\Data-analyse\Interactieve plots\interactive_weight_VC_expander_p_thresh_deg_4_straight_sewing_no_overlap.html"

# mpld3.save_html(fig2, save_path_weight)

# plt.show()




# # Do the same for average cycle length
# # For FH
# mean_cycle_length_FH_array = np.array(mean_cycle_length_FH)
# mean_cycle_length_FH_ci_array = np.array(mean_cycle_length_FH_ci)

# mean_cycle_length_FH_fit = mean_cycle_length_FH_array[mask_L]
# mean_cycle_length_FH_ci_fit = mean_cycle_length_FH_ci_array[mask_L]

# mean_cycle_length_FH_lower_fit = [ci[0] for ci in mean_cycle_length_FH_ci_fit]
# mean_cycle_length_FH_upper_fit = [ci[1] for ci in mean_cycle_length_FH_ci_fit]

# # print(mean_cycle_length_FH_ci)

# # # For VC
# # mean_cycle_length_VC_array = np.array(mean_cycle_length_VC)
# # mean_cycle_length_VC_ci_array = np.array(mean_cycle_length_VC_ci)

# # mean_cycle_length_VC_fit = mean_cycle_length_VC_array[mask_L]
# # mean_cycle_length_VC_ci_fit = mean_cycle_length_VC_ci_array[mask_L]

# # mean_cycle_length_VC_lower_fit = [ci[0] for ci in mean_cycle_length_VC_ci_fit]
# # mean_cycle_length_VC_upper_fit = [ci[1] for ci in mean_cycle_length_VC_ci_fit]


# # Perform curve fitting for the average cycle length using n and nlog(n)
# params_meanlength_nlog_FH, cov_meanlength_nlog_FH = curve_fit(func_nlog, L_fit, mean_cycle_length_FH_fit)
# params_meanlength_n_FH, cov_meanlength_n_FH = curve_fit(func_n, L_fit, mean_cycle_length_FH_fit)
# params_meanlength_log_FH, cov_meanlength_log_FH = curve_fit(func_log, L_fit, mean_cycle_length_FH_fit)
# params_meanlength_log_sqrd_FH, cov_meanlength_log_sqrd_FH = curve_fit(func_log_sqrd, L_fit, mean_cycle_length_FH_fit)

# # params_meanlength_nlog_VC, cov_meanlength_nlog_VC = curve_fit(func_nlog, L_fit, mean_cycle_length_VC_fit)
# # params_meanlength_n_VC, cov_meanlength_n_VC = curve_fit(func_n, L_fit, mean_cycle_length_VC_fit)
# # params_meanlength_log_VC, cov_meanlength_log_VC = curve_fit(func_log, L_fit, mean_cycle_length_VC_fit)
# # params_meanlength_log_sqrd_VC, cov_meanlength_log_sqrd_VC = curve_fit(func_log_sqrd, L_fit, mean_cycle_length_VC_fit)


# # Extract fitted values
# a_meanlength_nlog_FH = params_meanlength_nlog_FH[0]
# b_meanlength_nlog_FH = params_meanlength_nlog_FH[1]

# a_meanlength_n_FH = params_meanlength_n_FH[0]
# b_meanlength_n_FH = params_meanlength_n_FH[1]

# a_meanlength_log_FH = params_meanlength_log_FH[0]
# b_meanlength_log_FH = params_meanlength_log_FH[1]

# a_meanlength_log_sqrd_FH = params_meanlength_log_sqrd_FH[0]
# b_meanlength_log_sqrd_FH = params_meanlength_log_sqrd_FH[1]

# # a_meanlength_nlog_VC = params_meanlength_nlog_VC[0]
# # b_meanlength_nlog_VC = params_meanlength_nlog_VC[1]

# # a_meanlength_n_VC = params_meanlength_n_VC[0]
# # b_meanlength_n_VC = params_meanlength_n_VC[1]

# # a_meanlength_log_VC = params_meanlength_log_VC[0]
# # b_meanlength_log_VC = params_meanlength_log_VC[1]

# # a_meanlength_log_sqrd_VC = params_meanlength_log_sqrd_VC[0]
# # b_meanlength_log_sqrd_VC = params_meanlength_log_sqrd_VC[1]


# # Calculate standard deviations
# std_a_meanlength_nlog_FH = np.sqrt(np.diag(cov_meanlength_nlog_FH))[0]
# std_b_meanlength_nlog_FH = np.sqrt(np.diag(cov_meanlength_nlog_FH))[1]

# std_a_meanlength_n_FH = np.sqrt(np.diag(cov_meanlength_n_FH))[0]
# std_b_meanlength_n_FH = np.sqrt(np.diag(cov_meanlength_n_FH))[1]

# std_a_meanlength_log_FH = np.sqrt(np.diag(cov_meanlength_log_FH))[0]
# std_b_meanlength_log_FH = np.sqrt(np.diag(cov_meanlength_log_FH))[1]

# std_a_meanlength_log_sqrd_FH = np.sqrt(np.diag(cov_meanlength_log_sqrd_FH))[0]
# std_b_meanlength_log_sqrd_FH = np.sqrt(np.diag(cov_meanlength_log_sqrd_FH))[1]

# # std_a_meanlength_nlog_VC = np.sqrt(np.diag(cov_meanlength_nlog_VC))[0]
# # std_b_meanlength_nlog_VC = np.sqrt(np.diag(cov_meanlength_nlog_VC))[1]

# # std_a_meanlength_n_VC = np.sqrt(np.diag(cov_meanlength_n_VC))[0]
# # std_b_meanlength_n_VC = np.sqrt(np.diag(cov_meanlength_n_VC))[1]

# # std_a_meanlength_log_VC = np.sqrt(np.diag(cov_meanlength_log_VC))[0]
# # std_b_meanlength_log_VC = np.sqrt(np.diag(cov_meanlength_log_VC))[1]

# # std_a_meanlength_log_sqrd_VC = np.sqrt(np.diag(cov_meanlength_log_sqrd_VC))[0]
# # std_b_meanlength_log_sqrd_VC = np.sqrt(np.diag(cov_meanlength_log_sqrd_VC))[1]


# # Format the parameters and their standard deviations
# nlog_meanlength_label_FH = f"nlog: a = {a_meanlength_nlog_FH:.1f} ± {std_a_meanlength_nlog_FH:.1f}, b = {b_meanlength_nlog_FH:.1f} ± {std_b_meanlength_nlog_FH:.1f}"
# n_meanlength_label_FH = f"n: a = {a_meanlength_n_FH:.1f} ± {std_a_meanlength_n_FH:.1f}, b = {b_meanlength_n_FH:.1f} ± {std_b_meanlength_n_FH:.1f}"
# log_meanlength_label_FH = f"log(n): a = {a_meanlength_log_FH:.1f} ± {std_a_meanlength_log_FH:.1f}, b = {b_meanlength_log_FH:.1f} ± {std_b_meanlength_log_FH:.1f}"
# log_sqrd_meanlength_label_FH = f"log(n)^2: a = {a_meanlength_log_sqrd_FH:.1f} ± {std_a_meanlength_log_sqrd_FH:.1f}, b = {b_meanlength_log_sqrd_FH:.1f} ± {std_b_meanlength_log_sqrd_FH:.1f}"

# # nlog_meanlength_label_VC = f"nlog: a = {a_meanlength_nlog_VC:.1f} ± {std_a_meanlength_nlog_VC:.1f}, b = {b_meanlength_nlog_VC:.1f} ± {std_b_meanlength_nlog_VC:.1f}"
# # n_meanlength_label_VC = f"n: a = {a_meanlength_n_VC:.1f} ± {std_a_meanlength_n_VC:.1f}, b = {b_meanlength_n_VC:.1f} ± {std_b_meanlength_n_VC:.1f}"
# # log_meanlength_label_VC = f"log(n): a = {a_meanlength_log_VC:.1f} ± {std_a_meanlength_log_VC:.1f}, b = {b_meanlength_log_VC:.1f} ± {std_b_meanlength_log_VC:.1f}"
# # log_sqrd_meanlength_label_VC = f"log(n)^2: a = {a_meanlength_log_sqrd_VC:.1f} ± {std_a_meanlength_log_sqrd_VC:.1f}, b = {b_meanlength_log_sqrd_VC:.1f} ± {std_b_meanlength_log_sqrd_VC:.1f}"


# # Create the interactive plot
# fig3, ax3 = plt.subplots(figsize=(8, 6))

# # Plot the average cycle length vs. n
# ax3.plot(L_fit, mean_cycle_length_FH_fit, label="Average cycle length of FH", marker="s", linestyle="--", color="orange")
# # ax3.plot(L_fit, mean_cycle_length_VC_fit, label="Average cycle length of VC", marker=".", linestyle="--", color="blue")

# # Plot the confidence intervals
# ax3.fill_between(L_fit, mean_cycle_length_FH_lower_fit, mean_cycle_length_FH_upper_fit, color='orange', alpha=0.3)
# # ax3.fill_between(L_fit, mean_cycle_length_VC_lower_fit, mean_cycle_length_VC_upper_fit, color='blue', alpha=0.3)

# # Plot the fitted curves
# ax3.plot(L_fit, func_nlog(L_fit, a_meanlength_nlog_FH, b_meanlength_nlog_FH), label=f"Fitted curve {nlog_meanlength_label_FH}", color='black')
# ax3.plot(L_fit, func_n(L_fit, a_meanlength_n_FH, b_meanlength_n_FH), label=f"Fitted curve {n_meanlength_label_FH}", color='grey')
# ax3.plot(L_fit, func_log(L_fit, a_meanlength_log_FH, b_meanlength_log_FH), label=f"Fitted curve {log_meanlength_label_FH}", color='blue')
# ax3.plot(L_fit, func_log_sqrd(L_fit, a_meanlength_log_sqrd_FH, b_meanlength_log_sqrd_FH), label=f"Fitted curve {log_sqrd_meanlength_label_FH}", color='red')

# ax3.set_xlabel('Lattice size (n)')
# ax3.set_ylabel('Cycle length')
# ax3.set_title(f"Average length of cycles in the cycle basis with fits for L > {threshold_fit_L}, using a {name_graph} graph")

# ax3.legend()
# ax3.grid()

# # Save the second plot as an interactive HTML file
# save_path_meanlength = r"C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Figuren\Data-analyse\Interactieve plots\interactive_meanlength_FH_plot_expander_p_thresh_deg_4_straight_sewing.html"

# mpld3.save_html(fig3, save_path_meanlength)

# plt.show()




# # Do the same for the sparsity
# # For FH
# spars_FH_array = np.array(spars_FH)
# spars_FH_ci_array = np.array(spars_FH_ci)

# spars_FH_fit = spars_FH_array[mask_L]
# spars_FH_ci_fit = spars_FH_ci_array[mask_L]

# spars_FH_lower_fit = [ci[0] for ci in spars_FH_ci_fit]
# spars_FH_upper_fit = [ci[1] for ci in spars_FH_ci_fit]

# # # For VC
# # spars_VC_array = np.array(spars_VC)
# # spars_VC_ci_array = np.array(spars_VC_ci)

# # spars_VC_fit = spars_VC_array[mask_L]
# # spars_VC_ci_fit = spars_VC_ci_array[mask_L]

# # spars_VC_lower_fit = [ci[0] for ci in spars_VC_ci_fit]
# # spars_VC_upper_fit = [ci[1] for ci in spars_VC_ci_fit]


# # Perform curve fitting for the sparsity using log(n)^2, n and nlog(n)
# params_spars_log_sqrd_FH, cov_spars_log_sqrd_FH = curve_fit(func_log_sqrd, L_fit, spars_FH_fit)
# params_spars_n_FH, cov_spars_n_FH = curve_fit(func_n, L_fit, spars_FH_fit)
# params_spars_nlog_FH, cov_spars_nlog_FH = curve_fit(func_nlog, L_fit, spars_FH_fit)

# # params_spars_log_sqrd_VC, cov_spars_log_sqrd_VC = curve_fit(func_log_sqrd, L_fit, spars_VC_fit)
# # params_spars_n_VC, cov_spars_n_VC = curve_fit(func_n, L_fit, spars_VC_fit)
# # params_spars_nlog_VC, cov_spars_nlog_VC = curve_fit(func_nlog, L_fit, spars_VC_fit)


# # Extract fitted values
# a_spars_log_sqrd_FH = params_spars_log_sqrd_FH[0]
# b_spars_log_sqrd_FH = params_spars_log_sqrd_FH[1]

# a_spars_n_FH = params_spars_n_FH[0]
# b_spars_n_FH = params_spars_n_FH[1]

# a_spars_nlog_FH = params_spars_nlog_FH[0]
# b_spars_nlog_FH = params_spars_nlog_FH[1]

# # a_spars_log_sqrd_VC = params_spars_log_sqrd_VC[0]
# # b_spars_log_sqrd_VC = params_spars_log_sqrd_VC[1]

# # a_spars_n_VC = params_spars_n_VC[0]
# # b_spars_n_VC = params_spars_n_VC[1]

# # a_spars_nlog_VC = params_spars_nlog_VC[0]
# # b_spars_nlog_VC = params_spars_nlog_VC[1]


# # Calculate standard deviations
# std_a_spars_log_sqrd_FH = np.sqrt(np.diag(cov_spars_log_sqrd_FH))[0]
# std_b_spars_log_sqrd_FH = np.sqrt(np.diag(cov_spars_log_sqrd_FH))[1]

# std_a_spars_n_FH = np.sqrt(np.diag(cov_spars_n_FH))[0]
# std_b_spars_n_FH = np.sqrt(np.diag(cov_spars_n_FH))[1]

# std_a_spars_nlog_FH = np.sqrt(np.diag(cov_spars_nlog_FH))[0]
# std_b_spars_nlog_FH = np.sqrt(np.diag(cov_spars_nlog_FH))[1]

# # std_a_spars_log_sqrd_VC = np.sqrt(np.diag(cov_spars_log_sqrd_VC))[0]
# # std_b_spars_log_sqrd_VC = np.sqrt(np.diag(cov_spars_log_sqrd_VC))[1]

# # std_a_spars_n_VC = np.sqrt(np.diag(cov_spars_n_VC))[0]
# # std_b_spars_n_VC = np.sqrt(np.diag(cov_spars_n_VC))[1]

# # std_a_spars_nlog_VC = np.sqrt(np.diag(cov_spars_nlog_VC))[0]
# # std_b_spars_nlog_VC = np.sqrt(np.diag(cov_spars_nlog_VC))[1]


# # Format the parameters and their standard deviations
# log_sqrd_spars_label_FH = f"log^2: a = {a_spars_log_sqrd_FH:.1f} ± {std_a_spars_log_sqrd_FH:.1f}, b = {b_spars_log_sqrd_FH:.1f} ± {std_b_spars_log_sqrd_FH:.1f}"
# n_spars_label_FH = f"n: a = {a_spars_n_FH:.1f} ± {std_a_spars_n_FH:.1f}, b = {b_spars_n_FH:.1f} ± {std_b_spars_n_FH:.1f}"
# nlog_spars_label_FH = f"nlog(n): a = {a_spars_nlog_FH:.1f} ± {std_a_spars_nlog_FH:.1f}, b = {b_spars_nlog_FH:.1f} ± {std_b_spars_nlog_FH:.1f}"

# # log_sqrd_spars_label_VC = f"log^2: a = {a_spars_log_sqrd_VC:.1f} ± {std_a_spars_log_sqrd_VC:.1f}, b = {b_spars_log_sqrd_VC:.1f} ± {std_b_spars_log_sqrd_VC:.1f}"
# # n_spars_label_VC = f"n: a = {a_spars_n_VC:.1f} ± {std_a_spars_n_VC:.1f}, b = {b_spars_n_VC:.1f} ± {std_b_spars_n_VC:.1f}"
# # nlog_spars_label_VC = f"nlog(n): a = {a_spars_nlog_VC:.1f} ± {std_a_spars_nlog_VC:.1f}, b = {b_spars_nlog_VC:.1f} ± {std_b_spars_nlog_VC:.1f}"


# # Create the interactive plot
# fig4, ax4 = plt.subplots(figsize=(8, 6))

# # Plot the sparsity vs. n
# ax4.plot(log_L_sqrd_fit, spars_FH_fit, label="Maximal sparsity of FH", marker="s", linestyle="--", color="orange")
# # ax4.plot(log_L_sqrd_fit, spars_VC_fit, label="Maximal sparsity of VC", marker=".", linestyle="--", color="blue")

# # Plot the confidence intervals
# ax4.fill_between(log_L_sqrd_fit, spars_FH_lower_fit, spars_FH_upper_fit, color='orange', alpha=0.3)
# # ax4.fill_between(log_L_sqrd_fit, spars_VC_lower_fit, spars_VC_upper_fit, color='blue', alpha=0.3)

# # Plot the fitted curves
# ax4.plot(log_L_sqrd_fit, func_log_sqrd(L_fit, a_spars_log_sqrd_FH, b_spars_log_sqrd_FH), label=f"Fitted curve {log_sqrd_spars_label_FH}", color='black')
# ax4.plot(log_L_sqrd_fit, func_n(L_fit, a_spars_n_FH, b_spars_n_FH), label=f"Fitted curve {n_spars_label_FH}", color='red')
# ax4.plot(log_L_sqrd_fit, func_nlog(L_fit, a_spars_nlog_FH, b_spars_nlog_FH), label=f"Fitted curve {nlog_spars_label_FH}", color='blue')

# ax4.set_xlabel('Lattice size (log(n)^2)')
# ax4.set_ylabel('Sparsity of FH')
# ax4.set_title(f"Maximal sparsity of FH with fits for L > {threshold_fit_L}, using a {name_graph} graph")

# ax4.legend()
# ax4.grid()

# # Save the second plot as an interactive HTML file
# save_path_spars = r"C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Figuren\Data-analyse\Interactieve plots\interactive_sparsity_FH_plot_expander_p_thresh_deg_4_straight_sewing.html"

# mpld3.save_html(fig4, save_path_spars)

# plt.show()



# # Do the same for the locality
# # For FH
# loc_FH_array = np.array(loc_FH)
# loc_FH_ci_array = np.array(loc_FH_ci)

# loc_FH_fit = loc_FH_array[mask_L]
# loc_FH_ci_fit = loc_FH_ci_array[mask_L]

# loc_FH_lower_fit = [ci[0] for ci in loc_FH_ci_fit]
# loc_FH_upper_fit = [ci[1] for ci in loc_FH_ci_fit]

# # # For VC
# # loc_VC_array = np.array(loc_VC)
# # loc_VC_ci_array = np.array(loc_VC_ci)

# # loc_VC_fit = loc_VC_array[mask_L]
# # loc_VC_ci_fit = loc_VC_ci_array[mask_L]

# # loc_VC_lower_fit = [ci[0] for ci in loc_VC_ci_fit]
# # loc_VC_upper_fit = [ci[1] for ci in loc_VC_ci_fit]


# # Perform curve fitting for the sparsity using log(n) and n
# params_loc_log_FH, cov_loc_log_FH = curve_fit(func_log, L_fit, loc_FH_fit)
# params_loc_n_FH, cov_loc_n_FH = curve_fit(func_n, L_fit, loc_FH_fit)

# # params_loc_log_VC, cov_loc_log_VC = curve_fit(func_log, L_fit, loc_VC_fit)
# # params_loc_n_VC, cov_loc_n_VC = curve_fit(func_n, L_fit, loc_VC_fit)


# # Extract fitted values
# a_loc_log_FH = params_loc_log_FH[0]
# b_loc_log_FH = params_loc_log_FH[1]

# a_loc_n_FH = params_loc_n_FH[0]
# b_loc_n_FH = params_loc_n_FH[1]

# # a_loc_log_VC = params_loc_log_VC[0]
# # b_loc_log_VC = params_loc_log_VC[1]

# # a_loc_n_VC = params_loc_n_VC[0]
# # b_loc_n_VC = params_loc_n_VC[1]


# # Calculate standard deviations
# std_a_loc_log_FH = np.sqrt(np.diag(cov_loc_log_FH))[0]
# std_b_loc_log_FH = np.sqrt(np.diag(cov_loc_log_FH))[1]

# std_a_loc_n_FH = np.sqrt(np.diag(cov_loc_n_FH))[0]
# std_b_loc_n_FH = np.sqrt(np.diag(cov_loc_n_FH))[1]

# # std_a_loc_log_VC = np.sqrt(np.diag(cov_loc_log_VC))[0]
# # std_b_loc_log_VC = np.sqrt(np.diag(cov_loc_log_VC))[1]

# # std_a_loc_n_VC = np.sqrt(np.diag(cov_loc_n_VC))[0]
# # std_b_loc_n_VC = np.sqrt(np.diag(cov_loc_n_VC))[1]


# # Format the parameters and their standard deviations
# log_loc_label_FH = f"log(n): a = {a_loc_log_FH:.1f} ± {std_a_loc_log_FH:.1f}, b = {b_loc_log_FH:.1f} ± {std_b_loc_log_FH:.1f}"
# n_loc_label_FH = f"n: a = {a_loc_n_FH:.1f} ± {std_a_loc_n_FH:.1f}, b = {b_loc_n_FH:.1f} ± {std_b_loc_n_FH:.1f}"

# # log_loc_label_VC = f"log(n): a = {a_loc_log_VC:.1f} ± {std_a_loc_log_VC:.1f}, b = {b_loc_log_VC:.1f} ± {std_b_loc_log_VC:.1f}"
# # n_loc_label_VC = f"n: a = {a_loc_n_VC:.1f} ± {std_a_loc_n_VC:.1f}, b = {b_loc_n_VC:.1f} ± {std_b_loc_n_VC:.1f}"


# # Create the interactive plot
# fig5, ax5 = plt.subplots(figsize=(8, 6))

# # Plot the locality vs. log(n)
# ax5.plot(log_L_fit, loc_FH_fit, label="Maximal locality of FH", marker="s", linestyle="--", color="orange")
# # ax5.plot(log_L_fit, loc_VC_fit, label="Maximal locality of VC", marker=".", linestyle="--", color="blue")

# # Plot the confidence intervals
# ax5.fill_between(log_L_fit, loc_FH_lower_fit, loc_FH_upper_fit, color='orange', alpha=0.3)
# # ax5.fill_between(log_L_fit, loc_VC_lower_fit, loc_VC_upper_fit, color='blue', alpha=0.3)

# # Plot the fitted curves
# ax5.plot(log_L_fit, func_log(L_fit, a_loc_log_FH, b_loc_log_FH), label=f"Fitted curve {log_loc_label_FH}", color='black')
# ax5.plot(log_L_fit, func_n(L_fit, a_loc_n_FH, b_loc_n_FH), label=f"Fitted curve {n_loc_label_FH}", color='red')

# ax5.set_xlabel('Lattice size (log(n))')
# ax5.set_ylabel('Locality of FH')
# ax5.set_title(f"Maximal locality of FH with fits for L > {threshold_fit_L}, using a {name_graph} graph")

# ax5.legend()
# ax5.grid()

# # Save the second plot as an interactive HTML file
# save_path_loc = r"C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Figuren\Data-analyse\Interactieve plots\interactive_locality_FH_plot_expander_p_thresh_deg_4_straight_sewing.html"

# mpld3.save_html(fig5, save_path_loc)

# plt.show()


 

# # Do the same for the number of edges

# # For VC
# actual_edges_VC_array = np.array(actual_edges_VC)
# actual_edges_VC_ci_array = np.array(actual_edges_VC_ci)

# actual_edges_VC_fit = actual_edges_VC_array[mask_L]
# actual_edges_VC_ci_fit = actual_edges_VC_ci_array[mask_L]

# actual_edges_VC_lower_fit = [ci[0] for ci in actual_edges_VC_ci_fit]
# actual_edges_VC_upper_fit = [ci[1] for ci in actual_edges_VC_ci_fit]

# # Perform curve fitting using nlog(n) and n^2
# params_actual_edges_nlog3_VC, cov_actual_edges_nlog3_VC = curve_fit(func_nlog3, L_fit, actual_edges_VC_fit)
# params_actual_edges_nloga_VC, cov_actual_edges_nloga_VC = curve_fit(func_nloga, L_fit, actual_edges_VC_fit)
# params_actual_edges_n2_VC, cov_actual_edges_n2_VC = curve_fit(func_n2, L_fit, actual_edges_VC_fit)

# # Extract fitted values
# a_actual_edges_nlog3_VC = params_actual_edges_nlog3_VC[0]
# b_actual_edges_nlog3_VC = params_actual_edges_nlog3_VC[1]

# a_actual_edges_nloga_VC = params_actual_edges_nloga_VC[0]
# b_actual_edges_nloga_VC = params_actual_edges_nloga_VC[1]
# c_actual_edges_nloga_VC = params_actual_edges_nloga_VC[2]

# a_actual_edges_n2_VC = params_actual_edges_n2_VC[0]
# b_actual_edges_n2_VC = params_actual_edges_n2_VC[1]


# # Calculate standard deviations
# std_a_actual_edges_nlog3_VC = np.sqrt(np.diag(cov_actual_edges_nlog3_VC))[0]
# std_b_actual_edges_nlog3_VC = np.sqrt(np.diag(cov_actual_edges_nlog3_VC))[1]

# std_a_actual_edges_nloga_VC = np.sqrt(np.diag(cov_actual_edges_nloga_VC))[0]
# std_b_actual_edges_nloga_VC = np.sqrt(np.diag(cov_actual_edges_nloga_VC))[1]
# std_c_actual_edges_nloga_VC = np.sqrt(np.diag(cov_actual_edges_nloga_VC))[2]

# std_a_actual_edges_n2_VC = np.sqrt(np.diag(cov_actual_edges_n2_VC))[0]
# std_b_actual_edges_n2_VC = np.sqrt(np.diag(cov_actual_edges_n2_VC))[1]


# # Format the parameters and their standard deviations
# nlog3_actual_edges_label_VC = f"nlog(n)^3: a = {a_actual_edges_nlog3_VC:.1f} ± {std_a_actual_edges_nlog3_VC:.1f}, b = {b_actual_edges_nlog3_VC:.1f} ± {std_b_actual_edges_nlog3_VC:.1f}"
# nloga_actual_edges_label_VC = f"nlog(n)^c: a = {a_actual_edges_nloga_VC:.1f} ± {std_a_actual_edges_nloga_VC:.1f}, b = {b_actual_edges_nloga_VC:.1f} ± {std_b_actual_edges_nloga_VC:.1f}, c = {c_actual_edges_nloga_VC:.1f} ± {std_c_actual_edges_nloga_VC:.1f}"
# n2_actual_edges_label_VC = f"n^2: a = {a_actual_edges_n2_VC:.1f} ± {std_a_actual_edges_n2_VC:.1f}, b = {b_actual_edges_n2_VC:.1f} ± {std_b_actual_edges_n2_VC:.1f}"

# # Create the interactive plot
# fig6, ax6 = plt.subplots(figsize=(8, 6))

# # Plot the actual_edges vs. nlog(n)^3
# ax6.plot(x_cube_fit, actual_edges_VC_fit, label="# of edges after applying VC", marker=".", linestyle="--", color="blue")

# # Plot the confidence intervals
# ax6.fill_between(x_cube_fit, actual_edges_VC_lower_fit, actual_edges_VC_upper_fit, color='blue', alpha=0.3)

# # Plot the fitted curves
# ax6.plot(x_cube_fit, func_nlog3(L_fit, a_actual_edges_nlog3_VC, b_actual_edges_nlog3_VC), label=f"Fitted curve {nlog3_actual_edges_label_VC}", color='black')
# ax6.plot(x_cube_fit, func_nloga(L_fit, a_actual_edges_nloga_VC, b_actual_edges_nloga_VC, c_actual_edges_nloga_VC), label=f"Fitted curve {nloga_actual_edges_label_VC}", color='red')
# ax6.plot(x_cube_fit, func_n2(L_fit, a_actual_edges_n2_VC, b_actual_edges_n2_VC), label=f"Fitted curve {n2_actual_edges_label_VC}", color='grey')

# ax6.set_xlabel('Lattice size (nlog(n)^3)')
# ax6.set_ylabel('Number of edges')
# ax6.set_title(f"Number of edges in the final graph with fits for L > {threshold_fit_L}, using a {name_graph} graph")

# ax6.legend()
# ax6.grid()

# # Save the second plot as an interactive HTML file
# save_path_actual_edges = r"C:\Users\koene\OneDrive\Documenten Koen\Studie\Jaar 4\BEP\Figuren\Data-analyse\Interactieve plots\interactive_actual_edges_VC_plot_expander_p_thresh_deg_4_straight_sewing_no_overlap.html"

# mpld3.save_html(fig6, save_path_actual_edges)

# plt.show()




















# ---------------------------------------------------------------------------------------------------------------------------------------------------------------




# # Create the mask for the fit
# mask_L = L_range > 40

# # Apply the mask
# log_L_array = np.array(log_L)
# big_cycles_FH_array = np.array(big_cycles_FH)
# big_cycles_FH_ci_array = np.array(big_cycles_FH_ci)  # Assuming confidence intervals are stored in big_cycles_FH_ci

# # Full range of L (for displaying big_cycles_FH values)
# L_full = L_range
# log_L_full = log_L_array

# # For fitting, use only values where L > 80 (mask)
# L_fit = L_range[mask_L]
# log_L_fit = log_L_array[mask_L]

# x_fit = L_fit * log_L_fit

# big_cycles_FH_fit = big_cycles_FH_array[mask_L]
# big_cycles_FH_ci_fit = big_cycles_FH_ci_array[mask_L]  # Confidence intervals

# big_cycles_FH_lower_fit = [ci[0] for ci in big_cycles_FH_ci_fit]
# big_cycles_FH_upper_fit = [ci[1] for ci in big_cycles_FH_ci_fit]

# # Perform curve fitting for L > 80
# params_big_cycles_n_FH, cov_big_cycles_n_FH = curve_fit(func_n, L_fit, big_cycles_FH_fit)  # big_cycles vs L
# params_big_cycles_nlog_FH, cov_big_cycles_nlog_FH = curve_fit(func_nlog, L_fit, big_cycles_FH_fit)  # big_cycles vs nlog(L)

# # Extract fitted values
# a_big_cycles_n_FH = params_big_cycles_n_FH[0]
# e_big_cycles_n_FH = params_big_cycles_n_FH[1]

# d_big_cycles_nlog_FH = params_big_cycles_nlog_FH[0]
# h_big_cycles_nlog_FH = params_big_cycles_nlog_FH[1]

# # Calculate standard deviations (the square root of the diagonal elements of the covariance matrix)
# std_a = np.sqrt(np.diag(cov_big_cycles_n_FH))[0]
# std_e = np.sqrt(np.diag(cov_big_cycles_n_FH))[1]

# std_d = np.sqrt(np.diag(cov_big_cycles_nlog_FH))[0]
# std_h = np.sqrt(np.diag(cov_big_cycles_nlog_FH))[1]

# # Format the parameters and their standard deviations
# a_label = f"a = {a_big_cycles_n_FH:.1f} ± {std_a:.1f}"
# e_label = f"e = {e_big_cycles_n_FH:.1f} ± {std_e:.1f}"
# d_label = f"d = {d_big_cycles_nlog_FH:.1f} ± {std_d:.1f}"
# h_label = f"h = {h_big_cycles_nlog_FH:.1f} ± {std_h:.1f}"



# # Enable interactive mode
# plt.ion()

# # --- Plot big cycles with confidence intervals and fitted curves ---
# plt.figure(figsize=(8, 6))

# # Plot big cycles as a function of nlog(n) for all L values
# plt.plot(x_fit, big_cycles_FH_fit, label="Number of big cycles", marker="s", linestyle="--", color="orange")

# # Plot the confidence intervals as a shaded region (semi-transparent band) for the values of L > 80
# plt.fill_between(x_fit, big_cycles_FH_lower_fit, big_cycles_FH_upper_fit, color='orange', alpha=0.3)

# # Plot the fitted curves (for L > 80)
# plt.plot(x_fit, func_n(L_fit, a_big_cycles_n_FH, e_big_cycles_n_FH), label=f"Fitted curve {a_label}, {e_label}", color='grey')
# plt.plot(x_fit, func_nlog(L_fit, d_big_cycles_nlog_FH, h_big_cycles_nlog_FH), label=f"Fitted curve {d_label}, {h_label}", color='black')

# # Customize the plot
# plt.xlabel('Lattice size (nlog(n))')
# plt.ylabel('Number of cycles')
# plt.title(f"Number of cycles with length > {threshold_length} of FH with fits for L > 80, using a {name_graph} graph")

# # Update the legend to include the labels with the parameters and standard deviations
# plt.legend()
# plt.grid()

# # Show the plot (interactive mode is enabled, so you can zoom and pan)
# plt.show()

# # Disable interactive mode
# plt.ioff()




# # Now for the total weight
# weight_FH_array = np.array(weight_FH)  # Assuming weight_FH is a numpy array

# weight_FH_fit = weight_FH_array[mask_L]


# # Perform curve fitting for L > 80 using nlog(n) and n^2
# params_weight_nlog_FH, cov_weight_nlog_FH = curve_fit(func_nlog, L_fit, weight_FH_fit)
# params_weight_n2_FH, cov_weight_n2_FH = curve_fit(func_n2, L_fit, weight_FH_fit)

# # Extract fitted values for weight_FH
# d_weight_nlog_FH = params_weight_nlog_FH[0]
# h_weight_nlog_FH = params_weight_nlog_FH[1]

# k_weight_n2_FH = params_weight_n2_FH[0]
# l_weight_n2_FH = params_weight_n2_FH[1]

# # Calculate standard deviations for weight_FH fits
# std_d_weight_nlog = np.sqrt(np.diag(cov_weight_nlog_FH))[0]
# std_h_weight_nlog = np.sqrt(np.diag(cov_weight_nlog_FH))[1]

# std_k_weight_n2 = np.sqrt(np.diag(cov_weight_n2_FH))[0]
# std_l_weight_n2 = np.sqrt(np.diag(cov_weight_n2_FH))[1]

# # Format the parameters and their standard deviations for weight_FH fits
# nlog_weight_label = f"nlog: d = {d_weight_nlog_FH:.1f} ± {std_d_weight_nlog:.1f}, h = {h_weight_nlog_FH:.1f} ± {std_h_weight_nlog:.1f}"
# n2_weight_label = f"n^2: k = {k_weight_n2_FH:.1f} ± {std_k_weight_n2:.1f}, l = {l_weight_n2_FH:.1f} ± {std_l_weight_n2:.1f}"

# # Enable interactive mode
# plt.ion()

# # --- Plot weight_FH with fitted curves ---
# plt.figure(figsize=(8, 6))

# # Plot weight_FH as a function of nlog(n) for all L values
# plt.plot(x_fit, weight_FH_fit, label="Weight of FH", marker="o", linestyle="--", color="green")

# # Plot the fitted curves for nlog(n) and n^2
# plt.plot(x_fit, func_nlog(L_fit, d_weight_nlog_FH, h_weight_nlog_FH), label=f"Fitted curve {nlog_weight_label}", color='blue')
# plt.plot(x_fit, func_n2(L_fit, k_weight_n2_FH, l_weight_n2_FH), label=f"Fitted curve {n2_weight_label}", color='red')

# # Customize the plot
# plt.xlabel('Lattice size (nlog(n))')
# plt.ylabel('Weight of FH')
# plt.title(f"Weight of FH with fits for L > 80, using a {name_graph} graph")

# # Update the legend to include the labels with the parameters and standard deviations
# plt.legend()
# plt.grid()

# # Show the plot (interactive mode is enabled, so you can zoom and pan)
# plt.show()

# # Disable interactive mode
# plt.ioff()





# # --- Plot big cycles with confidence intervals and fitted curves ---
# plt.figure(figsize=(8, 6))

# # Plot big cycles as a function of nlog(n) for all L values
# plt.plot(x_fit, big_cycles_FH_fit, label="Number of big cycles", marker="s", linestyle="--", color="orange")

# # Plot the confidence intervals as a shaded region (semi-transparent band) for the values of L > 80
# plt.fill_between(x_fit, big_cycles_FH_lower_fit, big_cycles_FH_upper_fit, color='orange', alpha=0.3)

# # Plot the fitted curves (for L > 80)
# plt.plot(x_fit, func_n(L_fit, a_big_cycles_n_FH, e_big_cycles_n_FH), label=f"Fitted curve {a_label}, {e_label}", color='grey')
# plt.plot(x_fit, func_nlog(L_fit, d_big_cycles_nlog_FH, h_big_cycles_nlog_FH), label=f"Fitted curve {d_label}, {h_label}", color='black')

# # plt.plot(x_fit, 3.6*x_fit - 128, label="n", color="blue")
# # plt.plot(x_fit, 0.4*x_fit*np.log(x_fit) - 5, label="nlog(n)", color="red")

# # Customize the plot
# plt.xlabel('Lattice size (nlog(n))')
# plt.ylabel('Number of cycles')
# plt.title(f"Number of cycles with length > {threshold_length} of FH with fits for L > 80, using a {name_graph} graph")

# # Update the legend to include the labels with the parameters and standard deviations
# plt.legend()
# plt.grid()

# # Show the plot
# plt.show()





# # Create the mask for the fit
# mask_L = L_range > 80

# # Apply the mask
# log_L_array = np.array(log_L)
# big_cycles_FH_array = np.array(big_cycles_FH)
# big_cycles_FH_ci_array = np.array(big_cycles_FH_ci)  # Assuming confidence intervals are stored in big_cycles_FH_ci

# L_fit = L_range[mask_L]
# log_L_fit = log_L_array[mask_L]

# x_fit = L_fit * log_L_fit

# big_cycles_FH_fit = big_cycles_FH_array[mask_L]
# big_cycles_FH_ci_fit = big_cycles_FH_ci_array[mask_L]  # Confidence intervals

# big_cycles_FH_lower_fit = [ci[0] for ci in big_cycles_FH_ci_fit]
# big_cycles_FH_upper_fit = [ci[1] for ci in big_cycles_FH_ci_fit]


# # Perform curve fitting
# params_big_cycles_n_FH, cov_big_cycles_n_FH = curve_fit(func_n, L_fit, big_cycles_FH_fit)  # big_cycles vs L
# params_big_cycles_nlog_FH, cov_big_cycles_nlog_FH = curve_fit(func_nlog, L_fit, big_cycles_FH_fit)  # big_cycles vs nlog(L)

# # Extract fitted values
# a_big_cycles_n_FH = params_big_cycles_n_FH[0]
# e_big_cycles_n_FH = params_big_cycles_n_FH[1]

# d_big_cycles_nlog_FH = params_big_cycles_nlog_FH[0]
# h_big_cycles_nlog_FH = params_big_cycles_nlog_FH[1]

# # Calculate standard deviations (the square root of the diagonal elements of the covariance matrix)
# std_a = np.sqrt(np.diag(cov_big_cycles_n_FH))[0]
# std_e = np.sqrt(np.diag(cov_big_cycles_n_FH))[1]

# std_d = np.sqrt(np.diag(cov_big_cycles_nlog_FH))[0]
# std_h = np.sqrt(np.diag(cov_big_cycles_nlog_FH))[1]

# # Format the parameters and their standard deviations
# a_label = f"a = {a_big_cycles_n_FH:.1f} ± {std_a:.1f}"
# e_label = f"e = {e_big_cycles_n_FH:.1f} ± {std_e:.1f}"
# d_label = f"d = {d_big_cycles_nlog_FH:.1f} ± {std_d:.1f}"
# h_label = f"h = {h_big_cycles_nlog_FH:.1f} ± {std_h:.1f}"

# # --- Plot big cycles with confidence intervals and fitted curves ---
# plt.figure(figsize=(8, 6))

# # Plot big cycles as a function of nlog(n)
# plt.plot(x_fit, big_cycles_FH_fit, label="Number of big cycles", marker="s", linestyle="--", color="orange")

# # Plot the confidence intervals as a shaded region (semi-transparent band)
# plt.fill_between(x_fit, big_cycles_FH_lower_fit, big_cycles_FH_upper_fit, color='orange', alpha=0.3)

# # Plot the fitted curves (for L > 60)
# plt.plot(x_fit, func_n(L_fit, a_big_cycles_n_FH, e_big_cycles_n_FH), label=f"Fitted curve {a_label}, {e_label}", color='grey')
# plt.plot(x_fit, func_nlog(L_fit, d_big_cycles_nlog_FH, h_big_cycles_nlog_FH), label=f"Fitted curve {d_label}, {h_label}", color='black')

# # Customize the plot
# plt.xlabel('Lattice size (nlog(n))')
# plt.ylabel('Number of cycles')
# plt.title(f"Number of cycles with length > {threshold_length} of FH with fits for L > 80, using a {name_graph} graph")

# # Update the legend to include the labels with the parameters and standard deviations
# plt.legend()
# plt.grid()

# # Show the plot
# plt.show()



























# # Want to make a fit for locality, sparsity, big_cycles and size for FH
# params_loc_n_FH, cov_loc_n_FH = curve_fit(func_n, L_range, loc_FH) # locality vs n
# params_loc_log_FH, cov_loc_log_FH = curve_fit(func_log, L_range, loc_FH) # locality vs log(n)


# params_spars_log_FH, cov_spar_log_FH = curve_fit(func_log_sqrd, L_range, spars_FH) # sparsity vs log(n)^2

# params_big_cycles_n_FH, cov_big_cycles_n_FH = curve_fit(func_n, L_range, big_cycles_FH) # big_cycles vs n
# params_big_cycles_nlog_FH, cov_big_cycles_nlog_FH = curve_fit(func_nlog, L_range, big_cycles_FH) # big_cycles vs nlog(n)

# params_size_n_FH, cov_size_n_FH = curve_fit(func_n, L_range, size_FH) # big_cycles vs n
# params_size_nlog_FH, cov_size_nlog_FH = curve_fit(func_nlog, L_range, size_FH) # big_cycles vs nlog(n)


# # Extract fitted values
# a_loc_n_FH = params_loc_n_FH[0]
# b_loc_log_FH = params_loc_log_FH[0]

# c_spars_log_FH = params_spars_log_FH[0]

# a_big_cycles_n_FH = params_big_cycles_n_FH[0]
# d_big_cycles_nlog_FH = params_big_cycles_nlog_FH[0]

# a_size_n_FH = params_size_n_FH[0]
# d_size_nlog_FH = params_size_nlog_FH[0]


# # --- Plot locality ---
# plt.figure(figsize=(8, 6))
# plt.plot(L_range, loc_FH, label="Freedman-Hastings", marker="s", linestyle="dashed", color="orange")
# plt.plot(L_range, func_n(L_range, a_loc_n_FH), 
#          label="Fitted curve a*n", color='grey')
# plt.plot(L_range, func_log(L_range, b_loc_log_FH), 
#          label="Fitted curve b*log(n)", color='black')
# plt.xlabel('Lattice size (n)')
# plt.ylabel('Locality')
# plt.title(f"Locality of FH fitted against a*n and b*log(n), using a {name_graph} graph")
# plt.legend()
# plt.grid()
# plt.show()


# # --- Plot sparsity ---
# plt.figure(figsize=(8, 6))
# plt.plot(L_range, spars_FH, label="Freedman-Hastings", marker="s", linestyle="dashed", color="orange")
# plt.plot(L_range, func_log_sqrd(L_range, c_spars_log_FH), 
#          label="Fitted curve c*log(n)^2", color='brown')
# plt.xlabel('Lattice size (n)')
# plt.ylabel('Sparsity')
# plt.title(f"Sparsity of FH fitted against c*log(n)^2, using a {name_graph} graph")
# plt.legend()
# plt.grid()
# plt.show()


# # --- Plot big cycles ---
# plt.figure(figsize=(8, 6))
# plt.plot(L_range, big_cycles_FH, label="Freedman-Hastings", marker="s", linestyle="dashed", color="orange")
# plt.plot(L_range, func_n(L_range, a_big_cycles_n_FH), 
#          label="Fitted curve a*n", color='grey')
# plt.plot(L_range, func_nlog(L_range, d_big_cycles_nlog_FH), 
#          label="Fitted curve d*nlog(n)", color='purple')
# plt.xlabel('Lattice size (n)')
# plt.ylabel('Number of cycles')
# plt.title(f"Number of cycles with length > 4 of FH fitted against d*nlog(n), using a {name_graph} graph")
# plt.legend()
# plt.grid()
# plt.show()


# # --- Plot size ---
# plt.figure(figsize=(8, 6))
# plt.plot(L_range, size_FH, label="Freedman-Hastings", marker="s", linestyle="dashed", color="orange")
# plt.plot(L_range, func_n(L_range, a_size_n_FH), 
#          label="Fitted curve a*n", color='grey')
# plt.plot(L_range, func_nlog(L_range, d_size_nlog_FH), 
#          label="Fitted curve d*nlog(n)", color='purple')
# plt.xlabel('Lattice size (n)')
# plt.ylabel('Number of cycles')
# plt.title(f"Number of cycles from FH fitted against d*nlog(n), using a {name_graph} graph")
# plt.legend()
# plt.grid()
# plt.show()







# Function to fit and plot the data
def fit_data(L_range, loc_FH, spars_FH, model_loc_log, model_n, model_spars, name_graph):

    # Fit curves
    params_loc_log_FH, cov_loc_log_FH = curve_fit(model_loc_log, L_range, loc_FH)
    params_loc_n_FH, cov_loc_n_FH = curve_fit(model_n, L_range, loc_FH)

    params_spars_log_FH, cov_spars_log_FH = curve_fit(model_spars, L_range, spars_FH)
    params_spars_n_FH, cov_spars_n_FH = curve_fit(model_n, L_range, spars_FH)

    # Extract fitted values
    k_loc_log_FH = params_loc_log_FH[0]
    a_loc_n_FH = params_loc_n_FH[0]

    c_spars_log_FH = params_spars_log_FH[0]
    b_spars_n_FH = params_spars_n_FH[0]

    # Calculate standard deviations (uncertainties)
    std_k_FH = np.sqrt(cov_loc_log_FH[0, 0])
    std_a_FH = np.sqrt(cov_loc_n_FH[0, 0])

    std_c_FH = np.sqrt(cov_spars_log_FH[0, 0])
    std_b_FH = np.sqrt(cov_spars_n_FH[0, 0])

    # Calculate confidence intervals (95% confidence level)
    conf_interval_k_FH = 1.96 * std_k_FH
    conf_interval_a_FH = 1.96 * std_a_FH

    conf_interval_c_FH = 1.96 * std_c_FH
    conf_interval_b_FH = 1.96 * std_b_FH

    # --- Plot Locality ---
    plt.figure(figsize=(8, 6))
    plt.plot(L_range, loc_FH, label="Freedman-Hastings", marker="s", linestyle="dashed", color="orange")
    plt.plot(L_range, model_loc_log(L_range, k_loc_log_FH), 
             label=f"Fitted curve k*log(n): k = {k_loc_log_FH:.2f} ± {conf_interval_k_FH:.2f}", color='black')
    plt.plot(L_range, model_n(L_range, a_loc_n_FH), 
             label=f"Fitted curve a*n: a = {a_loc_n_FH:.2f} ± {conf_interval_a_FH:.2f}", color='grey')
    plt.xlabel('Lattice size (n)')
    plt.ylabel('Locality')
    plt.title('Locality of FH fitted against k*log(n) and a*n, using a {name_graph} graph')
    plt.legend()
    plt.grid()
    plt.show()

    # --- Plot Sparsity ---
    plt.figure(figsize=(8, 6))
    plt.plot(L_range, spars_FH, label="Freedman-Hastings", marker="s", linestyle="dashed", color="orange")
    plt.plot(L_range, model_spars(L_range, c_spars_log_FH), 
             label=f"Fitted curve c*log(n)^2: c = {c_spars_log_FH:.2f} ± {conf_interval_c_FH:.2f}", color='black')
    plt.plot(L_range, model_n(L_range, b_spars_n_FH), 
             label=f"Fitted curve b*n: b = {b_spars_n_FH:.2f} ± {conf_interval_b_FH:.2f}", color='grey')
    plt.xlabel('Lattice size (n)')
    plt.ylabel('Sparsity')
    plt.title('Sparsity of FH fitted against c*log(n)^2, using a {name_graph} graph')
    plt.legend()
    plt.grid()
    plt.show()

    # --- Print Results ---
    print(f"The value of k for locality of FH using log(n) is {k_loc_log_FH:.2f} ± {conf_interval_k_FH:.2f}")
    print(f"The value of a for locality of FH using n is {a_loc_n_FH:.2f} ± {conf_interval_a_FH:.2f}")
    print()
    print(f"The value of c for sparsity of FH using log(n)^2 is {c_spars_log_FH:.2f} ± {conf_interval_c_FH:.2f}")
    print(f"The value of b for sparsity of FH using n is {b_spars_n_FH:.2f} ± {conf_interval_b_FH:.2f}")
    print()

    # Return fitted parameters and confidence intervals
    return {
        "k_loc_log_FH": (k_loc_log_FH, conf_interval_k_FH),
        "a_loc_n_FH": (a_loc_n_FH, conf_interval_a_FH),
        "c_spars_log_FH": (c_spars_log_FH, conf_interval_c_FH),
        "b_spars_n_FH": (b_spars_n_FH, conf_interval_b_FH)
    }


# Fit the data and plot it
# results_fit = fit_data(L_range, loc_FH, spars_FH, model_loc_log, model_n, model_spars)


# Check the distribution of the confidence intervals
def plot_distribution_for_L(index, L_values, loc_ST_values_all, loc_FH_values_all, spar_ST_values_all, spar_FH_values_all, name_graph):

    L_selected = L_values[index]

    # Data for the locality
    data_ST_loc = np.array(loc_ST_values_all[index]).flatten()
    data_FH_loc = np.array(loc_FH_values_all[index]).flatten()
    
    # Data for the sparsity
    data_ST_spar = np.array(spar_ST_values_all[index]).flatten()
    data_FH_spar = np.array(spar_FH_values_all[index]).flatten()


    # Plots for the locality    
    plt.figure(figsize=(10, 6))

    sns.histplot(data_ST_loc, bins=20, kde=True, label="Spanning Tree", color="blue", alpha=0.5)
    sns.histplot(data_FH_loc, bins=20, kde=True, label="Freedman-Hastings", color="green", alpha=0.5)

    plt.xlabel("Locality")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Locality for L = {L_selected}, using a {name_graph} graph")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=[data_ST_loc, data_FH_loc], palette=["blue", "green"])
    plt.xticks([0, 1], ["Spanning Tree", "Freedman-Hastings"])
    plt.ylabel("Locality")
    plt.title(f"Boxplot of Locality for L = {L_selected}, using a {name_graph} graph")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=[data_ST_loc, data_FH_loc], palette=["blue", "green"])
    plt.xticks([0, 1], ["Spanning Tree", "Freedman-Hastings"])
    plt.ylabel("Locality")
    plt.title(f"Violin Plot of Locality for L = {L_selected}, using a {name_graph} graph")
    plt.show()


    # Plots for the sparsity
    plt.figure(figsize=(10, 6))

    sns.histplot(data_ST_spar, bins=20, kde=True, label="Spanning Tree", color="blue", alpha=0.5)
    sns.histplot(data_FH_spar, bins=20, kde=True, label="Freedman-Hastings", color="green", alpha=0.5)

    plt.xlabel("Sparsity")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Sparsity for L = {L_selected}, using a {name_graph} graph")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=[data_ST_spar, data_FH_spar], palette=["blue", "green"])
    plt.xticks([0, 1], ["Spanning Tree", "Freedman-Hastings"])
    plt.ylabel("Sparsity")
    plt.title(f"Boxplot of Sparsity for L = {L_selected}, using a {name_graph} graph")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=[data_ST_spar, data_FH_spar], palette=["blue", "green"])
    plt.xticks([0, 1], ["Spanning Tree", "Freedman-Hastings"])
    plt.ylabel("Sparsity")
    plt.title(f"Violin Plot of Sparsity for L = {L_selected}, using a {name_graph} graph")
    plt.show()


# # Make the distribution plots
# index = 7

# plot_distribution_for_L(index, L_range, loc_ST_values_all, loc_FH_values_all, spar_ST_values_all, spar_FH_values_all, name_graph)







# Step 7 - Add edges to maintain sparsity and locality and investigate how many edges (qubits) we need to achieve this ----------------------



# Visualization of the graph ----------------------------------------------------------------------------

# Function to visualize the graph with its cycle basis
def visualize_graph_CB(G, cycle_basis_Horton, cycle_basis_FH):

    # Position nodes in a circular layout
    pos = nx.circular_layout(G)
    nx.set_node_attributes(G, pos, "pos")  # Store positions in G

    # Define colors for visualizing the cycle basis
    colors = ['r', 'g', 'b', 'm', 'y', 'c']

    # --- Draw the Graph ---
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=12)
    plt.title("Graph with size " + str(G.size()))
    plt.show()

    # --- Draw Horton Cycle Basis if Provided ---
    if cycle_basis_Horton is not None:
        plt.figure(figsize=(6, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=12)

        # Draw cycles in different colors
        for i, cycle in enumerate(cycle_basis_Horton):
            cycle = [node for node in cycle if node in G.nodes()]
            if len(cycle) < 2:
                continue

            cycle_edges = [(cycle[j], cycle[(j+1) % len(cycle)]) for j in range(len(cycle))]
            cycle_edges = [edge for edge in cycle_edges if edge[0] in pos and edge[1] in pos]

            nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, edge_color=colors[i % len(colors)], width=2)

        plt.title("Graph with a cycle basis from Horton, with size " + str(len(cycle_basis_Horton)))
        plt.show()

    # --- Draw Freedman-Hastings Cycle Basis if Provided ---
    if cycle_basis_FH is not None:
        plt.figure(figsize=(6, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=12)

        # Draw cycles in different colors
        for i, cycle in enumerate(cycle_basis_FH):
            cycle = [node for node in cycle if node in G.nodes()]
            if len(cycle) < 2:
                continue

            cycle_edges = [(cycle[j], cycle[(j+1) % len(cycle)]) for j in range(len(cycle))]
            cycle_edges = [edge for edge in cycle_edges if edge[0] in pos and edge[1] in pos]

            nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, edge_color=colors[i % len(colors)], width=2)

        plt.title("Graph with a cycle basis from Freedman-Hastings, with size " + str(len(cycle_basis_FH)))
        plt.show()


# Visualize the graph
# visualize_graph_CB(G)
# visualize_graph_CB(G, cycle_basis_Horton=cycle_basis_Horton, cycle_basis_FH=cycle_basis_FH)


# Function to visualize the 3D stacked graph
def visualize_3D_stacked_graph(G, G_prime, node_mapping, num_layers):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Generate a 2D layout for the original graph
    pos_2D = nx.spring_layout(G, seed=42)  # Force-directed layout
    pos_3D = {}  # Store (X, Y, Z) positions for all nodes in G_prime

    # Assign positions in 3D space: (X, Y from G, Z from layers)
    for (layer, node), mapped_id in node_mapping.items():
        x, y = pos_2D[node]  # Get original 2D layout coordinates
        z = layer  # Assign layer as Z-coordinate
        pos_3D[mapped_id] = (x, y, z)

    # Draw intra-layer edges (Gray) - Edges within the same layer
    for layer in range(num_layers):
        for u, v in G.edges():
            if (layer, u) in node_mapping and (layer, v) in node_mapping:
                u_id = node_mapping[(layer, u)]
                v_id = node_mapping[(layer, v)]

                x_vals = [pos_3D[u_id][0], pos_3D[v_id][0]]
                y_vals = [pos_3D[u_id][1], pos_3D[v_id][1]]
                z_vals = [pos_3D[u_id][2], pos_3D[v_id][2]]

                ax.plot(x_vals, y_vals, z_vals, color='gray', linewidth=1.5, alpha=0.7)

    # Draw inter-layer edges (Blue) - Vertical edges connecting layers
    for (u, v) in G_prime.edges():
        if abs(pos_3D[u][2] - pos_3D[v][2]) > 0:  # Only inter-layer edges
            x_vals = [pos_3D[u][0], pos_3D[v][0]]
            y_vals = [pos_3D[u][1], pos_3D[v][1]]
            z_vals = [pos_3D[u][2], pos_3D[v][2]]

            ax.plot(x_vals, y_vals, z_vals, color='blue', linewidth=1.5, alpha=0.9)

    # Draw nodes
    for node, (x, y, z) in pos_3D.items():
        ax.scatter(x, y, z, color="black", s=50, edgecolors="k", alpha=1)

    # Label axes
    ax.set_xlabel("X-axis (Original Graph Layout)")
    ax.set_ylabel("Y-axis (Original Graph Layout)")
    ax.set_zlabel("Layers (Z-axis)")
    ax.set_title("3D Stacked Graph Visualization")

    plt.show()

# visualize_3D_stacked_graph(G, G_prime, node_mapping, num_layers=len(cycle_basis_ST))

