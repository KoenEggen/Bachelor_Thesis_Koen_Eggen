# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 14:32:14 2025

@author: koene
"""

import numpy as np
import matplotlib.pyplot as plt

n_even = np.arange(2, 251, 2)
n_odd = np.arange(1, 251, 2)

def qubit_scaling_straight(n):
    
    if n % 2 == 0:
        return (n - 2) / 2
    
    elif n % 2 == 1:
        return (n - 3) / 2


def qubit_scaling_triangle(n):
    
    return n - 3


def qubit_scaling_straight_6(n):
    
    if n % 2 == 0:
        return (n - 2) / 4
    
    elif n % 2 == 1:
        return (n - 3) / 4


def qubit_scaling_skew(n):
    
    if n % 2 == 0:
        return (n - 2) / 2 - 1
    
    elif n % 2 == 1:
        return (n - 3) / 2


one_array = np.ones_like(n_even)

# For the straight version
loc_stabilizer_straight = 4 * one_array
loc_Ham_straight = 1 * one_array
qubit_straight_even = np.vectorize(qubit_scaling_straight)(n_even)
qubit_straight_odd = np.vectorize(qubit_scaling_straight)(n_odd)

# For the triangular version
loc_stabilizer_triangle = 3 * one_array
loc_Ham_triangle = 2 * one_array
qubit_triangle_even = np.vectorize(qubit_scaling_triangle)(n_even)
qubit_triangle_odd = np.vectorize(qubit_scaling_triangle)(n_odd)

# For the straight-6 version
loc_stabilizer_straight_6 = 6 * one_array
loc_Ham_straight_6 = 1 * one_array
qubit_straight_6_even = np.vectorize(qubit_scaling_straight_6)(n_even)
qubit_straight_6_odd = np.vectorize(qubit_scaling_straight_6)(n_odd)

# For the straight-6 version
loc_stabilizer_skew = 4 * one_array
loc_Ham_skew = 1 * one_array
qubit_skew_even = np.vectorize(qubit_scaling_skew)(n_even)
qubit_skew_odd = np.vectorize(qubit_scaling_skew)(n_odd)


# Plot for the qubit scaling
plt.figure(figsize=(8, 6))
plt.plot(n_even, qubit_straight_even, linestyle = "--", color = "orange", label="Straight version")
plt.plot(n_even, qubit_triangle_even, linestyle = "--", color = "blue", label="Triangular version")
plt.plot(n_even, qubit_straight_6_even, linestyle = "--", color = "green", label="Straight_6 version")
plt.plot(n_even, qubit_skew_even, linestyle = "--", color = "black", label="Skew version")
plt.title("Qubit scaling for n even")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(n_odd, qubit_straight_odd, linestyle = "--", color = "orange", label="Straight version")
plt.plot(n_odd, qubit_triangle_odd, linestyle = "--", color = "blue", label="Triangular version")
plt.plot(n_odd, qubit_straight_6_odd, linestyle = "--", color = "green", label="Straight_6 version")
plt.plot(n_odd, qubit_skew_odd, linestyle = "--", color = "black", label="Skew version")
plt.title("Qubit scaling for n odd")
plt.grid()
plt.legend()
plt.show()






