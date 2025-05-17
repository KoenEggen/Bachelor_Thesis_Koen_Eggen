# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:59:58 2025

@author: koene
"""

# import sys, os

# os.environ["KMP_DUPLICATE_LIB_OK"] = (
#     "True"  # uncomment this line if omp error occurs on OSX for python 3
# )
# os.environ["OMP_NUM_THREADS"] = "1"  # set number of OpenMP threads to run in parallel
# os.environ["MKL_NUM_THREADS"] = "1"  # set number of MKL threads to run in parallel

import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spinless_fermion_basis_general
from itertools import product

# Set seed for reproducibility
seed = 0
np.random.seed(seed)


# Function for creating the SYK Hamiltonian
def syk_hamiltonian(L, J):
    SYK_int = [(J[i, j, k, l], i, j, k, l) for i, j, k, l in product(range(L), repeat=4) if J[i, j, k, l] != 0]

    basis = spinless_fermion_basis_general(L)
    H_SYK = hamiltonian([["xxxx", SYK_int]], [], basis=basis, check_symm=False)

    return H_SYK
