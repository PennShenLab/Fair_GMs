import random

import numpy as np
import networkx as nx
from numpy.random import multivariate_normal

def soft_threshold(X, threshold, modify_diag=False):
    if modify_diag:
        diag_vals = np.diag(X)
    res = np.sign(X, dtype='float32') * np.maximum(np.abs(X, dtype='float32') - threshold, 0, dtype='float32')
    if modify_diag:
        np.fill_diagonal(res, diag_vals)
    return res

def HubNetwork(p, sparsity, hubnumber, hubsparsity, type="Gaussian", hubcol=None):
    # Generate an Erdos Renyi type network with positive and negative entries
    sparse = (np.random.binomial(1, 1-sparsity, p*p) * np.random.choice([-1, 1], p*p) *
              np.random.uniform(0.25, 0.5, p*p))
    
    Theta = sparse.reshape(p, p)
    Theta[np.tril_indices(p, -1)] = 0
    Theta += Theta.T

    # Add in Hub Nodes and make the matrix symmetric
    if hubcol is None:
        hubcol = np.random.choice(range(p), hubnumber, replace=False)
    hub_values = (np.random.binomial(1, 1-hubsparsity, hubnumber*p) *
                  np.random.choice([-1, 1], hubnumber*p) *
                  np.random.uniform(0.25, 0.75, hubnumber*p))
    for i, col in enumerate(hubcol):
        Theta[:, col] = hub_values[i*p:(i+1)*p]
    Theta = (Theta + Theta.T) / 2

    if type == "binary":
        np.fill_diagonal(Theta, np.random.choice([-1, 1], p) * np.random.uniform(0.25, 0.75, p))
        return {"Theta": Theta, "hubcol": hubcol}

    # Make the matrix positive definite
    np.fill_diagonal(Theta, 0)
    min_eigenvalue = np.min(np.linalg.eigvals(Theta))
    np.fill_diagonal(Theta, -min_eigenvalue + 0.1 if min_eigenvalue < 0 else 0.1)

    if type == "covariance":
        d = np.sqrt(np.diag(Theta))
        Theta /= d[:, None]
        Theta /= d[None, :]

    return {"Theta": Theta, "hubcol": hubcol}

def binary_mcmc(n, Theta, burnin, skip, trace=False):
    p = Theta.shape[0]
    X = np.zeros((n, p))
    temp = np.random.choice([1, 0], p)

    # samples from burn in period
    for t in range(burnin):
        for j in range(p):
            prob = np.exp(Theta[j, j] + np.sum(temp * Theta[j, :]) - temp[j] * Theta[j, j]) / (1 + np.exp(Theta[j, j] + np.sum(temp * Theta[j, :]) - temp[j] * Theta[j, j]))
            temp[j] = np.random.binomial(1, prob)
        if t % 10000 == 0:
            pass

    # Samples obtained by skipping certain number of samples
    k = 1
    samples = 0
    while samples < n:
        for j in range(p):
            prob = np.exp(Theta[j, j] + np.sum(temp * Theta[j, :]) - temp[j] * Theta[j, j]) / (1 + np.exp(Theta[j, j] + np.sum(temp * Theta[j, :]) - temp[j] * Theta[j, j]))
            temp[j] = np.random.binomial(1, prob)

        if k % skip == 0:
            X[samples, :] = temp
            samples += 1
            if trace:
                print(samples)
        k += 1

    return X

def compute_sigma(matrix):
    inv_matrix = np.linalg.inv(matrix)
    sigma = np.zeros(matrix.shape)
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            dij = 0.6 if i != j else 1
            sigma[i][j] = dij * inv_matrix[i][j] / (inv_matrix[i][i] * inv_matrix[j][j]) ** 0.5
            
    return sigma

def simulation_join_graph(n, N, C, g, prep):
    
    Graph = np.zeros((n, n))
    p0 = n // prep
    
    for i in range(prep):
        G = nx.barabasi_albert_graph(p0, C)
        sparse_adj_matrix = nx.adjacency_matrix(G)
        dense_adj_matrix = sparse_adj_matrix.todense()
        sub0 = np.array(dense_adj_matrix)
        Graph[i*p0:(i+1)*p0, i*p0:(i+1)*p0] = sub0

    Sigma = []
    Theta = np.zeros((n, n))
    for i in range(n):
        Theta[i][i] = 1

    for i in range(n):
        for j in range(n):
            if Graph[i][j] > 0:
                Theta[i][j] = np.random.choice([-0.4, -0.1, 0.1, 0.4])

    # Ensure positive definiteness
    for i in range(n):
        row_sum = sum([abs(Theta[i][j]) for j in range(n) if j != i])
        for j in range(n):
            if i != j:
                Theta[i][j] /= (1.5 * row_sum)
    
    Theta = (Theta + Theta.T) / 2
    Sigma.append(compute_sigma(Theta))

    for i in range(1, g):
        Temp = Sigma[-1].copy()
        Temp[0:i*p0, 0:i*p0] = np.identity(i*p0)
        Sigma.append(Temp)

    # Sample
    data_class = []
    for i in range(g):
        data_class.append(multivariate_normal(np.zeros(n), Sigma[i], N[i]))

    return data_class, Sigma


def simulation_random_graph(n, N, g, prep):
    
    Theta = np.zeros((n, n))
    p0 = n // prep
    
    for i in range(prep):
        sub0 = np.random.normal(0.7, 0.2, (p0, p0))
        sub0 = (sub0 + sub0)/2
        Theta[i*p0:(i+1)*p0, i*p0:(i+1)*p0] = sub0

    Theta = (Theta + Theta.T) / 2
    Sigma = []

    # Ensure positive definiteness
    eigenvalues, eigenvectors = np.linalg.eigh(Theta)
    epsilon = 1e-5
    eigenvalues_modified = np.maximum(eigenvalues, epsilon)
    Theta = eigenvectors @ np.diag(eigenvalues_modified) @ eigenvectors.T
    Theta = (Theta + Theta.T) / 2
    Sigma.append(compute_sigma(Theta))

    for i in range(1, g):
        Temp = Sigma[-1].copy()
        Temp[0:i*p0, 0:i*p0] = np.identity(i*p0)     
        Sigma.append(Temp)

    # Sample
    data_class = []
    for i in range(g):
        data_class.append(multivariate_normal(np.zeros(n), Sigma[i], N[i]))

    return data_class, Sigma

def simulation_random_cov(n, N, g, prep):
    
    Theta = np.zeros((n, n))
    p0 = n // prep
    
    for i in range(prep):
        sub0 = np.random.normal(0.7, 0.1, (p0, p0))
        sub0[sub0<0.5] = 0
        sub0 = (sub0 + sub0) / 2
        Theta[i*p0:(i+1)*p0, i*p0:(i+1)*p0] = sub0
    Theta = (Theta + Theta.T) / 2
    np.fill_diagonal(Theta, 1)

    Sigma = []
    Sigma.append(Theta)
    for i in range(1, g):
        Temp = Sigma[-1].copy()
        Temp[0:i*p0, 0:i*p0] = np.identity(i*p0)            
        Sigma.append(Temp)

    # Sample
    data_class = []
    for i in range(g):
        data_class.append(multivariate_normal(np.zeros(n), Sigma[i], N[i]))

    return data_class, Sigma

def generate_SBM(n, a, b, c, d, k, h, block_sizes, sensitive):
    """
    Generates a stochastic block model.
    
    Parameters:
    n (int): number of elements
    a, b, c, d (float): parameters / probabilities
    k (int): number of clusters
    h (int): number of groups
    block_sizes (list[int]): vector of length k*h with sum(block_sizes)=n
    sensitive (list[int]): sensitive attribute for group membership
    
    Returns:
    A (numpy.ndarray): adjacency matrix of size n x n
    D (numpy.ndarray): degree matrix of A
    F (numpy.ndarray): group membership matrix of size n x (h-1)
    """
    
    if sum(block_sizes) != n or len(block_sizes) != (k * h):
        raise ValueError('wrong input')
    
    # Initialize the adjacency matrix with binomial distribution
    adja = np.random.binomial(1, d, size=(n, n))
    
    # Build the block model
    cumsum_blocks = np.cumsum(block_sizes)
    starts = np.hstack(([0], cumsum_blocks[:-1]))
    
    for ell in range(k):
        for mmm in range(k):
            for ggg in range(h):
                for fff in range(h):
                    if ell == mmm:
                        if ggg == fff:
                            i_start, i_end = starts[ell*h + ggg], cumsum_blocks[ell*h + ggg]
                            adja[i_start:i_end, i_start:i_end] = np.random.binomial(1, a, (block_sizes[ell*h + ggg], block_sizes[ell*h + ggg]))
                        else:
                            i_start, i_end = starts[ell*h + ggg], cumsum_blocks[ell*h + ggg]
                            j_start, j_end = starts[ell*h + fff], cumsum_blocks[ell*h + fff]
                            adja[i_start:i_end, j_start:j_end] = np.random.binomial(1, c, (block_sizes[ell*h + ggg], block_sizes[ell*h + fff]))
                    else:
                        if ggg == fff:
                            i_start, i_end = starts[ell*h + ggg], cumsum_blocks[ell*h + ggg]
                            j_start, j_end = starts[mmm*h + ggg], cumsum_blocks[mmm*h + ggg]
                            adja[i_start:i_end, j_start:j_end] = np.random.binomial(1, b, (block_sizes[ell*h + ggg], block_sizes[mmm*h + ggg]))
    
    # Symmetrize the matrix
    A = np.triu(adja, 1)
    A = A + A.T
    
    # Map the sensitive attributes to a new range starting from 1
    sens_unique = np.unique(sensitive)
    sens_map = {old: new for new, old in enumerate(sens_unique, start=1)}
    sensitive_new = np.vectorize(sens_map.get)(sensitive)
    
    # Build the group membership matrix
    F = np.zeros((n, h - 1))
    for ell in range(h - 1):
        temp = (sensitive_new == (ell + 1))
        F[temp, ell] = 1
        group_size = temp.sum()
        F[:, ell] -= group_size / n
    
    # Compute the degree matrix
    degrees = A.sum(axis=1)
    D = np.diag(degrees)
    
    return A, D, F