# Visualize FKRK Percolation Transition in Ising Spin Glass Model
import numpy as np
from numba import jit
import os
import matplotlib.pyplot as plt
#import numba

J0 = 0
J = 1
N = 8
J_arr = np.random.normal(J0,J,(N,N,3))

# Equilibriation Algorithms
@jit(nopython=True)
def sweep(S_current, J_arr, beta):
    N = len(S_current)
    for _ in range(N*N*N):
        i, j, k = np.random.randint(0, N), np.random.randint(0, N), np.random.randint(0, N)
        # Considering periodic boundary conditions (PBC)
        delta_E = 2 * S_current[i, j, k] * (
                    J_arr[i, j, k, 0] * S_current[(i + 1) % N, j, k] +  # Interaction with the right neighbor
                    J_arr[(i - 1) % N, j, k, 0] * S_current[(i - 1) % N, j, k] +  # Interaction with the left neighbor
                    J_arr[i, j, k, 1] * S_current[i, (j + 1) % N, k] +  # Interaction with the bottom neighbor
                    J_arr[i, (j - 1) % N, k, 1] * S_current[i, (j - 1) % N, k] +  # Interaction with the top neighbor
                    J_arr[i, j, k, 2] * S_current[i, j, (k + 1) % N] +  # Interaction with the front neighbor
                    J_arr[i, j, (k - 1) % N, 2] * S_current[i, j, (k - 1) % N]  # Interaction with the back neighbor
            )
        # Metropolis criterion
        if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
            S_current[i, j, k] *= -1

    return S_current
@jit(nopython=True)
def cluster_flip_wolf(S, J_arr, beta):
    N = len(S)
    cluster_mark = np.zeros((N, N, N), dtype=np.bool_)  # Array to mark sites in the cluster
    
    # Generate random numbers for horizontal, vertical, and depth checks
    random_numbers = np.random.random((N, N, N, 3))

    # Randomly select the starting point
    i, j, k = np.random.randint(0, N, size=3)
    stack = [(i, j, k)]
    cluster_mark[i, j, k] = True

    while stack:
        i_, j_, k_ = stack.pop()

        # Right neighbor with periodic boundary conditions
        right_i, right_j, right_k = (i_ + 1) % N, j_, k_
        if not cluster_mark[right_i, right_j, right_k] and J_arr[i_, j_, k_, 0] * S[right_i, right_j, right_k] * S[i_, j_, k_] > 0 and random_numbers[i_, j_, k_, 0] < 1 - np.exp(-2 * abs(J_arr[i_, j_, k_, 0]) * beta):
            cluster_mark[right_i, right_j, right_k] = True
            stack.append((right_i, right_j, right_k))

        # Left neighbor with periodic boundary conditions
        left_i, left_j, left_k = (i_ - 1) % N, j_, k_
        if not cluster_mark[left_i, left_j, left_k] and J_arr[left_i, left_j, left_k, 0] * S[left_i, left_j, left_k] * S[i_, j_, k_] > 0 and random_numbers[left_i, left_j, left_k, 0] < 1 - np.exp(-2 * abs(J_arr[left_i, left_j, left_k, 0]) * beta):
            cluster_mark[left_i, left_j, left_k] = True
            stack.append((left_i, left_j, left_k))

        # Bottom neighbor with periodic boundary conditions
        bottom_i, bottom_j, bottom_k = i_, (j_ + 1) % N, k_
        if not cluster_mark[bottom_i, bottom_j, bottom_k] and J_arr[i_, j_, k_, 1] * S[bottom_i, bottom_j, bottom_k] * S[i_, j_, k_] > 0 and random_numbers[i_, j_, k_, 1] < 1 - np.exp(-2 * abs(J_arr[i_, j_, k_, 1]) * beta):
            cluster_mark[bottom_i, bottom_j, bottom_k] = True
            stack.append((bottom_i, bottom_j, bottom_k))

        # Top neighbor with periodic boundary conditions
        top_i, top_j, top_k = i_, (j_ - 1) % N, k_
        if not cluster_mark[top_i, top_j, top_k] and J_arr[top_i, top_j, top_k, 1] * S[top_i, top_j, top_k] * S[i_, j_, k_] > 0 and random_numbers[top_i, top_j, top_k, 1] < 1 - np.exp(-2 * abs(J_arr[top_i, top_j, top_k, 1]) * beta):
            cluster_mark[top_i, top_j, top_k] = True
            stack.append((top_i, top_j, top_k))

        # Front neighbor with periodic boundary conditions
        front_i, front_j, front_k = i_, j_, (k_ + 1) % N
        if not cluster_mark[front_i, front_j, front_k] and J_arr[i_, j_, k_, 2] * S[front_i, front_j, front_k] * S[i_, j_, k_] > 0 and random_numbers[i_, j_, k_, 2] < 1 - np.exp(-2 * abs(J_arr[i_, j_, k_, 2]) * beta):
            cluster_mark[front_i, front_j, front_k] = True
            stack.append((front_i, front_j, front_k))

        # Back neighbor with periodic boundary conditions
        back_i, back_j, back_k = i_, j_, (k_ - 1) % N
        if not cluster_mark[back_i, back_j, back_k] and J_arr[back_i, back_j, back_k, 2] * S[back_i, back_j, back_k] * S[i_, j_, k_] > 0 and random_numbers[back_i, back_j, back_k, 2] < 1 - np.exp(-2 * abs(J_arr[back_i, back_j, back_k, 2]) * beta):
            cluster_mark[back_i, back_j, back_k] = True
            stack.append((back_i, back_j, back_k))

    # Flip all spins in the cluster
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if cluster_mark[i, j, k]:
                    S[i, j, k] *= -1

    return S
@jit(nopython=True)
def cluster_flip_sw(S, J_arr, beta):
    clusters = find_clusters(get_fkck(S, J_arr, beta))
    cluster_max = np.max(clusters)
    for cluster_id in range(1, cluster_max + 1):
        if np.random.random() < 0.5:
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        if clusters[i, j, k] == cluster_id:
                            S[i, j, k] *= -1
    return S
@jit(nopython=True)
def cluster_flip_cmrj_sw(S1, S2, J_arr, beta):
    edges = get_cmrj(S1, S2, J_arr, beta)
    clusters_gray = find_clusters(edges.astype(np.bool_))
    
    clusters_gray_max = np.max(clusters_gray)
    clusters_blue = find_clusters((edges // 2).astype(np.bool_))
    clusters_blue_max = np.max(clusters_blue)
    # Shuffle cluster IDs
    cluster_ids_gray = np.random.permutation(np.arange(clusters_gray_max + 1))
    random_arr_blue = np.random.random(clusters_blue_max + 1) < 0.5
    # Select half of the clusters to flip (including zero)
    clusters_to_flip_gray = cluster_ids_gray[:clusters_gray_max // 2 + 1]
    # Flip the spins in the selected clusters
    for cluster_id in clusters_to_flip_gray:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if clusters_gray[i, j, k] == cluster_id:
                        if random_arr_blue[clusters_blue[i, j, k] - 1]:
                            S1[i, j, k] *= -1
                        else:
                            S2[i, j, k] *= -1

    return S1, S2
@jit(nopython=True)
def cluster_flip_houdayer_sw(S1, S2):
    N = len(S1)
    clusters = find_clusters(get_houdayer(S1, S2))
    cluster_max = np.max(clusters)
    for cluster_id in range(1, cluster_max + 1):
        if np.random.random() < 0.5:
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        if clusters[i, j, k] == cluster_id:
                            S1[i, j, k] *= -1
                            S2[i, j, k] *= -1  
    return S1, S2
@jit(nopython=True)
def cluster_flip_jorg_sw(S1, S2, J_arr, beta):
    N = len(S1)
    clusters = find_clusters(get_jorg(S1, S2, J_arr, beta))
    cluster_max = np.max(clusters)
    for cluster_id in range(1, cluster_max + 1):
        if np.random.random() < 0.5:
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        if clusters[i, j, k] == cluster_id:
                            S1[i, j, k] *= -1
                            S2[i, j, k] *= -1   
    return S1, S2
@jit(nopython=True)
def exchange_mc(S_arr, beta_arr, J_arr):
    N = len(beta_arr)
    
    # Precompute total energies for each configuration
    energies = np.array([total_energy(S, J_arr) for S in S_arr])
    
    for i in range(N - 1):
        # Compute the delta energy
        delta = (energies[i] - energies[i + 1]) *  (beta_arr[i + 1] - beta_arr[i])
        
        # Metropolis criterion
        if delta <= 0 or np.random.random() < np.exp(-delta):
            # Swap configurations
            S_arr[i], S_arr[i + 1] = S_arr[i + 1], S_arr[i]
            
            # Update the total energies after swap
            energies[i], energies[i + 1] = energies[i + 1], energies[i]
    return S_arr
# Calculate Clusters and Observables 
@jit(nopython=True)
def total_energy(S, J_arr):
    energy = 0
    N = len(S)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                energy -= (S[i, j, k] * S[(i + 1) % N, j, k] * J_arr[i, j, k, 0] +
                           S[i, j, k] * S[i, (j + 1) % N, k] * J_arr[i, j, k, 1] +
                           S[i, j, k] * S[i, j, (k + 1) % N] * J_arr[i, j, k, 2])
    return energy
@jit(nopython=True)
def get_ql(S1, S2):
    ql = 0
    N = len(S1)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                ql += (S1[i, j, k] * S1[(i + 1) % N, j, k] * S2[i, j, k] * S2[(i + 1) % N, j, k] +
                       S1[i, j, k] * S1[i, (j + 1) % N, k] * S2[i, j, k] * S2[i, (j + 1) % N, k] +
                       S1[i, j, k] * S1[i, j, (k + 1) % N] * S2[i, j, k] * S2[i, j, (k + 1) % N])
    return ql / (3 * N * N * N)
@jit(nopython=True)
def get_fkck_single(S, J_arr, beta, PBC=True):
    N = len(S)
    connection = np.zeros((N, N, N, 3), dtype=np.bool_)
    if not PBC:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # Horizontal bond check
                    if i < N - 1:
                        sxsy_h = S[i, j, k] * S[i + 1, j, k]
                        if sxsy_h * J_arr[i, j, k, 0] > 0 and np.random.random() < 1 - np.exp(-2 * beta * abs(J_arr[i, j, k, 0])):
                            connection[i, j, k, 0] = True
                    if j < N - 1:
                        # Vertical bond check
                        sxsy_v = S[i, j, k] * S[i, j + 1, k]
                        if sxsy_v * J_arr[i, j, k, 1] > 0 and np.random.random() < 1 - np.exp(-2 * beta * abs(J_arr[i, j, k, 1])):
                            connection[i, j, k, 1] = True
                    if k < N - 1:
                        # Depth bond check
                        sxsy_d = S[i, j, k] * S[i, j, k + 1]
                        if sxsy_d * J_arr[i, j, k, 2] > 0 and np.random.random() < 1 - np.exp(-2 * beta * abs(J_arr[i, j, k, 2])):
                            connection[i, j, k, 2] = True
    else:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # Horizontal bond check
                    sxsy_h = S[i, j, k] * S[(i + 1) % N, j, k]
                    if sxsy_h * J_arr[i, j, k, 0] > 0 and np.random.random() < 1 - np.exp(-2 * beta * abs(J_arr[i, j, k, 0])):
                        connection[i, j, k, 0] = True

                    # Vertical bond check
                    sxsy_v = S[i, j, k] * S[i, (j + 1) % N, k]
                    if sxsy_v * J_arr[i, j, k, 1] > 0 and np.random.random() < 1 - np.exp(-2 * beta * abs(J_arr[i, j, k, 1])):
                        connection[i, j, k, 1] = True
                    
                    # Depth bond check
                    sxsy_d = S[i, j, k] * S[i, j, (k + 1) % N]
                    if sxsy_d * J_arr[i, j, k, 2] > 0 and np.random.random() < 1 - np.exp(-2 * beta * abs(J_arr[i, j, k, 2])):
                        connection[i, j, k, 2] = True
    return connection
@jit(nopython=True)
def get_fkck(S1, S2, J_arr, beta, PBC=True):
    N = len(S1)
    connection = np.zeros((N, N, N, 3), dtype=np.bool_)
    
    if not PBC:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if i < N - 1:
                        # Horizontal bond check
                        sxsy_h = S1[i, j, k] * S1[i + 1, j, k] + S2[i, j, k] * S2[i + 1, j, k]
                        if abs(sxsy_h) == 2 and sxsy_h * J_arr[i, j, k, 0] > 0 and np.random.random() < (1 - np.exp(-2 * beta * abs(J_arr[i, j, k, 0]))) ** 2:
                            connection[i, j, k, 0] = True
                    if j < N - 1:
                        # Vertical bond check
                        sxsy_v = S1[i, j, k] * S1[i, j + 1, k] + S2[i, j, k] * S2[i, j + 1, k]
                        if abs(sxsy_v) == 2 and sxsy_v * J_arr[i, j, k, 1] > 0 and np.random.random() < (1 - np.exp(-2 * beta * abs(J_arr[i, j, k, 1]))) ** 2:
                            connection[i, j, k, 1] = True
                    if k < N - 1:
                        # Depth bond check
                        sxsy_d = S1[i, j, k] * S1[i, j, k + 1] + S2[i, j, k] * S2[i, j, k + 1]
                        if abs(sxsy_d) == 2 and sxsy_d * J_arr[i, j, k, 2] > 0 and np.random.random() < (1 - np.exp(-2 * beta * abs(J_arr[i, j, k, 2]))) ** 2:
                            connection[i, j, k, 2] = True
    else:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # Horizontal bond check with PBC
                    sxsy_h = S1[i, j, k] * S1[(i + 1) % N, j, k] + S2[i, j, k] * S2[(i + 1) % N, j, k]
                    if abs(sxsy_h) == 2 and sxsy_h * J_arr[i, j, k, 0] > 0 and np.random.random() < (1 - np.exp(-2 * beta * abs(J_arr[i, j, k, 0]))) ** 2:
                        connection[i, j, k, 0] = True

                    # Vertical bond check with PBC
                    sxsy_v = S1[i, j, k] * S1[i, (j + 1) % N, k] + S2[i, j, k] * S2[i, (j + 1) % N, k]
                    if abs(sxsy_v) == 2 and sxsy_v * J_arr[i, j, k, 1] > 0 and np.random.random() < (1 - np.exp(-2 * beta * abs(J_arr[i, j, k, 1]))) ** 2:
                        connection[i, j, k, 1] = True

                    # Depth bond check with PBC
                    sxsy_d = S1[i, j, k] * S1[i, j, (k + 1) % N] + S2[i, j, k] * S2[i, j, (k + 1) % N]
                    if abs(sxsy_d) == 2 and sxsy_d * J_arr[i, j, k, 2] > 0 and np.random.random() < (1 - np.exp(-2 * beta * abs(J_arr[i, j, k, 2]))) ** 2:
                        connection[i, j, k, 2] = True

    return connection
@jit(nopython=True)
def get_cmrj(S1, S2, J_arr, beta, PBC=True):
    N = len(S1)
    connection = np.zeros((N, N, N, 3), dtype=np.int32)
    
    if not PBC:
        for i in range(N - 1):
            for j in range(N - 1):
                for k in range(N - 1):
                    # Horizontal bond check
                    sxsy_h = S1[i, j, k] * S1[i + 1, j, k] + S2[i, j, k] * S2[i + 1, j, k]
                    J_h = J_arr[i, j, k, 0]
                    if sxsy_h * J_h > 0:
                        if np.random.random() < 1 - np.exp(-4 * beta * abs(J_h)):
                            connection[i, j, k, 0] = 2
                    elif J_h * S1[i, j, k] * S1[i + 1, j, k] > 0 or J_h * S2[i, j, k] * S2[i + 1, j, k] > 0:
                        if np.random.random() < 1 - np.exp(-2 * beta * abs(J_h)):
                            connection[i, j, k, 0] = 1

                    # Vertical bond check
                    sxsy_v = S1[i, j, k] * S1[i, j + 1, k] + S2[i, j, k] * S2[i, j + 1, k]
                    J_v = J_arr[i, j, k, 1]
                    if sxsy_v * J_v > 0:
                        if np.random.random() < 1 - np.exp(-4 * beta * abs(J_v)):
                            connection[i, j, k, 1] = 2
                    elif J_v * S1[i, j, k] * S1[i, j + 1, k] > 0 or J_v * S2[i, j, k] * S2[i, j + 1, k] > 0:
                        if np.random.random() < 1 - np.exp(-2 * beta * abs(J_v)):
                            connection[i, j, k, 1] = 1

                    # Depth bond check
                    sxsy_d = S1[i, j, k] * S1[i, j, k + 1] + S2[i, j, k] * S2[i, j, k + 1]
                    J_d = J_arr[i, j, k, 2]
                    if sxsy_d * J_d > 0:
                        if np.random.random() < 1 - np.exp(-4 * beta * abs(J_d)):
                            connection[i, j, k, 2] = 2
                    elif J_d * S1[i, j, k] * S1[i, j, k + 1] > 0 or J_d * S2[i, j, k] * S2[i, j, k + 1] > 0:
                        if np.random.random() < 1 - np.exp(-2 * beta * abs(J_d)):
                            connection[i, j, k, 2] = 1
    else:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # Horizontal bond check with PBC
                    sxsy_h = S1[i, j, k] * S1[(i + 1) % N, j, k] + S2[i, j, k] * S2[(i + 1) % N, j, k]
                    J_h = J_arr[i, j, k, 0]
                    if sxsy_h * J_h > 0:
                        if np.random.random() < 1 - np.exp(-4 * beta * abs(J_h)):
                            connection[i, j, k, 0] = 2
                    elif J_h * S1[i, j, k] * S1[(i + 1) % N, j, k] > 0 or J_h * S2[i, j, k] * S2[(i + 1) % N, j, k] > 0:
                        if np.random.random() < 1 - np.exp(-2 * beta * abs(J_h)):
                            connection[i, j, k, 0] = 1

                    # Vertical bond check with PBC
                    sxsy_v = S1[i, j, k] * S1[i, (j + 1) % N, k] + S2[i, j, k] * S2[i, (j + 1) % N, k]
                    J_v = J_arr[i, j, k, 1]
                    if sxsy_v * J_v > 0:
                        if np.random.random() < 1 - np.exp(-4 * beta * abs(J_v)):
                            connection[i, j, k, 1] = 2
                    elif J_v * S1[i, j, k] * S1[i, (j + 1) % N, k] > 0 or J_v * S2[i, j, k] * S2[i, (j + 1) % N, k] > 0:
                        if np.random.random() < 1 - np.exp(-2 * beta * abs(J_v)):
                            connection[i, j, k, 1] = 1

                    # Depth bond check with PBC
                    sxsy_d = S1[i, j, k] * S1[i, j, (k + 1) % N] + S2[i, j, k] * S2[i, j, (k + 1) % N]
                    J_d = J_arr[i, j, k, 2]
                    if sxsy_d * J_d > 0:
                        if np.random.random() < 1 - np.exp(-4 * beta * abs(J_d)):
                            connection[i, j, k, 2] = 2
                    elif J_d * S1[i, j, k] * S1[i, j, (k + 1) % N] > 0 or J_d * S2[i, j, k] * S2[i, j, (k + 1) % N] > 0:
                        if np.random.random() < 1 - np.exp(-2 * beta * abs(J_d)):
                            connection[i, j, k, 2] = 1

    return connection
@jit(nopython=True)
def get_houdayer(S1, S2, PBC=True):
    N = len(S1)
    connection = np.zeros((N, N, N, 3), dtype=np.bool_)
    
    if not PBC:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if i < N - 1 and abs(S1[i, j, k] * S1[i + 1, j, k] + S2[i, j, k] * S2[i + 1, j, k]) == 2:
                        connection[i, j, k, 0] = True
                    if j < N - 1 and abs(S1[i, j, k] * S1[i, j + 1, k] + S2[i, j, k] * S2[i, j + 1, k]) == 2:
                        connection[i, j, k, 1] = True
                    if k < N - 1 and abs(S1[i, j, k] * S1[i, j, k + 1] + S2[i, j, k] * S2[i, j, k + 1]) == 2:
                        connection[i, j, k, 2] = True
    else:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # Horizontal bond check with PBC
                    if abs(S1[i, j, k] * S1[(i + 1) % N, j, k] + S2[i, j, k] * S2[(i + 1) % N, j, k]) == 2:
                        connection[i, j, k, 0] = True

                    # Vertical bond check with PBC
                    if abs(S1[i, j, k] * S1[i, (j + 1) % N, k] + S2[i, j, k] * S2[i, (j + 1) % N, k]) == 2:
                        connection[i, j, k, 1] = True

                    # Depth bond check with PBC
                    if abs(S1[i, j, k] * S1[i, j, (k + 1) % N] + S2[i, j, k] * S2[i, j, (k + 1) % N]) == 2:
                        connection[i, j, k, 2] = True

    return connection
@jit(nopython=True)
def get_jorg(S1, S2, J_arr, beta, PBC=True):
    N = len(S1)
    connection = np.zeros((N, N, N, 3), dtype=np.bool_)
    
    if not PBC:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if i < N - 1:
                        # Horizontal bond check
                        sxsy_h = (S1[i, j, k] * S1[i + 1, j, k] + S2[i, j, k] * S2[i + 1, j, k]) * J_arr[i, j, k, 0]
                        if sxsy_h > 0 and np.random.random() < 1 - np.exp(-2 * beta * sxsy_h):
                            connection[i, j, k, 0] = True

                    if j < N - 1:
                        # Vertical bond check
                        sxsy_v = (S1[i, j, k] * S1[i, j + 1, k] + S2[i, j, k] * S2[i, j + 1, k]) * J_arr[i, j, k, 1]
                        if sxsy_v > 0 and np.random.random() < 1 - np.exp(-2 * beta * sxsy_v):
                            connection[i, j, k, 1] = True

                    if k < N - 1:
                        # Depth bond check
                        sxsy_d = (S1[i, j, k] * S1[i, j, k + 1] + S2[i, j, k] * S2[i, j, k + 1]) * J_arr[i, j, k, 2]
                        if sxsy_d > 0 and np.random.random() < 1 - np.exp(-2 * beta * sxsy_d):
                            connection[i, j, k, 2] = True
    else:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # Horizontal bond check with PBC
                    sxsy_h = (S1[i, j, k] * S1[(i + 1) % N, j, k] + S2[i, j, k] * S2[(i + 1) % N, j, k]) * J_arr[i, j, k, 0]
                    if sxsy_h > 0 and np.random.random() < 1 - np.exp(-2 * beta * sxsy_h):
                        connection[i, j, k, 0] = True

                    # Vertical bond check with PBC
                    sxsy_v = (S1[i, j, k] * S1[i, (j + 1) % N, k] + S2[i, j, k] * S2[i, (j + 1) % N, k]) * J_arr[i, j, k, 1]
                    if sxsy_v > 0 and np.random.random() < 1 - np.exp(-2 * beta * sxsy_v):
                        connection[i, j, k, 1] = True

                    # Depth bond check with PBC
                    sxsy_d = (S1[i, j, k] * S1[i, j, (k + 1) % N] + S2[i, j, k] * S2[i, j, (k + 1) % N]) * J_arr[i, j, k, 2]
                    if sxsy_d > 0 and np.random.random() < 1 - np.exp(-2 * beta * sxsy_d):
                        connection[i, j, k, 2] = True

    return connection

# Cluster Calculations
@jit(nopython=True)
def dfs(i, j, k, edges, visited, cluster_id, cluster_array, N):
    stack = [(i, j, k)]

    while stack:
        x, y, z = stack.pop()
        if visited[x, y, z]:
            continue
        visited[x, y, z] = True
        cluster_array[x, y, z] = cluster_id  # Assign cluster ID

        # Periodic Boundary Conditions
        ni_h, nj_h, nk_h = (x + 1) % N, y, z  # Right neighbor
        ni_v, nj_v, nk_v = x, (y + 1) % N, z  # Up neighbor
        ni_d, nj_d, nk_d = x, y, (z + 1) % N  # Front neighbor
        pi_h, pj_h, pk_h = (x - 1) % N, y, z  # Left neighbor
        pi_v, pj_v, pk_v = x, (y - 1) % N, z  # Down neighbor
        pi_d, pj_d, pk_d = x, y, (z - 1) % N  # Back neighbor

        # Check right neighbor
        if edges[x, y, z, 0] and not visited[ni_h, nj_h, nk_h]:
            stack.append((ni_h, nj_h, nk_h))
        # Check up neighbor
        if edges[x, y, z, 1] and not visited[ni_v, nj_v, nk_v]:
            stack.append((ni_v, nj_v, nk_v))
        # Check front neighbor
        if edges[x, y, z, 2] and not visited[ni_d, nj_d, nk_d]:
            stack.append((ni_d, nj_d, nk_d))
        # Check left neighbor
        if edges[pi_h, pj_h, pk_h, 0] and not visited[pi_h, pj_h, pk_h]:
            stack.append((pi_h, pj_h, pk_h))
        # Check down neighbor
        if edges[pi_v, pj_v, pk_v, 1] and not visited[pi_v, pj_v, pk_v]:
            stack.append((pi_v, pj_v, pk_v))
        # Check back neighbor
        if edges[pi_d, pj_d, pk_d, 2] and not visited[pi_d, pj_d, pk_d]:
            stack.append((pi_d, pj_d, pk_d))
@jit(nopython=True)
def find_clusters(edges):
    N = edges.shape[0]
    visited = np.zeros((N, N, N), dtype=np.bool_)
    cluster_array = np.zeros((N, N, N), dtype=np.int16)  # Array to hold cluster IDs
    cluster_id = 0  # Start cluster IDs from 1

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if not visited[i, j, k]:
                    dfs(i, j, k, edges, visited, cluster_id, cluster_array, N)
                    cluster_id += 1  # Increment cluster ID

    return cluster_array
@jit(nopython=True)
def find_cluster_sizes(cluster_matrix):
    N = cluster_matrix.shape[0]
    cluster_sizes = {}

    for i in range(N):
        for j in range(N):
            for k in range(N):
                cluster_id = cluster_matrix[i, j, k]
                if cluster_id not in cluster_sizes:
                    cluster_sizes[cluster_id] = 0
                cluster_sizes[cluster_id] += 1
    
    sizes = list(cluster_sizes.values())
    sizes.sort(reverse=True)
    
    largest_sizes = sizes[:3]
    
    while len(largest_sizes) < 3:
        largest_sizes.append(0)
    
    return np.array(largest_sizes, dtype=np.float32)
@jit(nopython=True)
def contains_spanning_cluster(cluster_matrix):
    N = cluster_matrix.shape[0]
    # Create sets to store unique IDs in each boundary
    left_ids = set()
    right_ids = set()
    top_ids = set()
    bottom_ids = set()
    front_ids = set()
    back_ids = set()
    # Collect IDs from the boundaries, excluding 0
    for i in range(N):
        left_id = cluster_matrix[i, 0, 0]
        left_ids.add(left_id)
        right_id = cluster_matrix[i, N-1, 0]
        right_ids.add(right_id)
        top_id = cluster_matrix[0, i, 0]
        top_ids.add(top_id)
        bottom_id = cluster_matrix[N-1, i, 0]
        bottom_ids.add(bottom_id)
        front_id = cluster_matrix[0, 0, i]
        front_ids.add(front_id)
        back_id = cluster_matrix[N-1, 0, i]
        back_ids.add(back_id)
    
    # Check for intersection in vertical spanning
    if left_ids.intersection(right_ids):
        return 1
    
    # Check for intersection in horizontal spanning
    if top_ids.intersection(bottom_ids):
        return 1

    # Check for intersection in depth spanning
    if front_ids.intersection(back_ids):
        return 1
    
    return 0
@jit(nopython=True)
def count_spanning_clusters(cluster_matrix):
    N = cluster_matrix.shape[0]
    
    # Create sets to store unique IDs in each boundary
    left_ids = set()
    right_ids = set()
    top_ids = set()
    bottom_ids = set()
    front_ids = set()
    back_ids = set()
    
    # Collect IDs from the boundaries
    for i in range(N):
        left_id = cluster_matrix[i, 0, 0]
        left_ids.add(left_id)
        right_id = cluster_matrix[i, N-1, 0]
        right_ids.add(right_id)
        top_id = cluster_matrix[0, i, 0]
        top_ids.add(top_id)
        bottom_id = cluster_matrix[N-1, i, 0]
        bottom_ids.add(bottom_id)
        front_id = cluster_matrix[0, 0, i]
        front_ids.add(front_id)
        back_id = cluster_matrix[N-1, 0, i]
        back_ids.add(back_id)
    
    # Track clusters spanning in each direction
    spanning_clusters = 0
    
    # Check for clusters spanning vertically
    vertical_spanning_clusters = left_ids.intersection(right_ids)
    spanning_clusters += len(vertical_spanning_clusters)
    
    # Check for clusters spanning horizontally
    horizontal_spanning_clusters = top_ids.intersection(bottom_ids)
    spanning_clusters += len(horizontal_spanning_clusters)
    
    # Check for clusters spanning in depth
    depth_spanning_clusters = front_ids.intersection(back_ids)
    spanning_clusters += len(depth_spanning_clusters)
    
    return spanning_clusters

# Collect Cases 
@jit(nopython=True)
def get_sizes_and_R_houdayer(numSweeps, num_ave, beta_arr, N):
    Sarr1 = np.zeros((len(beta_arr), N, N, N), dtype=np.int8)
    Sarr2 = np.zeros((len(beta_arr), N, N, N), dtype=np.int8)
    
    for i in range(len(beta_arr)):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    Sarr1[i, j, k, l] = -1 + 2 * np.random.randint(2)
                    Sarr2[i, j, k, l] = -1 + 2 * np.random.randint(2)

    J_arr = np.random.normal(J0, J, (N, N, N, 3))  # Adjusted for 3D, assuming 3D interactions

    R_list = np.zeros(len(beta_arr), dtype=np.float32)
    sizes_list = np.zeros((len(beta_arr), 3), dtype=np.float32)
    e_arr1 = np.zeros(len(beta_arr), dtype=np.float32)
    e_arr2 = np.zeros(len(beta_arr), dtype=np.float32)
    
    for j in range(numSweeps):
        for i in range(len(beta_arr)):
            # Perform cluster flips and sweeps
            if j % 2 == 1:
                Sarr1[i] = sweep(Sarr1[i], J_arr, beta_arr[i])
                Sarr1[i] = cluster_flip_wolf(Sarr1[i], J_arr, beta_arr[i])
            else:
                Sarr2[i] = sweep(Sarr2[i], J_arr, beta_arr[i])
                Sarr2[i] = cluster_flip_wolf(Sarr2[i], J_arr, beta_arr[i])

            # Perform specific cluster flips based on conditions
            Sarr1[i], Sarr2[i] = cluster_flip_cmrj_sw(Sarr1[i], Sarr2[i], J_arr, beta_arr[i])
        
        # Exchange Monte Carlo updates
        Sarr1, Sarr2 = exchange_mc(Sarr1, beta_arr, J_arr), exchange_mc(Sarr2, beta_arr, J_arr)
        
        for i in range(len(beta_arr)):
            if j > numSweeps - num_ave:
                # Calculate Spanning Probability
                clusters = find_clusters(get_houdayer(Sarr1[i], Sarr2[i], False))
                R_list[i] += contains_spanning_cluster(clusters) / num_ave
                
                sizes = find_cluster_sizes(find_clusters(get_houdayer(Sarr1[i], Sarr2[i], True)))
                for k in range(3):
                    sizes_list[i, k] += sizes[k] / num_ave / N**3  # Adjusted for 3D
                
                e_arr1[i] += get_ql(Sarr1[i], Sarr2[i]) / num_ave
                e_arr2[i] += total_energy(Sarr2[i], J_arr) / num_ave / N**3  # Adjusted for 3D

    return R_list, sizes_list, e_arr1, e_arr2
@jit(nopython=True)
def get_sizes_and_R_fkck(numSweeps, num_ave, beta_arr, N):
    Sarr1 = np.zeros((len(beta_arr), N, N, N), dtype=np.int8)
    Sarr2 = np.zeros((len(beta_arr), N, N, N), dtype=np.int8)
    
    for i in range(len(beta_arr)):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    Sarr1[i, j, k, l] = -1 + 2 * np.random.randint(2)
                    Sarr2[i, j, k, l] = -1 + 2 * np.random.randint(2)

    J_arr = np.random.normal(J0, J, (N, N, N, 3))  # Adjusted for 3D, assuming 3D interactions

    R_list = np.zeros(len(beta_arr), dtype=np.float32)
    sizes_list = np.zeros((len(beta_arr), 3), dtype=np.float32)
    e_arr1 = np.zeros(len(beta_arr), dtype=np.float32)
    e_arr2 = np.zeros(len(beta_arr), dtype=np.float32)
    
    for j in range(numSweeps):
        for i in range(len(beta_arr)):
            # Perform specific cluster flips
            if j % 100 == 1:
                Sarr1[i], Sarr2[i] = cluster_flip_jorg_sw(Sarr1[i], Sarr2[i], J_arr, beta_arr[i])
            elif j % 100 == 51:
                Sarr1[i], Sarr2[i] = cluster_flip_houdayer_sw(Sarr1[i], Sarr2[i])

            Sarr1[i] = sweep(Sarr1[i], J_arr, beta_arr[i])
            Sarr2[i] = sweep(Sarr2[i], J_arr, beta_arr[i])
            Sarr1[i], Sarr2[i] = cluster_flip_wolf(Sarr1[i], J_arr, beta_arr[i]), cluster_flip_wolf(Sarr2[i], J_arr, beta_arr[i])
            
            if j > numSweeps - num_ave:
                # Calculate Spanning Probability
                clusters = find_clusters(get_fkck(Sarr1[i], Sarr2[i], J_arr, beta_arr[i], False))
                R_list[i] += contains_spanning_cluster(clusters) / num_ave
                
                sizes = find_cluster_sizes(clusters)
                for k in range(3):
                    sizes_list[i, k] += sizes[k] / num_ave / N**3  # Adjusted for 3D
                
                e_arr1[i] += total_energy(Sarr1[i], J_arr) / num_ave / N**3  # Adjusted for 3D
                e_arr2[i] += get_ql(Sarr1[i], Sarr2[i]) / num_ave

        Sarr1 = exchange_mc(Sarr1, beta_arr, J_arr)
        Sarr2 = exchange_mc(Sarr2, beta_arr, J_arr)

    return R_list, sizes_list, e_arr1, e_arr2
@jit(nopython=True)
def get_sizes_and_R_single(numSweeps, num_ave, beta_arr, N):
    Sarr1 = np.zeros((len(beta_arr), N, N, N), dtype=np.int8)
    Sarr2 = np.zeros((len(beta_arr), N, N, N), dtype=np.int8)
    
    for i in range(len(beta_arr)):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    Sarr1[i, j, k, l] = -1 + 2 * np.random.randint(2)
                    Sarr2[i, j, k, l] = -1 + 2 * np.random.randint(2)

    J_arr = np.random.normal(J0, J, (N, N, N, 3))  # Adjusted for 3D, assuming 3D interactions

    R_list1 = np.zeros(len(beta_arr), dtype=np.float32)
    size1 = np.zeros(len(beta_arr), dtype=np.float32)
    R_list2 = np.zeros(len(beta_arr), dtype=np.float32)
    size2 = np.zeros(len(beta_arr), dtype=np.float32)

    for j in range(numSweeps):
        for i in range(len(beta_arr)):
            Sarr1[i] = sweep(Sarr1[i], J_arr, beta_arr[i])
            Sarr2[i] = sweep(Sarr2[i], J_arr, beta_arr[i])
            Sarr1[i] = cluster_flip_wolf(Sarr1[i], J_arr, beta_arr[i])
            Sarr2[i] = cluster_flip_wolf(Sarr2[i], J_arr, beta_arr[i])
            
            if j % 2 == 1:
                Sarr1[i], Sarr2[i] = cluster_flip_jorg_sw(Sarr1[i], Sarr2[i], J_arr, beta_arr[i])
            else:
                Sarr1[i], Sarr2[i] = cluster_flip_houdayer_sw(Sarr1[i], Sarr2[i])
            
            if j > numSweeps - num_ave:
                # Calculate Spanning Probability and Sizes
                clusters1 = find_clusters(get_fkck(Sarr1[i], Sarr2[i], J_arr, beta_arr[i], False))
                R_list1[i] += contains_spanning_cluster(clusters1) / num_ave
                size1[i] += find_cluster_sizes(find_clusters(get_fkck(Sarr1[i], Sarr2[i], J_arr, beta_arr[i], True)))[0] / num_ave / N**3
                
                clusters2 = find_clusters(get_fkck(Sarr2[i], Sarr1[i], J_arr, beta_arr[i], False))
                R_list2[i] += contains_spanning_cluster(clusters2) / num_ave
                size2[i] += find_cluster_sizes(find_clusters(get_fkck(Sarr2[i], Sarr1[i], J_arr, beta_arr[i], True)))[0] / num_ave / N**3

        Sarr1 = exchange_mc(Sarr1, beta_arr, J_arr)
        Sarr2 = exchange_mc(Sarr2, beta_arr, J_arr)

    return R_list1, R_list2, size1, size2
@jit(nopython=True)
def get_sizes_and_R_tot(numSweeps, num_ave, beta_arr, N):
    # Initialize arrays
    Sarr1 = np.zeros((len(beta_arr), N, N, N), dtype=np.int8)
    Sarr2 = np.zeros((len(beta_arr), N, N, N), dtype=np.int8)
    
    # Fill the arrays with random spin configurations
    for i in range(len(beta_arr)):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    Sarr1[i, j, k, l] = -1 + 2 * np.random.randint(2)
                    Sarr2[i, j, k, l] = -1 + 2 * np.random.randint(2)

    J_arr = np.random.normal(J0, J, (N, N, N, 3))  # Adjusted for 3D lattice

    # Initialize result lists
    R_list_houdayer = np.zeros(len(beta_arr), dtype=np.float32)
    sizes_list_houdayer = np.zeros((len(beta_arr), 3), dtype=np.float32)
    R_list_fkck = np.zeros(len(beta_arr), dtype=np.float32)
    sizes_list_fkck = np.zeros((len(beta_arr), 3), dtype=np.float32)
    R_list_cmrj = np.zeros(len(beta_arr), dtype=np.float32)
    sizes_list_cmrj = np.zeros((len(beta_arr), 3), dtype=np.float32)
    e_arr = np.zeros(len(beta_arr),dtype=np.float32)
    ql_arr = np.zeros(len(beta_arr),dtype=np.float32)
    num_wrap_cmrj = np.zeros(len(beta_arr),dtype=np.float32)
    num_wrap_fkck = np.zeros(len(beta_arr),dtype=np.float32)

    for j in range(numSweeps):
        for i in range(len(beta_arr)):
            # Perform sweeps
            Sarr1[i] = sweep(Sarr1[i], J_arr, beta_arr[i])
            Sarr2[i] = sweep(Sarr2[i], J_arr, beta_arr[i])
            Sarr1[i], Sarr2[i] = cluster_flip_wolf(Sarr1[i], J_arr, beta_arr[i]), cluster_flip_wolf(Sarr2[i], J_arr, beta_arr[i])
            
            if j % 2 == 1:
                Sarr1[i], Sarr2[i] = cluster_flip_cmrj_sw(Sarr1[i], Sarr2[i], J_arr, beta_arr[i])
            #else:
            #    Sarr1[i], Sarr2[i] = cluster_flip_houdayer_sw(Sarr1[i], Sarr2[i])
            
            if j > numSweeps - num_ave:
                # Calculate Observables Houdayer
                clusters_houdayer = find_clusters(get_houdayer(Sarr1[i], Sarr2[i], False))
                R_list_houdayer[i] += contains_spanning_cluster(clusters_houdayer) / num_ave
                sizes_houdayer = find_cluster_sizes(find_clusters(get_houdayer(Sarr1[i], Sarr2[i], True)))
                for k in range(3):
                    sizes_list_houdayer[i, k] += sizes_houdayer[k] / num_ave / N**3  # Adjusted for 3D
                
                # Calculate Observables FKCK
                clusters_fkck = find_clusters(get_fkck(Sarr1[i], Sarr2[i], J_arr, beta_arr[i], False))
                R_list_fkck[i] += contains_spanning_cluster(clusters_fkck) / num_ave
                sizes_fkck = find_cluster_sizes(find_clusters(get_fkck(Sarr1[i], Sarr2[i], J_arr, beta_arr[i], True)))
                for k in range(3):
                    sizes_list_fkck[i, k] += sizes_fkck[k] / num_ave / N**3  # Adjusted for 3D
                num_wrap_fkck += count_spanning_clusters(find_clusters(get_fkck(Sarr1[i], Sarr2[i], J_arr, beta_arr[i], True)))/num_ave
                # Calculate Observables CMRJ
                clusters_cmrj = find_clusters(get_jorg(Sarr1[i], Sarr2[i], J_arr, beta_arr[i], False))
                R_list_cmrj[i] += contains_spanning_cluster(clusters_cmrj) / num_ave
                sizes_cmrj = find_cluster_sizes(find_clusters(get_jorg(Sarr1[i], Sarr2[i], J_arr, beta_arr[i])))
                for k in range(3):
                    sizes_list_cmrj[i, k] += sizes_cmrj[k] / num_ave / N**3  # Adjusted for 3D
                num_wrap_cmrj += count_spanning_clusters(find_clusters(get_cmrj(Sarr1[i], Sarr2[i], J_arr, beta_arr[i], True)))/num_ave
                e_arr[i] += total_energy(Sarr1[i],J_arr)/num_ave/N**3/3
                ql_arr[i] += beta_arr[i]*(get_ql(Sarr1[i],Sarr2[i])-1)/num_ave
        Sarr1 = exchange_mc(Sarr1, beta_arr, J_arr)
        Sarr2 = exchange_mc(Sarr2, beta_arr, J_arr)

    return R_list_houdayer, sizes_list_houdayer, R_list_fkck, sizes_list_fkck, R_list_cmrj, sizes_list_cmrj,e_arr,ql_arr,num_wrap_cmrj,num_wrap_fkck

# Cluster Calculation Methods for 2D Cross Sections
@jit(nopython=True)
def dfs_2D(i, j, edges, visited, cluster_id, cluster_array, N):
    stack = [(i, j)]

    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue
        visited[x, y] = True
        cluster_array[x, y] = cluster_id  # Assign cluster ID

        # Periodic Boundary Conditions
        ni_h, nj_h = (x + 1) % N, y  # Right neighbor
        ni_v, nj_v = x, (y + 1) % N  # Down neighbor
        pi_h, pj_h = (x - 1) % N, y  # Left neighbor
        pi_v, pj_v = x, (y - 1) % N  # Up neighbor

        # Check right neighbor
        if edges[x, y, 0] and not visited[ni_h, nj_h]:
            stack.append((ni_h, nj_h))
        # Check down neighbor
        if edges[x, y, 1] and not visited[ni_v, nj_v]:
            stack.append((ni_v, nj_v))
        # Check left neighbor
        if edges[pi_h, pj_h, 0] and not visited[pi_h, pj_h]:
            stack.append((pi_h, pj_h))
        # Check up neighbor
        if edges[pi_v, pj_v, 1] and not visited[pi_v, pj_v]:
            stack.append((pi_v, pj_v))
@jit(nopython=True)
def get_fkck_2D(S1, S2, J_arr, beta,PBC=True):
    N = len(S1)
    connection = np.zeros((N, N, 2), dtype=np.bool_)
    if not PBC:
        for i in range(N - 1):
            for j in range(N - 1):
                # Horizontal bond check
                sxsy_h = S1[i, j] * S1[i + 1, j] + S2[i, j] * S2[i + 1, j]
                if abs(sxsy_h) == 2 and sxsy_h * J_arr[i, j, 0] > 0 and np.random.random() < (1 - np.exp(-2 * beta * abs(J_arr[i, j, 0]))) ** 2:
                    connection[i, j, 0] = True

                # Vertical bond check
                sxsy_v = S1[i, j] * S1[i, j + 1] + S2[i, j] * S2[i, j + 1]
                if abs(sxsy_v) == 2 and sxsy_v * J_arr[i, j, 1] > 0 and np.random.random() < (1 - np.exp(-2 * beta * abs(J_arr[i, j, 1]))) ** 2:
                    connection[i, j, 1] = True
    else:
        for i in range(N):
            for j in range(N):
                # Horizontal bond check
                sxsy_h = S1[i, j] * S1[(i + 1)%N, j] + S2[i, j] * S2[(i + 1)%N, j]
                if abs(sxsy_h) == 2 and sxsy_h * J_arr[i, j, 0] > 0 and np.random.random() < (1 - np.exp(-2 * beta * abs(J_arr[i, j, 0])))**2:
                    connection[i, j, 0] = True

                # Vertical bond check
                sxsy_v = S1[i, j] * S1[i, (j + 1)%N] + S2[i, j] * S2[i, (j + 1)%N]
                if abs(sxsy_v) == 2 and sxsy_v * J_arr[i, j, 1] > 0 and np.random.random() < (1 - np.exp(-2 * beta * abs(J_arr[i, j, 1]))) ** 2:
                    connection[i, j, 1] = True 
    
    return connection
@jit(nopython=True)
def get_cmrj_2D(S1, S2, J_arr, beta, PBC=True):
    N = len(S1)
    connection = np.zeros((N, N, 2), dtype=np.int32)
    
    if not PBC:
        for i in range(N - 1):
            for j in range(N - 1):
                # Horizontal bond check
                sxsy_h = S1[i, j] * S1[i + 1, j] + S2[i, j] * S2[i + 1, j]
                J_h = J_arr[i, j, 0]
                if sxsy_h * J_h > 0:
                    if np.random.random() < 1 - np.exp(-4 * beta * abs(J_h)):
                        connection[i, j, 0] = 2
                elif J_h * S1[i, j] * S1[i + 1, j] > 0 or J_h * S2[i, j] * S2[i + 1, j] > 0:
                    if np.random.random() < 1 - np.exp(-2 * beta * abs(J_h)):
                        connection[i, j, 0] = 1

                # Vertical bond check
                sxsy_v = S1[i, j] * S1[i, j + 1] + S2[i, j] * S2[i, j + 1]
                J_v = J_arr[i, j, 1]
                if sxsy_v * J_v > 0:
                    if np.random.random() < 1 - np.exp(-4 * beta * abs(J_v)):
                        connection[i, j, 1] = 2
                elif J_v * S1[i, j] * S1[i, j + 1] > 0 or J_v * S2[i, j] * S2[i, j + 1] > 0:
                    if np.random.random() < 1 - np.exp(-2 * beta * abs(J_v)):
                        connection[i, j, 1] = 1

    else:
        for i in range(N):
            for j in range(N):
                # Horizontal bond check with PBC
                sxsy_h = S1[i, j] * S1[(i + 1) % N, j] + S2[i, j] * S2[(i + 1) % N, j]
                J_h = J_arr[i, j, 0]
                if sxsy_h * J_h > 0:
                    if np.random.random() < 1 - np.exp(-4 * beta * abs(J_h)):
                        connection[i, j, 0] = 2
                elif J_h * S1[i, j] * S1[(i + 1) % N, j] > 0 or J_h * S2[i, j] * S2[(i + 1) % N, j] > 0:
                    if np.random.random() < 1 - np.exp(-2 * beta * abs(J_h)):
                        connection[i, j, 0] = 1

                # Vertical bond check with PBC
                sxsy_v = S1[i, j] * S1[i, (j + 1) % N] + S2[i, j] * S2[i, (j + 1) % N]
                J_v = J_arr[i, j, 1]
                if sxsy_v * J_v > 0:
                    if np.random.random() < 1 - np.exp(-4 * beta * abs(J_v)):
                        connection[i, j, 1] = 2
                elif J_v * S1[i, j] * S1[i, (j + 1) % N] > 0 or J_v * S2[i, j] * S2[i, (j + 1) % N] > 0:
                    if np.random.random() < 1 - np.exp(-2 * beta * abs(J_v)):
                        connection[i, j, 1] = 1

    return connection
@jit(nopython=True)
def get_houdayer_2D(S1, S2,PBC=True):
    N = len(S1)
    connection = np.zeros((N, N, 2), dtype=np.bool_)
    if not PBC:
        for i in range(N):
            for j in range(N):
                if i < N-1 and abs(S1[i, j] * S1[i + 1, j] + S2[i, j] * S2[i + 1, j]) == 2:
                    connection[i, j, 0] = True
                if j < N-1 and abs(S1[i, j] * S1[i, j + 1] + S2[i, j] * S2[i, j + 1]) == 2:
                    connection[i, j, 1] = True
    else:
        for i in range(N):
            for j in range(N):
                # Horizontal bond check
                if abs(S1[i, j] * S1[(i + 1)%N, j] + S2[i, j] * S2[(i + 1)%N, j]) == 2:
                    connection[i, j, 0] = True

                # Vertical bond check
                if abs(S1[i, j] * S1[i, (j + 1)%N] + S2[i, j] * S2[i, (j + 1)%N]) == 2:
                    connection[i, j, 1] = True
    return connection
@jit(nopython=True)
def get_jorg_2D(S1, S2, J_arr, beta,PBC=True):
    N = len(S1)
    connection = np.zeros((N, N, 2), dtype=np.bool_)
    if not PBC:
        for i in range(N):
            for j in range(N):
                # Horizontal bond check
                if i < N-1:
                    sxsy_h = (S1[i, j] * S1[i + 1, j] + S2[i, j] * S2[i + 1, j]) * J_arr[i, j, 0]
                    if sxsy_h > 0 and np.random.random() < 1 - np.exp(-2 * beta * sxsy_h):
                            connection[i, j, 0] = True

                # Vertical bond check
                if j < N-1:
                    sxsy_v = (S1[i, j] * S1[i, j + 1] + S2[i, j] * S2[i, j + 1]) *  J_arr[i, j, 1]
                    if sxsy_v  > 0 and np.random.random() < 1 - np.exp(-2 * beta * sxsy_v):
                            connection[i, j, 1] = True
    else:
        for i in range(N):
            for j in range(N):
                # Horizontal bond check
                sxsy_h = (S1[i, j] * S1[(i + 1)%N, j] + S2[i, j] * S2[(i + 1)%N, j]) * J_arr[i, j, 0]
                if sxsy_h > 0 and np.random.random() < 1 - np.exp(-2 * beta * sxsy_h):
                        connection[i, j, 0] = True

                # Vertical bond check
                sxsy_v = (S1[i, j] * S1[i, (j + 1)%N] + S2[i, j] * S2[i, (j + 1)%N]) *  J_arr[i, j, 1]
                if sxsy_v > 0 and np.random.random() < 1 - np.exp(-2 * beta * sxsy_v):
                        connection[i, j, 1] = True
    
    return connection
@jit(nopython=True)
def find_clusters_2D(edges):
    N = edges.shape[0]
    visited = np.zeros((N, N), dtype=np.bool_)
    cluster_array = np.zeros((N, N), dtype=np.int16)  # Array to hold cluster IDs
    cluster_id = 0  # Start cluster IDs from 1

    for i in range(N):
        for j in range(N):
            if not visited[i, j]:
                dfs_2D(i, j, edges, visited, cluster_id, cluster_array, N)
                cluster_id += 1  # Increment cluster ID

    return cluster_array
@jit(nopython=True)
def find_cluster_sizes_2D(cluster_matrix):
    """
    Given a matrix where each cell contains a cluster ID, return the sizes of the largest three clusters.
    """
    N = cluster_matrix.shape[0]
    
    # Dictionary to count sizes of each cluster ID
    cluster_sizes = {}
    
    # Calculate sizes of each cluster
    for i in range(N):
        for j in range(N):
            cluster_id = cluster_matrix[i, j]
            if cluster_id not in cluster_sizes:
                cluster_sizes[cluster_id] = 0
            cluster_sizes[cluster_id] += 1
    
    # Convert cluster sizes to a list and sort in descending order
    sizes = list(cluster_sizes.values())
    sizes.sort(reverse=True)
    
    # Get the largest three sizes
    largest_sizes = sizes[:3]
    
    # Pad with zeros if there are fewer than three clusters
    while len(largest_sizes) < 3:
        largest_sizes.append(0)
    
    # Convert list to numpy array for return
    return np.array(largest_sizes, dtype=np.float32)
@jit(nopython=True)
def contains_spanning_cluster_2d(cluster_matrix):
    N = cluster_matrix.shape[0]
    # Create sets to store unique IDs in each boundary
    left_ids = set()
    right_ids = set()
    top_ids = set()
    bottom_ids = set()
    
    # Collect IDs from the boundaries, excluding 0
    for i in range(N):
        left_id = cluster_matrix[i, 0]
        left_ids.add(left_id)
        right_id = cluster_matrix[i, N-1]
        right_ids.add(right_id)
        top_id = cluster_matrix[0, i]
        top_ids.add(top_id)
        bottom_id = cluster_matrix[N-1, i]
        bottom_ids.add(bottom_id)
    
    # Check for intersection in vertical spanning
    if left_ids.intersection(right_ids):
        return 1
    
    # Check for intersection in horizontal spanning
    if top_ids.intersection(bottom_ids):
        return 1
    
    return 0
@jit(nopython=True)
def count_spanning_clusters_2d(cluster_matrix):
    N = cluster_matrix.shape[0]
    
    # Create sets to store unique IDs in each boundary
    left_ids = set()
    right_ids = set()
    top_ids = set()
    bottom_ids = set()
    
    # Collect IDs from the boundaries
    for i in range(N):
        # Left boundary (first column)
        left_id = cluster_matrix[i, 0]
        left_ids.add(left_id)
        # Right boundary (last column)
        right_id = cluster_matrix[i, N-1]
        right_ids.add(right_id)
        # Top boundary (first row)
        top_id = cluster_matrix[0, i]
        top_ids.add(top_id)
        # Bottom boundary (last row)
        bottom_id = cluster_matrix[N-1, i]
        bottom_ids.add(bottom_id)
    
    # Track clusters spanning in each direction
    spanning_clusters = 0
    
    # Check for clusters spanning vertically
    vertical_spanning_clusters = left_ids.intersection(right_ids)
    spanning_clusters += len(vertical_spanning_clusters)
    
    # Check for clusters spanning horizontally
    horizontal_spanning_clusters = top_ids.intersection(bottom_ids)
    spanning_clusters += len(horizontal_spanning_clusters)
    
    return spanning_clusters

numSweeps = 20000
num_ave = numSweeps/2
# N=8
beta_arr = np.asarray([1.1, 1.002704507512521, 0.9226711185308849, 0.8536227045075127, 0.7939899833055093, 0.7406343906510853, 0.6935559265442405, 0.6511853088480802, 0.6135225375626044, 0.5789983305509182, 0.546043405676127, 0.5162270450751253, 0.4879799666110184, 0.4613021702838064, 0.43462437395659437, 0.40794657762938236, 0.38126878130217035, 0.3545909849749583, 0.3294824707846411, 0.30437395659432387, 0.2792654424040067, 0.2557262103505843, 0.23218697829716195, 0.20864774624373958, 0.1851085141903172, 0.16156928213689484]
)
R_list_houdayer,sizes_list_houdayer,R_list_fkck,sizes_list_fkck,R_list_cmrj,sizes_list_cmrj,e_arr,ql_arr,num_wrap_cmrj,num_wrap_fkck = get_sizes_and_R_tot(numSweeps,num_ave,beta_arr,N)


# Define the folder name
folder_name = 'data'

# Create the folder if it doesn't exi#st
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Define the file paths
file_path_houdayer = os.path.join(folder_name,f'Houdayer_R_N{N}.txt')
file_path_sizes_houdayer = os.path.join(folder_name, f'sizes_list_houdayer_{N}.txt')
file_path_FKCK = os.path.join(folder_name,f'FKCK_R_N{N}.txt')
file_path_sizes_FKCK = os.path.join(folder_name, f'sizes_list_FKCK_{N}.txt')
file_path_CMRJ = os.path.join(folder_name,f'CMRJ_R_N{N}.txt')
file_path_sizes_CMRJ = os.path.join(folder_name, f'sizes_list_CMRJ_{N}.txt')
file_path_E= os.path.join(folder_name,f'E_{N}.txt')
file_path_QL = os.path.join(folder_name, f'QL_{N}.txt')
file_path_NFKCK= os.path.join(folder_name,f'E_{N}.txt')
file_path_NCMRJ= os.path.join(folder_name, f'QL_{N}.txt')
# Save the files into the folder
np.savetxt(file_path_houdayer, np.column_stack((beta_arr, R_list_houdayer)), fmt='%f', delimiter=',', header='Beta,R')
np.savetxt(file_path_sizes_houdayer, np.column_stack((beta_arr, sizes_list_houdayer)), fmt='%f', delimiter=',', header='Beta,Rho1,Rho2,Rho3')
np.savetxt(file_path_FKCK, np.column_stack((beta_arr, R_list_fkck)), fmt='%f', delimiter=',', header='Beta,R')
np.savetxt(file_path_sizes_FKCK, np.column_stack((beta_arr, sizes_list_fkck)), fmt='%f', delimiter=',', header='Beta,Rho1,Rho2,Rho3')
np.savetxt(file_path_CMRJ, np.column_stack((beta_arr, R_list_cmrj)), fmt='%f', delimiter=',', header='Beta,R')
np.savetxt(file_path_sizes_CMRJ, np.column_stack((beta_arr, sizes_list_cmrj)), fmt='%f', delimiter=',', header='Beta,Rho1,Rho2,Rho3')
np.savetxt(file_path_E, np.column_stack((beta_arr, e_arr)), fmt='%f', delimiter=',', header='Beta,E')
np.savetxt(file_path_QL, np.column_stack((beta_arr, ql_arr)), fmt='%f', delimiter=',', header='Beta,Beta*(QL-1)')
np.savetxt(file_path_NFKCK, np.column_stack((beta_arr, num_wrap_fkck)), fmt='%f', delimiter=',', header='Beta,FKCK')
np.savetxt(file_path_NCMRJ, np.column_stack((beta_arr, num_wrap_cmrj)), fmt='%f', delimiter=',', header='Beta,CMRJ')

plt.scatter(beta_arr,ql_arr,label='Beta*(Ql-1)')
plt.scatter(beta_arr,e_arr,label='Energy')
plt.savefig('data/comp.png')
plt.scatter(beta_arr,sizes_list_houdayer[:,0])
plt.scatter(beta_arr,sizes_list_houdayer[:,1])
plt.scatter(beta_arr,sizes_list_houdayer[:,2])
plt.savefig('data/houdayersize.png')
plt.scatter(beta_arr,sizes_list_fkck[:,0])
plt.scatter(beta_arr,sizes_list_fkck[:,1])
plt.scatter(beta_arr,sizes_list_fkck[:,2])
plt.savefig('data/fkcksize.png')
plt.scatter(beta_arr,sizes_list_cmrj[:,0])
plt.scatter(beta_arr,sizes_list_cmrj[:,1])
plt.scatter(beta_arr,sizes_list_cmrj[:,2])
plt.savefig('data/fkcksize.png')
plt.scatter(beta_arr,R_list_houdayer)
plt.savefig('data/Houdayer_R.png')
plt.scatter(beta_arr,R_list_fkck)
plt.savefig('data/FKCK_R.png')
plt.scatter(beta_arr,R_list_cmrj)
plt.savefig('data/CMRJ_R.png')
plt.scatter(beta_arr,num_wrap_cmrj)
plt.savefig('data/cmrj.png')
plt.scatter(beta_arr,num_wrap_fkck)
plt.savefig('data/fkck.png')
