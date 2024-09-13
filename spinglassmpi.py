import numpy as np
from numba import jit
import os
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys
from numpy import linalg as LA
from numpy.fft import fftshift

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
@jit(nopython=True)
def sweep(S_current, J_arr, beta):
    N = len(S_current)
    for i in range(N):
        for j in range(N):
            for k in range(N):
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
@jit(nopython=True)
def cluster_flip_wolf_houdayer(S1, S2):
    N = len(S1)
    cluster_mark = np.zeros((N, N, N), dtype=np.bool_)  # Array to mark sites in the cluster
    
    # Randomly select the starting point
    i, j, k = np.random.randint(0, N, size=3)
    stack = [(i, j, k)]
    cluster_mark[i, j, k] = True

    while stack:
        i_, j_, k_ = stack.pop()

        # Check all 6 neighbors
        neighbors = [
            ((i_ + 1) % N, j_, k_),  # Right
            ((i_ - 1) % N, j_, k_),  # Left
            (i_, (j_ + 1) % N, k_),  # Up
            (i_, (j_ - 1) % N, k_),  # Down
            (i_, j_, (k_ + 1) % N),  # Front
            (i_, j_, (k_ - 1) % N)   # Back
        ]
        
        for neighbor in neighbors:
            ni, nj, nk = neighbor
            
            # Check if the neighbor is already in the cluster
            if not cluster_mark[ni, nj, nk]:
                # Check the Houdayer condition for bond overlap
                if abs(S1[ni, nj, nk] * S1[i_, j_, k_] + S2[ni, nj, nk] * S2[i_, j_, k_]) == 2:
                    # Calculate the probability of adding this bond:
                        cluster_mark[ni, nj, nk] = True
                        stack.append((ni, nj, nk))

    # Flip all spins in the cluster
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if cluster_mark[i, j, k]:
                    S1[i, j, k] *= -1
                    S2[i, j, k] *= -1

    #return S1,S2
@jit(nopython=True)
def cluster_flip_sw(S, J_arr, beta):
    N = len(S)
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
    N = len(S1)
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
def exchange_mc(S_current, betas, J_arr, comm, rank, size, timestep):
    # Determine partner
    partner = (rank + 1 if (rank % 2) else rank - 1)
    comm.Barrier()  # Synchronize after MC step
    if 0 <= partner < size:
        N = S_current.shape[-1]  # Assuming S_current is of shape (2, N, N, N)

        # Buffers for energy and acceptance
        energy_buffer = np.empty(2, dtype=np.float64)  # Only used by the non-initiating rank
        accepted = np.array([False, False], dtype=bool)  # Used by both ranks

        if rank % 2 == 0:  # Initiating ranks
            # Perform energy comparison and determine acceptance
            # Receive energy information from the partner
            comm.Recv(energy_buffer, source=partner)

            accepted = perform_exchange(S_current, betas, J_arr, comm, rank, partner, N, energy_buffer)

            # Send acceptance info
            comm.Send(accepted, dest=partner)

            # Send S_current data before receiving new data if accepted
            for replica_index in range(2):
                if accepted[replica_index]:
                    comm.Send(np.copy(S_current[replica_index]), dest=partner)

                    # Receive S_current data from partner
                    partner_S_current = np.empty((N, N, N), dtype=S_current.dtype)
                    comm.Recv(partner_S_current, source=partner)
                    S_current[replica_index] = np.copy(partner_S_current)

        else:  # Non-initiating ranks
            # Calculate and send energy information to the partner
            for replica_index in range(2):
                energy_buffer[replica_index] = total_energy(S_current[replica_index], J_arr)
            comm.Send(energy_buffer, dest=partner)

            # Receive acceptance info from the initiating rank
            comm.Recv(accepted, source=partner)

            # If accepted, handle S_current exchanges
            for replica_index in range(2):
                if accepted[replica_index]:
                    # Send the local S_current data to the initiating rank before making any changes
                    comm.Send(np.copy(S_current[replica_index]), dest=partner)

                    # Receive S_current from the initiating rank
                    partner_S_current = np.empty((N, N, N), dtype=S_current.dtype)
                    comm.Recv(partner_S_current, source=partner)
                    S_current[replica_index] = np.copy(partner_S_current)
    partner = (rank - 1 if (rank % 2) else rank + 1)
    comm.Barrier()  # Synchronize after MC step
    """
    if 0 <= partner < size:
        N = S_current.shape[-1]  # Assuming S_current is of shape (2, N, N, N)

        # Buffers for energy and acceptance
        energy_buffer = np.empty(2, dtype=np.float64)  # Only used by the non-initiating rank
        accepted = np.array([False, False], dtype=bool)  # Used by both ranks

        if rank % 2 == 1:  # Initiating ranks
            # Perform energy comparison and determine acceptance
            # Receive energy information from the partner
            comm.Recv(energy_buffer, source=partner)

            accepted = perform_exchange(S_current, betas, J_arr, comm, rank, partner, N, energy_buffer)

            # Send acceptance info
            comm.Send(accepted, dest=partner)

            # Send S_current data before receiving new data if accepted
            for replica_index in range(2):
                if accepted[replica_index]:
                    comm.Send(np.copy(S_current[replica_index]), dest=partner)

                    # Receive S_current data from partner
                    partner_S_current = np.empty((N, N, N), dtype=S_current.dtype)
                    comm.Recv(partner_S_current, source=partner)
                    S_current[replica_index] = np.copy(partner_S_current)

        else:  # Non-initiating ranks
            # Calculate and send energy information to the partner
            for replica_index in range(2):
                energy_buffer[replica_index] = total_energy(S_current[replica_index], J_arr)
            comm.Send(energy_buffer, dest=partner)

            # Receive acceptance info from the initiating rank
            comm.Recv(accepted, source=partner)

            # If accepted, handle S_current exchanges
            for replica_index in range(2):
                if accepted[replica_index]:
                    # Send the local S_current data to the initiating rank before making any changes
                    comm.Send(np.copy(S_current[replica_index]), dest=partner)

                    # Receive S_current from the initiating rank
                    partner_S_current = np.empty((N, N, N), dtype=S_current.dtype)
                    comm.Recv(partner_S_current, source=partner)
                    S_current[replica_index] = np.copy(partner_S_current)"""
def perform_exchange(S_current, betas, J_arr, comm, rank, partner, N, energy_recv_buffer):
    # Determine acceptance based on energy difference
    accepted = np.array([False, False], dtype=bool)
    for replica_index in range(2):
        energy_local = total_energy(S_current[replica_index], J_arr)
        energy_partner = energy_recv_buffer[replica_index]
        delta = (energy_local - energy_partner) * (betas[partner] - betas[rank])
        if delta <= 0 or np.random.rand() < np.exp(-delta):
            accepted[replica_index] = True
    return accepted

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
def get_q(S1,S2):
    N = len(S1)
    q = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                q += S1[i,j,k]*S2[i,j,k]
    return q / N**3
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

    # Collect IDs from the boundaries
    for i in range(N):
        for j in range(N):
            # Collecting from left boundary (x=0) and right boundary (x=N-1)
            left_id = cluster_matrix[i, 0, j]
            right_id = cluster_matrix[i, N-1, j]
            left_ids.add(left_id)
            right_ids.add(right_id)
            
            # Collecting from top boundary (y=0) and bottom boundary (y=N-1)
            top_id = cluster_matrix[0, i, j]
            bottom_id = cluster_matrix[N-1, i, j]
            top_ids.add(top_id)
            bottom_ids.add(bottom_id)
            
            # Collecting from front boundary (z=0) and back boundary (z=N-1)
            front_id = cluster_matrix[i, j, 0]
            back_id = cluster_matrix[i, j, N-1]
            front_ids.add(front_id)
            back_ids.add(back_id)
    
    # Check for spanning clusters in each direction
    vertical_spanning = left_ids.intersection(right_ids)
    horizontal_spanning = top_ids.intersection(bottom_ids)
    depth_spanning = front_ids.intersection(back_ids)
    
    # Return True if any spanning cluster is found
    if vertical_spanning or horizontal_spanning or depth_spanning:
        return True
    
    return False
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
        for j in range(N):
            left_id = cluster_matrix[i, 0, j]
            right_id = cluster_matrix[i, N-1, j]
            top_id = cluster_matrix[0, i, j]
            bottom_id = cluster_matrix[N-1, i, j]
            front_id = cluster_matrix[i, j, 0]
            back_id = cluster_matrix[i, j, N-1]
            
            left_ids.add(left_id)
            right_ids.add(right_id)
            top_ids.add(top_id)
            bottom_ids.add(bottom_id)
            front_ids.add(front_id)
            back_ids.add(back_id)
    
    # Track unique clusters that span in any direction
    spanning_clusters = set()
    
    # Check for clusters spanning in each direction
    vertical_spanning_clusters = left_ids.intersection(right_ids)
    horizontal_spanning_clusters = top_ids.intersection(bottom_ids)
    depth_spanning_clusters = front_ids.intersection(back_ids)
    
    spanning_clusters.update(vertical_spanning_clusters)
    spanning_clusters.update(horizontal_spanning_clusters)
    spanning_clusters.update(depth_spanning_clusters)
    
    # The number of unique spanning clusters
    return len(spanning_clusters)
@jit(nopython=True)
def get_G_x(S1, S2,r):
    ql = 0
    N = len(S1)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                ql += S1[i, j, k] * S1[(i + r) % N, j, k] * S2[i, j, k] * S2[(i + r) % N, j, k]
    return ql / (N * N * N)
@jit(nopython=True)
def get_FT_G(S1, S2,k):
    ft = 0
    N = len(S1)
    for r in range(N):
        ft += get_G(S1,S2,r)*np.exp(1j*k*r)
    return ft / (N * N * N)
@jit(nopython=True)
def get_G_r(S1, S2,rx,ry,rz):
    ql = 0
    N = len(S1)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                ql += S1[i, j, k] * S1[(i + rx) % N, (j + ry) % N, (k + rz) % N] * S2[i, j, k] * S2[(i + rx) % N, (j + ry) % N, (k + rz) % N]
    return ql 
# Collect Cases 
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
        for i in range(N):
            for j in range(N):
                # Horizontal bond check
                if i < N-1:
                    sxsy_h = S1[i, j] * S1[i + 1, j] + S2[i, j] * S2[i + 1, j]
                    if abs(sxsy_h) == 2 and sxsy_h * J_arr[i, j, 0] > 0 and np.random.random() < (1 - np.exp(-2 * beta * abs(J_arr[i, j, 0]))) ** 2:
                        connection[i, j, 0] = True

                # Vertical bond check
                if j < N-1:
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
def get_fkck_single_2D(S,J_arr, beta,PBC=True):
    N = len(S)
    connection = np.zeros((N, N, 2), dtype=np.bool_)
    if not PBC:
        for i in range(N):
            for j in range(N):
                # Horizontal bond check
                if i < N-1:
                    sxsy_h = S[i, j] * S[i + 1, j]
                    if sxsy_h * J_arr[i, j, 0] > 0 and np.random.random() < 1 - np.exp(-2 * beta * abs(J_arr[i, j, 0])):
                        connection[i, j, 0] = True
                if j < N-1:
                    # Vertical bond check
                    sxsy_v = S[i, j] * S[i, j + 1]
                    if sxsy_v * J_arr[i, j, 1] > 0 and np.random.random() < 1 - np.exp(-2 * beta * abs(J_arr[i, j, 1])):
                            connection[i, j, 1] = True
    else:
        for i in range(N):
            for j in range(N):
                # Horizontal bond check
                sxsy_h = S[i, j] * S[(i + 1)%N, j]
                if sxsy_h * J_arr[i, j, 0] > 0 and np.random.random() < 1 - np.exp(-2 * beta * abs(J_arr[i, j, 0])):
                    connection[i, j, 0] = True

                # Vertical bond check
                sxsy_v = S[i, j] * S[i, (j + 1)%N]
                if sxsy_v * J_arr[i, j, 1] > 0 and np.random.random() < 1 - np.exp(-2 * beta * abs(J_arr[i, j, 1])):
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
def find_largest_cluster_size_2D(cluster_matrix):
    """
    Given a matrix where each cell contains a cluster ID, return the size of the largest cluster.
    """
    N = cluster_matrix.shape[0]
    
    # Dictionary to count sizes of each cluster ID
    cluster_sizes = {}
    
    # Initialize the maximum size of clusters
    max_size = 0
    
    # Calculate sizes of each cluster and update max_size dynamically
    for i in range(N):
        for j in range(N):
            cluster_id = cluster_matrix[i, j]
            if cluster_id not in cluster_sizes:
                cluster_sizes[cluster_id] = 0
            cluster_sizes[cluster_id] += 1
            # Update the maximum size found so far
            if cluster_sizes[cluster_id] > max_size:
                max_size = cluster_sizes[cluster_id]
    
    return max_size
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
        left_id = cluster_matrix[i, 0]
        left_ids.add(left_id)
        right_id = cluster_matrix[i, N-1]
        right_ids.add(right_id)
        top_id = cluster_matrix[0, i]
        top_ids.add(top_id)
        bottom_id = cluster_matrix[N-1, i]
        bottom_ids.add(bottom_id)
    
    # Track unique clusters that span in any direction
    spanning_clusters = set()
    
    # Check for clusters spanning horizontally
    horizontal_spanning_clusters = left_ids.intersection(right_ids)
    spanning_clusters.update(horizontal_spanning_clusters)
    
    # Check for clusters spanning vertically
    vertical_spanning_clusters = top_ids.intersection(bottom_ids)
    spanning_clusters.update(vertical_spanning_clusters)
    
    # The number of unique spanning clusters
    return len(spanning_clusters)
def get_q_2D(S1, S2):
    # Get the number of rows and columns
    N = len(S1)
    
    # Initialize the correlation measure
    q = 0
    
    # Compute the sum of the element-wise product
    for i in range(N):
        for j in range(N):
            q += S1[i, j] * S2[i, j]
    
    # Normalize by the total number of elements
    return q / (N * N)
def get_ql_2D(S1,S2):
    ql = 0
    N = len(S1)
    for i in range(N):
        for j in range(N):
            ql += S1[i,j]*S1[(i+1)%N,j]*S2[i,j]*S2[(i+1)%N,j] + S1[i,j]*S1[i,(j+1)%N]*S2[i,j]*S2[i,(j+1)%N]
    return ql/(2*N*N) 
def get_q_arr(S1,S2):
    N = len(S1)
    q_arr = np.zeros((N,N),dtype=np.complex64)
    for i in range(N):
        for j in range(N):
            if S1[i,j]==S2[i,j]:
                q_arr[i,j] = 1
            else:
                q_arr[i,j] = -1
    return (q_arr + q_arr.T)/np.sqrt(N)
@jit(nopython=True)
def get_ql_arr(edges):
    N = len(edges)
    ql_arr = np.zeros((N,N),dtype=np.complex64)
    for i in range(N):
        for j in range(N):
            if edges[i,j,0]:
                ql_arr[i,j] += 1
            else:
                ql_arr[i,j] -= 1
            if edges[i,j,1]:
                ql_arr[i,j] += 1j
            else:
                ql_arr[i,j] -= 1j
    return (ql_arr + ql_arr.conj().T)/np.sqrt(N)
@jit(nopython=True)
def get_E_xy(S,J):
    N = len(S)
    E_xy = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                E_xy[i,j] += 1/2*S[i,j,k]*(S[(i+1)%N,j,k]*J[i,j,k,0]+S[(i-1)%N,j,k]*J[(i-1)%N,j,k,0]+
                                           S[i,(j+1)%N,k]*J[i,j,k,1]+S[i,(j-1)%N,k]*J[i,(j-1)%N,k,1]
                                           +S[i,j,(k+1)%N]*J[i,j,k,2]+S[i,j,(k-1)%N]*J[i,j,(k-1)%N,2])
    return E_xy
def MC_step(S_arr,J_arr,beta,timestep):
    sweep(S_arr[0],J_arr,beta)
    sweep(S_arr[1],J_arr,beta)
    cluster_flip_wolf(S_arr[0],J_arr,beta)
    cluster_flip_wolf(S_arr[1],J_arr,beta)
    
    #if timestep % 2 == 0:
    if get_q(S_arr[0],S_arr[1]) < 0:
        if np.random.random() < 0.5:
            S_arr[0] *= -1
        else:
            S_arr[1] *= -1
                
    if beta >= 0.3:
        cluster_flip_wolf_houdayer(S_arr[0],S_arr[1])

def main(beta_seed):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Scatter beta values to each rank
    N = 36
    num_sweeps = 1200000
    num_eig = 200
    betas = np.array([1.229755, 1.1578985, 1.1,1.086042, 1.0263455, 0.966649, 
                 0.9180075, 0.869366, 0.828463, 0.78756, 0.7521845, 0.716809, 0.685855, 
                 0.654901, 0.6261585, 0.597416, 0.5708845, 0.544353, 0.520032, 0.495711, 
                 0.472496, 0.449281, 0.427171, 0.405061, 0.384057, 0.363053, 0.343154, 
                 0.323255, 0.3033565, 0.283458, 0.2646645, 0.245871, 0.2270775, 0.208284, 
                 0.1905965, 0.172909, 0.155221, 0.137533,0.1])
    rng = np.random.default_rng(seed=beta_seed)
    J_arr = rng.standard_normal((N, N, N, 3))
    local_beta = np.empty(1)
    comm.Scatter(betas, local_beta, root=0)
    local_beta = local_beta[0]  # Convert from array to scalar
    S_current = np.random.choice([-1, 1], size=(2, N, N, N))
    beta = betas[rank]
    local_sum_e = 0
    local_sum_ql = 0
    local_sum_houd = 0
    local_sum_cmrj = 0
    local_sum_fkck = 0
    local_sum_sing = 0
    local_sum_q = 0
    local_arr_eigc = np.zeros(num_eig*N)
    local_arr_eighoud = np.zeros(num_eig*N)
    local_arr_eigfkck = np.zeros(num_eig*N)
    local_arr_eigsing = np.zeros(num_eig*N)
    local_arr_eigcmrj = np.zeros(num_eig*N)
    local_arr_eigxy = np.zeros(num_eig*N)
    local_arr_eigE = np.zeros(num_eig*N)
    local_arr_eigfourier = np.zeros(num_eig*N)
    num_ave = num_sweeps//50
    recv_data = np.zeros((size,7), dtype=float)  # Adjusted to num_betas
    recv_eig = None
    if rank == 0:
      # Create a single array of shape (8, size, num_eig * N) for all eigenvalue sets
        recv_eig = np.zeros((size, 8,num_eig * N), dtype=float)
    tau = 1000 
    Sarrdata1 = np.zeros((num_ave,N,N,N))
    Sarrdata2 = np.zeros((num_ave,N,N,N))
    # Main loop
    for j in range(num_sweeps):
        # Perform local sweeps
        comm.Barrier()  # Synchronize before starting

        # Perform MC step
        MC_step(S_current, J_arr, beta,j)

        # Exchange configurations with the partner rank
        exchange_mc(S_current,betas, J_arr, comm, rank, size,j)
        if j >= num_sweeps - num_ave:
            Sarrdata1[j-num_sweeps+num_ave] = np.copy(S_current[0])
            Sarrdata2[j-num_sweeps+num_ave] = np.copy(S_current[1])
            if j >= num_sweeps - num_ave + tau:
                local_sum_e += (total_energy(Sarrdata1[j-num_sweeps+num_ave],J_arr)+total_energy(Sarrdata2[j-num_sweeps+num_ave],J_arr))/6/N**3/(num_ave-tau)
                local_sum_ql += beta*(get_ql(Sarrdata1[j-num_sweeps+num_ave],Sarrdata2[(j-num_sweeps+num_ave+1)%tau])-1)/(num_ave-tau)
                local_sum_cmrj += (contains_spanning_cluster_2d(find_clusters_2D(get_jorg_2D(Sarrdata1[(j-num_sweeps+num_ave)%tau,0],Sarrdata2[(j-num_sweeps+num_ave+1)%tau,0],J_arr[0,:,:,1:],beta,False))))/ (num_ave-tau)
                local_sum_houd += (contains_spanning_cluster_2d(find_clusters_2D(get_houdayer_2D(Sarrdata1[(j-num_sweeps+num_ave)%tau,0],Sarrdata2[(j-num_sweeps+num_ave+1)%tau,0],False))))/ (num_ave-tau)
                local_sum_fkck += (contains_spanning_cluster_2d(find_clusters_2D(get_fkck_2D(Sarrdata1[(j-num_sweeps+num_ave)%tau,0],Sarrdata2[(j-num_sweeps+num_ave+1)%tau,0],J_arr[0,:,:,1:],beta,False))))/ (num_ave-tau)
                local_sum_sing += (contains_spanning_cluster_2d(find_clusters_2D(get_fkck_single_2D(Sarrdata1[(j-num_sweeps+num_ave)%tau,0],J_arr[0,:,:,1:],beta,False)))+contains_spanning_cluster_2d(find_clusters_2D(get_fkck_single_2D(Sarrdata2[(j-num_sweeps+num_ave)%tau,0],J_arr[0,:,:,1:],beta,False))))/2/ (num_ave-tau)
                local_sum_q +=  abs(get_q_2D(Sarrdata1[(j-num_sweeps+num_ave)%tau,0],Sarrdata2[(j-num_sweeps+num_ave+1)%tau,0])/(num_ave-tau))
                if j > num_sweeps - num_eig:
                    local_arr_eigc[(j-num_sweeps+num_eig)*N:(j-num_sweeps+num_eig+1)*N] = LA.eig(get_q_arr(Sarrdata1[(j-num_sweeps+num_ave)%tau,0],Sarrdata2[(j-num_sweeps+num_ave+1)%tau,0]))[0].real
                    local_arr_eighoud[(j-num_sweeps+num_eig)*N:(j-num_sweeps+num_eig+1)*N] = LA.eig(get_ql_arr(get_houdayer_2D(Sarrdata1[(j-num_sweeps+num_ave)%tau,0],Sarrdata2[(j-num_sweeps+num_ave+1)%tau,0])))[0].real
                    local_arr_eigfkck[(j-num_sweeps+num_eig)*N:(j-num_sweeps+num_eig+1)*N] = LA.eig(get_ql_arr(get_fkck_2D(Sarrdata1[(j-num_sweeps+num_ave)%tau,0],Sarrdata2[(j-num_sweeps+num_ave+1)%tau,0],J_arr[0,:,:,1:],beta)))[0].real
                    local_arr_eigsing[(j-num_sweeps+num_eig)*N:(j-num_sweeps+num_eig+1)*N] = LA.eig(get_ql_arr(get_fkck_single_2D(S_current[0,0],J_arr[0,:,:,1:],beta)))[0].real
                    local_arr_eigcmrj[(j-num_sweeps+num_eig)*N:(j-num_sweeps+num_eig+1)*N] = LA.eig(get_ql_arr(get_cmrj_2D(Sarrdata1[(j-num_sweeps+num_ave)%tau,0],Sarrdata2[(j-num_sweeps+num_ave+1)%tau,0],J_arr[0,:,:,1:],beta)))[0].real
                    local_arr_eigxy[(j-num_sweeps+num_eig)*N:(j-num_sweeps+num_eig+1)*N] = LA.eig(get_G_arr(Sarrdata1[(j-num_sweeps+num_ave)%tau],Sarrdata2[(j-num_sweeps+num_ave+1)%tau]))[0].real
                    local_arr_eigE[(j-num_sweeps+num_eig)*N:(j-num_sweeps+num_eig+1)*N] = LA.eig(get_E_xy(Sarrdata1[(j-num_sweeps+num_ave)%tau],Sarrdata2[(j-num_sweeps+num_ave+1)%tau],J_arr))[0].real
                    local_arr_eigfourier[(j-num_sweeps+num_eig)*N:(j-num_sweeps+num_eig+1)*N] = LA.eig(get_q_fft(Sarrdata1[(j-num_sweeps+num_ave)%tau],Sarrdata2[(j-num_sweeps+num_ave+1)%tau]))[0].real
    # Gather all local sums from each rank
    all_sums = np.array([local_sum_e, local_sum_ql,local_sum_cmrj,local_sum_houd,local_sum_fkck,local_sum_sing,local_sum_q], dtype=float)
    comm.Gather(all_sums, recv_data, root=0)
    all_sums_eig = np.array([local_arr_eigc, local_arr_eighoud,local_arr_eigfkck,local_arr_eigsing,local_arr_eigcmrj,local_arr_eigxy,local_arr_eigE,local_arr_eigfourier], dtype=float)
    comm.Gather(all_sums_eig, recv_eig, root=0)
    if rank == 0:
        # Save the gathered data to a file
        folder_name = 'data_36_eig'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_path_data = f'data_{beta_seed}.txt'
        np.savetxt(file_path_data, np.column_stack((betas, recv_data)), fmt='%f', delimiter=',', header='Beta,QL,CMRJ,HOUD,FKCK,SING,Q')
        file_path_eig = os.path.join(folder_name, f'eig{beta_seed}.npy')
        np.save(file_path_eig, recv_eig)

if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)
