import numpy as np
import matplotlib.pyplot as plt
from numba import jit

J0 = 0
J = 1
N = 32

#Equilibriation Algorithms
def sweep(S_current, J_arr, beta):
    N = len(S_current)
    for _ in range(N*N):
        i,j = np.random.randint(0,N),np.random.randint(0,N)
        # Considering periodic boundary conditions (PBC)
        delta_E = 2 * S_current[i, j] * (
                    J_arr[i, j, 0] * S_current[(i + 1)%N, j] +  # Interaction with the right neighbor
                    J_arr[(i - 1)%N, j, 0] * S_current[(i - 1)%N, j] +  # Interaction with the left neighbor
                    J_arr[i, j, 1] * S_current[i, (j + 1)%N] +  # Interaction with the bottom neighbor
                    J_arr[i, (j - 1)%N, 1] * S_current[i, (j - 1)%N]  # Interaction with the top neighbor
            ) 
                # Metropolis criterion
        if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
            S_current[i, j] *= -1

    return S_current
    
@jit(nopython=True)
def cluster_flip_wolf(S, J_arr, beta):
    N = len(S)
    cluster_mark = np.zeros((N, N), dtype=np.bool_)  # Array to mark sites in the cluster
    
    # Generate random numbers for horizontal and vertical checks
    random_numbers = np.random.random((N, N, 2))

    # Randomly select the starting point
    i, j = np.random.randint(0, N, size=2)
    stack = [(i, j)]
    cluster_mark[i, j] = True

    while stack:
        i_, j_ = stack.pop()

        # Right neighbor with periodic boundary conditions
        right_i, right_j = (i_ + 1) % N, j_
        if not cluster_mark[right_i, right_j] and J_arr[i_, j_, 0] * S[right_i, right_j] * S[i_, j_] > 0 and random_numbers[i_, j_, 0] < 1 - np.exp(-2 * abs(J_arr[i_, j_, 0]) * beta):
            cluster_mark[right_i, right_j] = True
            stack.append((right_i, right_j))

        # Left neighbor with periodic boundary conditions
        left_i, left_j = (i_ - 1) % N, j_
        if not cluster_mark[left_i, left_j] and J_arr[left_i, left_j, 0] * S[left_i, left_j] * S[i_, j_] > 0 and random_numbers[left_i, left_j, 0] < 1 - np.exp(-2 * abs(J_arr[left_i, left_j, 0]) * beta):
            cluster_mark[left_i, left_j] = True
            stack.append((left_i, left_j))

        # Bottom neighbor with periodic boundary conditions
        bottom_i, bottom_j = i_, (j_ + 1) % N
        if not cluster_mark[bottom_i, bottom_j] and J_arr[i_, j_, 1] * S[bottom_i, bottom_j] * S[i_, j_] > 0 and random_numbers[i_, j_, 1] < 1 - np.exp(-2 * abs(J_arr[i_, j_, 1]) * beta):
            cluster_mark[bottom_i, bottom_j] = True
            stack.append((bottom_i, bottom_j))

        # Top neighbor with periodic boundary conditions
        top_i, top_j = i_, (j_ - 1) % N
        if not cluster_mark[top_i, top_j] and  J_arr[top_i, top_j, 1] * S[top_i, top_j] * S[i_, j_] > 0 and random_numbers[top_i, top_j, 1] < 1 - np.exp(-2 * abs(J_arr[top_i, top_j, 1]) * beta):
            cluster_mark[top_i, top_j] = True
            stack.append((top_i, top_j))

    # Flip all spins in the cluster
    for i in range(N):
        for j in range(N):
            if cluster_mark[i,j]:
                S[i,j] *= -1

    return S
    
@jit(nopython=True)
def cluster_flip_sw(S,J_arr,beta):
    clusters = find_clusters(get_fkck_single(S,J_arr,beta))
    cluster_max = np.max(clusters)
    for cluster_id in range(1, cluster_max+1):
        if np.random.random() < 0.5:
            for i in range(N):
                for j in range(N):
                    if clusters[i, j]==cluster_id:
                        S1[i, j] *= -1
                        S2[i, j] *= -1  
    return S1,S2
    
@jit(nopython=True)
def cluster_flip_houdayer_sw(S1,S2):
    N = len(S1)
    clusters = find_clusters(get_houdayer(S1,S2))
    cluster_max = np.max(clusters)
    for cluster_id in range(1, cluster_max+1):
        if np.random.random() < 0.5:
            for i in range(N):
                for j in range(N):
                    if clusters[i, j]==cluster_id:
                        S1[i, j] *= -1
                        S2[i, j] *= -1  
    return S1,S2
    
@jit(nopython=True)
def cluster_flip_jorg_sw(S1,S2,J_arr,beta):
    N = len(S1)
    clusters = find_clusters(get_jorg(S1,S2,J_arr,beta))
    cluster_max = np.max(clusters)
    for cluster_id in range(1, cluster_max+1):
        if np.random.random() < 0.5:
            for i in range(N):
                for j in range(N):
                    if clusters[i, j]==cluster_id:
                        S1[i, j] *= -1
                        S2[i, j] *= -1   
    return S1, S2
    
@jit(nopython=True)
def cluster_flip_cmrj_sw(S1,S2,J_arr,beta):
    edges = get_cmrj(S1,S2,J_arr,beta)
    clusters_gray = find_clusters(edges.astype(np.bool_))
    
    clusters_gray_max = np.max(clusters_gray)
    clusters_blue = find_clusters((edges//2).astype(np.bool_))
    clusters_blue_max = np.max(clusters_blue)
    # Shuffle cluster IDs
    cluster_ids_gray = np.random.permutation(np.arange(clusters_gray_max + 1))
    random_arr_blue = np.random.random(clusters_blue_max+1) < 0.5
    # Select half of the clusters to flip (including zero)
    clusters_to_flip_gray = cluster_ids_gray[:clusters_gray_max // 2+1]
    # Flip the spins in the selected clusters
    for cluster_id in clusters_to_flip_gray:
        for i in range(N):
            for j in range(N):
                if clusters_gray[i, j] == cluster_id:
                    if random_arr_blue[clusters_blue[i,j]-1]:
                        S1[i, j] *= -1
                    else:
                        S2[i, j] *= -1

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

# Find Satisfied Edges Clusters
@jit(nopython=True)
def get_fkck_single(S,J_arr, beta,PBC=True):
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
def get_fkck(S1, S2, J_arr, beta,PBC=True):
    N = len(S1)
    connection = np.zeros((N, N, 2), dtype=np.bool_)
    if not PBC:
        for i in range(N):
            for j in range(N):
                if i < N-1:
                    # Horizontal bond check
                    sxsy_h = S1[i, j] * S1[i + 1, j] + S2[i, j] * S2[i + 1, j]
                    if abs(sxsy_h) == 2 and sxsy_h * J_arr[i, j, 0] > 0 and np.random.random() < (1 - np.exp(-2 * beta * abs(J_arr[i, j, 0]))) ** 2:
                        connection[i, j, 0] = True
                if j < N-1:
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
def get_cmrj(S1, S2, J_arr, beta, PBC=True):
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
def get_houdayer(S1, S2,PBC=True):
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
def get_jorg(S1, S2, J_arr, beta,PBC=True):
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

# Cluster Calculation Methods
@jit(nopython=True)
def dfs(i, j, edges, visited, cluster_id, cluster_array, N):
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
def find_clusters(edges):
    N = edges.shape[0]
    visited = np.zeros((N, N), dtype=np.bool_)
    cluster_array = np.zeros((N, N), dtype=np.int16)  # Array to hold cluster IDs
    cluster_id = 1  # Start cluster IDs from 1

    for i in range(N):
        for j in range(N):
            if not visited[i, j]:
                dfs(i, j, edges, visited, cluster_id, cluster_array, N)
                cluster_id += 1  # Increment cluster ID

    return cluster_array
    
@jit(nopython=True)
def find_cluster_sizes(cluster_matrix):
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
def contains_spanning_cluster(cluster_matrix):
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

# Calculate Observables
@jit(nopython=True)
def total_energy(S,J_arr):
    energy = 0
    N = len(S)
    for i in range(N):
        for j in range(N):
            energy -= (S[i,j]*S[(i+1)%N,j]*J_arr[i,j,0] + S[i,j]*S[i,(j+1)%N]*J_arr[i,j,1])
    return energy
    
@jit(nopython=True)
def get_ql(S1,S2):
    ql = 0
    N = len(S1)
    for i in range(N):
        for j in range(N):
            ql += S1[i,j]*S1[(i+1)%N,j]*S2[i,j]*S2[(i+1)%N,j] + S1[i,j]*S1[i,(j+1)%N]*S2[i,j]*S2[i,(j+1)%N]
    return ql/(2*N*N) 

# Obtain Appropriate Betas for Parallel Tempering
@jit(nopython=True)
def get_new_beta(beta_arr,e_arr):
    beta_arr = beta_arr
    e_arr = e_arr
    beta_new = [beta_arr[0]]
    ind = 0
    while ind < len(beta_arr)-1:
        for i in range(ind+1,len(beta_arr)):
            check = np.exp((beta_arr[i]-beta_arr[ind])*(e_arr[i]-e_arr[ind]))
            if 0.4< check and check < 0.5:
                ind = i
                beta_new.append(beta_arr[ind])
                continue
        break
    return beta_new
    
@jit(nopython=True)
def get_new_beta_opt(beta_arr):
    N = len(beta_arr)
    T_arr = [1/i for i in beta_arr]
    for i in range(1,N):
        T_arr[i] = T_arr[0]*np.power(T_arr[-1]/T_arr[0],i/(N-1))
    return np.array([1/i for i in T_arr],dtype=np.float32)

# Run Simulations
def get_sizes_and_R_fkck(numSweeps,num_ave,beta_arr,N):
    Sarr1 = np.zeros((len(beta_arr), N, N), dtype=np.int8)
    Sarr2 = np.zeros((len(beta_arr), N, N), dtype=np.int8)
    
    for i in range(len(beta_arr)):
        for j in range(N):
            for k in range(N):
                Sarr1[i, j, k] = -1 + 2 * np.random.randint(2)
                Sarr2[i, j, k] = -1 + 2 * np.random.randint(2)
    J_arr = np.random.normal(J0,J,(N,N,2))
    R_list = np.zeros((len(beta_arr)),dtype=np.float32)
    sizes_list = np.zeros((len(beta_arr),3),dtype=np.float32)
    e_arr1 = np.zeros((len(beta_arr)),dtype=np.float32)
    e_arr2 = np.zeros((len(beta_arr)),dtype=np.float32)
    
    for j in range(numSweeps):
        for i in range(len(beta_arr)):
            #Sarr1[i],Sarr2[i] = cluster_flip_fkck_wolf(Sarr1[i],Sarr2[i],J_arr,beta_arr[i])
            if j % 2 ==1:
            #Sarr1[i],Sarr2[i] = cluster_flip_houdayer_sw(Sarr1[i],Sarr2[i],J_arr,beta_arr[i])
                Sarr1[i],Sarr2[i] = cluster_flip_jorg_sw(Sarr1[i],Sarr2[i],J_arr,beta_arr[i])
            else: 
                Sarr1[i],Sarr2[i] = cluster_flip_houdayer_sw(Sarr1[i],Sarr2[i])
            #
            Sarr1[i] = sweep(Sarr1[i],J_arr,beta_arr[i])
            Sarr2[i] = sweep(Sarr2[i],J_arr,beta_arr[i])
            Sarr1[i],Sarr2[i] = cluster_flip_wolf(Sarr1[i],J_arr,beta_arr[i]),cluster_flip_wolf(Sarr2[i],J_arr,beta_arr[i])
            
            if j > numSweeps - num_ave:
                # Calculate Spanning Probability
                clusters = find_clusters(get_fkck(Sarr1[i],Sarr2[i],J_arr,beta_arr[i],False))
                R_list[i] += contains_spanning_cluster(clusters)/num_ave 
                sizes = find_cluster_sizes(clusters)            
                for k in range(3):
                    sizes_list[i,k] += sizes[k]/num_ave/N**2
                e_arr1[i] += total_energy(Sarr1[i],J_arr)/num_ave/N**2
                e_arr2[i] += get_ql(Sarr1[i],Sarr2[i])/num_ave
            
        Sarr1=exchange_mc(Sarr1,beta_arr,J_arr)
        Sarr2=exchange_mc(Sarr2,beta_arr,J_arr)
    return R_list,sizes_list,e_arr1,e_arr2
def get_sizes_and_R_single(numSweeps,num_ave,beta_arr,N):
    Sarr1 = np.zeros((len(beta_arr), N, N), dtype=np.int8)
    Sarr2 = np.zeros((len(beta_arr), N, N), dtype=np.int8)
    
    for i in range(len(beta_arr)):
        for j in range(N):
            for k in range(N):
                Sarr1[i, j, k] = -1 + 2 * np.random.randint(2)
                Sarr2[i, j, k] = -1 + 2 * np.random.randint(2)
    J_arr = np.random.normal(J0,J,(N,N,2))
    R_list1 = np.zeros((len(beta_arr)),dtype=np.float32)
    size1 = np.zeros((len(beta_arr)),dtype=np.float32)
    R_list2 = np.zeros((len(beta_arr)),dtype=np.float32)
    size2 = np.zeros((len(beta_arr)),dtype=np.float32)
    for j in range(numSweeps):
        for i in range(len(beta_arr)):
            Sarr1[i] = sweep(Sarr1[i],J_arr,beta_arr[i])
            Sarr2[i] = sweep(Sarr2[i],J_arr,beta_arr[i])
            Sarr1[i] = cluster_flip_wolf(Sarr1[i],J_arr,beta_arr[i])
            Sarr2[i] = cluster_flip_wolf(Sarr2[i],J_arr,beta_arr[i])
            if j % 2 == 1:
                Sarr1[i],Sarr2[i] = cluster_flip_jorg_sw(Sarr1[i],Sarr2[i],J_arr,beta_arr[i])
            else: 
                Sarr1[i],Sarr2[i] = cluster_flip_houdayer_sw(Sarr1[i],Sarr2[i])

            
            if j > numSweeps - num_ave:
                # Calculate Spanning Probability
                R_list1[i] += contains_spanning_cluster(find_clusters(get_fkck_single(Sarr1[i],J_arr,beta_arr[i],False)))/num_ave
                size1[i] += find_cluster_sizes(find_clusters(get_fkck_single(Sarr1[i],J_arr,beta_arr[i],True)))[0]/num_ave/N**2
                R_list2[i] += contains_spanning_cluster(find_clusters(get_fkck_single(Sarr2[i],J_arr,beta_arr[i],False)))/num_ave
                size2[i] += find_cluster_sizes(find_clusters(get_fkck_single(Sarr2[i],J_arr,beta_arr[i],True)))[0]/num_ave/N**2

        Sarr1=exchange_mc(Sarr1,beta_arr,J_arr)
        Sarr2=exchange_mc(Sarr2,beta_arr,J_arr)
                
    
    return R_list1,R_list2,size1,size2
def get_sizes_and_R_houdayer(numSweeps,num_ave,beta_arr,N):
    Sarr1 = np.zeros((len(beta_arr), N, N), dtype=np.int8)
    Sarr2 = np.zeros((len(beta_arr), N, N), dtype=np.int8)
    
    for i in range(len(beta_arr)):
        for j in range(N):
            for k in range(N):
                Sarr1[i, j, k] = -1 + 2 * np.random.randint(2)
                Sarr2[i, j, k] = -1 + 2 * np.random.randint(2)
    J_arr = np.random.normal(J0,J,(N,N,2))
    R_list = np.zeros((len(beta_arr)),dtype=np.float32)
    sizes_list = np.zeros((len(beta_arr),3),dtype=np.float32)
    e_arr1 = np.zeros((len(beta_arr)),dtype=np.float32)
    e_arr2 = np.zeros((len(beta_arr)),dtype=np.float32)
    
    for j in range(numSweeps):
        for i in range(len(beta_arr)):
            Sarr1[i] = sweep(Sarr1[i],J_arr,beta_arr[i])
            Sarr2[i] = sweep(Sarr2[i],J_arr,beta_arr[i])
            Sarr1[i] = cluster_flip_wolf(Sarr1[i],J_arr,beta_arr[i])
            Sarr2[i] = cluster_flip_wolf(Sarr2[i],J_arr,beta_arr[i])
            if j % 2 == 1:
                Sarr1[i],Sarr2[i] = cluster_flip_houdayer_sw(Sarr1[i],Sarr2[i])
            else: 
                Sarr1[i],Sarr2[i] = cluster_flip_jorg_sw(Sarr1[i],Sarr2[i],J_arr,beta_arr[i])
            if j > numSweeps - num_ave:
                # Calculate Spanning Probability
                clusters = find_clusters(get_houdayer(Sarr1[i],Sarr2[i],False))
                R_list[i] += contains_spanning_cluster(clusters)/num_ave
                sizes = find_cluster_sizes(find_clusters(get_houdayer(Sarr1[i],Sarr2[i],True)))
                for k in range(3):
                    sizes_list[i,k] += sizes[k]/num_ave/N**2
                e_arr1[i] += total_energy(Sarr1[i],J_arr)/num_ave/N**2
                e_arr2[i] += get_ql(Sarr1[i],Sarr2[i])/num_ave
        Sarr1=exchange_mc(Sarr1,beta_arr,J_arr)
        Sarr2=exchange_mc(Sarr2,beta_arr,J_arr)
                
    
    return R_list,sizes_list,e_arr1,e_arr2#R_list,sizes_list,e_arr1,e_arr2
