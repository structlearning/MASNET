import torch
from itertools import combinations

# Function to generate subsets as index lists
def subsets_at_most_k(iterable, k):
    """Generate all subsets of size at most k from the given iterable."""
    s = list(iterable)
    return [list(comb) for r in range(1, k + 1) for comb in combinations(s, r)]

# Function to compute f(S) using max operation
def vector_sum_with_noise(subset, f_values):
    """
    Compute f(S) as max of vectors (indexed from tensor f_values).
    
    Parameters:
        subset: List of indices forming the subset
        f_values: Tensor of shape (n, dim), stored on a specific device.
    
    Returns:
        Tensor of shape (dim,), on the same device as f_values.
    """
    return torch.max(f_values[subset], dim=0).values  # Efficient tensor indexing

# Function to check subset property across all subset pairs
def check_vector_subset_ordering_k(f_values, universe, k, device):
    """
    Checks if S1 ⊆ S2 ⇔ f(S1) ≤ f(S2) (component-wise) for all subsets of size at most k.
    
    Parameters:
        f_values: Tensor of shape (n, dim), stored on a specific device.
        universe: Set of available indices.
        k: Maximum subset size to check.
        device: Torch device (CPU/GPU).
    
    Returns:
        Boolean: True if the property holds, False otherwise.
    """
    subsets = subsets_at_most_k(universe, k)

    # Compute f(S) for all subsets, ensuring tensors stay on `device`
    f_computed = {tuple(s): vector_sum_with_noise(s, f_values) for s in subsets}

    for S1 in subsets:
        for S2 in subsets:
            if not set(S1).issubset(S2):  # Only check non-subset cases
                f_S1 = f_computed[tuple(S1)]
                f_S2 = f_computed[tuple(S2)]

                # If f(S1) ≤ f(S2), but S1 is not a subset of S2, return False
                if torch.all(f_S1 <= f_S2):
                    return False

    return True  # Passed all checks

# Function to generate a tensor of random positive vectors
def generate_random_f(dim, n, device):
    """Generate an f where f(i) is a random positive vector in R^dim, stored on the specified device."""
    return torch.rand(n, dim, device=device)  # Shape: (n, dim)

# Function to find minimal dimension d that satisfies subset ordering property
def find_minimal_d_for_n(n, k, max_d=100, max_attempts=5*10**4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Finds the minimal dimensionality d such that a function f satisfying the subset ordering property exists.
    
    Parameters:
        n: Number of elements.
        k: Maximum subset size to check.
        max_d: Maximum dimension to search up to.
        max_attempts: Maximum number of function generations.
        device: Torch device (CPU/GPU).
    
    Returns:
        The minimal d that satisfies the property, or None if no valid d is found.
    """
    universe = set(range(n))  # Ensure indices start from 0

    for d in range(2, max_d + 1):  # Iterate over dimensions from 2 to max_d
        for _ in range(max_attempts):
            f = generate_random_f(dim=d, n=n, device=device)  # Generate random f

            if check_vector_subset_ordering_k(f, universe, k, device=device):
                return d  # Return the minimal dimension where a valid f is found

    return None  # If no valid f found within max_d, return None

# Test for multiple values of n
n_values = [10,20,30,40,50]  # Example range of n values to check
k = 2  # Maximum subset size to check

# Compute minimal d for each n
minimal_d_results = {n: find_minimal_d_for_n(n, k, max_attempts=5*10**4) for n in n_values}
print(minimal_d_results)



