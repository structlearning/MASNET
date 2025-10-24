import torch
from torch.utils.data import Dataset
import torch
from scipy.optimize import linear_sum_assignment

class SimpleDataset(Dataset):
    def __init__(self):
        """
        Simple dataset of size 3 with predefined samples.
        Each sample consists of (x1, x2, label).
        x1 and x2 are shaped as [2,1].
        """
        self.data = [
            (torch.tensor([[-0.75], [0.75]]), torch.tensor([[-0.5], [0.5]]), torch.tensor(1)),
            (torch.tensor([[-0.5], [0.5]]), torch.tensor([[-0.75], [0.75]]), torch.tensor(1)),
            # (torch.tensor([[-1.0, 1.0]]), torch.tensor([[-10.0, 10.0]]), torch.tensor(0)),
            (torch.tensor([[0.5]]), torch.tensor([[0.5], [0.75]]), torch.tensor(0)),
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SubsetDataset(Dataset):
    def __init__(self, m, n, d, num_samples):
        """
        Custom dataset where:
        - If index is even: S1 ⊆ S2, label = 1
        - If index is odd: S1 ⊈ S2, label = 0

        Args:
            m (int): Number of points in S1.
            n (int): Number of points in S2.
            d (int): Dimensionality of each point.
            num_samples (int): Total number of dataset samples.
        """
        self.m = m
        self.n = n
        self.d = d
        self.num_samples = num_samples

        # Pre-generate dataset
        self.data = []
        for i in range(num_samples):
            S2 = torch.rand(n, d)  # Generate S2 randomly

            if i % 2 == 0 :
                # Even index: S1 is a subset of S2
                subset_indices = torch.randperm(n)[:m]  # Pick m random indices from S2
                S1 = S2[subset_indices]  # Subset of S2
                label = 0
            elif i % 2 == 1:
                # Odd index: S1 is NOT a subset of S2
                S1 = torch.rand(m, d)  # Generate S1 randomly
                while all(any(torch.all(s1 == s2, dim=-1) for s2 in S2) for s1 in S1):
                    S1 = torch.rand(m, d)  # Regenerate until at least one point is unique
                label = 1
            mask = torch.tensor([1]).view(-1,1)
            self.data.append((S1, S2, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]  # Returns (S1, S2, label)

class FacilityLocation(Dataset):
    def __init__(self, n, d, num_samples):
        """
        Dataset of matrices where each person chooses one option (column),
        no two people choose the same option, and the label is the
        maximum total sum under this injective constraint.

        Args:
            n (int): Number of agents (rows).
            d (int): Number of options (columns), must satisfy d ≥ n.
            num_samples (int): Number of dataset samples to generate.
        """
        assert d >= n, "Number of columns (d) must be at least number of rows (n)"
        self.n = n
        self.d = d
        self.num_samples = num_samples
        self.data = []

        for _ in range(num_samples):
            # Generate matrix with positive values in (0, 1)
            mat = torch.rand(n, d)

            # Convert to NumPy and negate for maximization
            cost_matrix = -mat.numpy()
            _, col_ind = linear_sum_assignment(cost_matrix)

            # Compute max-sum using chosen assignment
            max_sum = mat[range(n), torch.tensor(col_ind)].mean().item()

            label = torch.tensor(max_sum, dtype=torch.float32)
            mask = torch.tensor([1]).view(-1, 1)  # dummy mask
            self.data.append((mat,  mat, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]  # (matrix, label)
