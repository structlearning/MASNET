import os
import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import glob
import trimesh
import requests
import zipfile
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import pickle
import sys
from argparse import ArgumentParser

class ShapeNetNoisySubsetDataset(torch.utils.data.Dataset):
    """
    Dataset for creating noisy subset containment tasks with ShapeNet 3D shapes.
    
    - For positive examples (label=1): 
      Take a cloud C1, generate a subset S1 from it, add noise to S1, and label (S1, C1) as 1.
      
    - For negative examples (label=0): 
      Take two clouds C1 and C2, sample S1 from C1, add noise to S1, 
      but pair it with the different cloud C2, labeling (S1, C2) as 0.
    """
    def __init__(self, root_dir, categories=None, num_samples=1000, frac_subset=0.3, 
                 num_points=1024, subset_ratio=0.5, noise_level=0.02, transform=None):
        """
        Args:
            root_dir (str): Path to the ShapeNet dataset.
            categories (list): List of categories to use. If None, use all categories.
            num_samples (int): Number of sample pairs to generate.
            frac_subset (float): Fraction of samples that should be true subsets (label=1).
            num_points (int): Number of points to sample from complete shapes.
            subset_ratio (float): Ratio of points to include in the subset (S1).
            noise_level (float): Standard deviation of Gaussian noise to add to subsets.
            transform (callable, optional): Optional transform to be applied to samples.
        """
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.frac_subset = frac_subset
        self.num_points = num_points
        self.subset_ratio = subset_ratio
        self.subset_size = int(num_points * subset_ratio)
        self.noise_level = noise_level
        self.transform = transform
        
        with open(os.path.join(root_dir, 'taxonomy.json'), 'r') as f:
            self.taxonomy = json.load(f)
            
        if categories is not None:
            self.taxonomy = [item for item in self.taxonomy if item['synsetId'] in categories]
        
        # Map from synset ID to name
        self.id_to_name = {item['synsetId']: item['name'] for item in self.taxonomy}
        
        # Collect model paths
        self.models = []
        for item in self.taxonomy:
            synset_id = item['synsetId']
            category_path = os.path.join(root_dir, synset_id)
            if os.path.exists(category_path):
                model_ids = os.listdir(category_path)
                for model_id in model_ids:
                    model_path = os.path.join(category_path, model_id, 'models', 'model_normalized.obj')
                    if os.path.exists(model_path):
                        self.models.append((synset_id, model_id, model_path))
        
        print(f"Found {len(self.models)} models across {len(self.taxonomy)} categories")
        
        # Pre-generate dataset pairs
        self.data = self._generate_data()
            
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            S1 (tensor): Points from first shape (subset with noise)
            S1_mask (tensor): Mask indicating valid points in S1
            C (tensor): Points from second shape (complete point cloud)
            C_mask (tensor): Mask indicating valid points in C
            label (int): 1 if S1 is derived from C, 0 otherwise
        """
        return self.data[idx]
    
    def _generate_data(self):
        """Generate dataset of noisy shape subset containment examples."""
        data = []
        inv_frac = math.ceil(1 / self.frac_subset)
        
        for i in tqdm(range(self.num_samples)):
            idx1 = random.randint(0, len(self.models) - 1)
            _, _, path1 = self.models[idx1]
            C1_points = self._sample_points_from_mesh(path1, self.num_points)
            
            indices = torch.randperm(self.num_points)[:self.subset_size]
            S1_points = C1_points[indices].clone()
            
            # Add Gaussian noise to S1
            noise = torch.randn_like(S1_points) * self.noise_level
            S1_points = S1_points + noise
            
            if i % inv_frac == 0:
                # Positive example: S1 (noisy subset) paired with its source C1
                C_points = C1_points
                label = 1  # S1 is derived from C1
            else:
                # Negative example: S1 paired with a different point cloud C2
                idx2 = random.randint(0, len(self.models) - 1)
                while idx2 == idx1:  # Ensure we get a different model
                    idx2 = random.randint(0, len(self.models) - 1)
                
                _, _, path2 = self.models[idx2]
                C_points = self._sample_points_from_mesh(path2, self.num_points)  # This is C2
                label = 0  # S1 is not derived from C2
            
            # Create masks (all 1's since we have complete point clouds)
            S1_mask = torch.ones(self.subset_size, dtype=torch.int)
            C_mask = torch.ones(self.num_points, dtype=torch.int)
            
            # Apply transform if provided
            if self.transform:
                S1_points = self.transform(S1_points)
                C_points = self.transform(C_points)
            
            data.append((S1_points, S1_mask, C_points, C_mask, label))
        
        return data
    
    def _sample_points_from_mesh(self, mesh_path, num_points=None):
        """Sample points from a mesh file."""
        if num_points is None:
            num_points = self.num_points
            
        try:
            mesh = trimesh.load(mesh_path)
            points = mesh.sample(num_points)
            
            # Normalize to unit cube
            points = points - points.mean(axis=0)
            max_abs = np.max(np.abs(points))
            if max_abs > 0:  # Avoid division by zero
                points = points / max_abs
            
            return torch.FloatTensor(points)
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {e}")
            # Return random points as fallback
            return torch.randn(num_points, 3)


class ShapeNetNoisyVariableSubsetDataset(ShapeNetNoisySubsetDataset):
    """
    Extended version of ShapeNetNoisySubsetDataset that supports variable point cloud sizes.
    """
    def __init__(self, root_dir, categories=None, num_samples=1000, frac_subset=0.3, 
                 max_points_c=1024, min_subset_ratio=0.2, max_subset_ratio=0.8,
                 noise_level=0.02, transform=None):
        """
        Args:
            root_dir (str): Path to the ShapeNet dataset.
            categories (list): List of categories to use. If None, use all categories.
            num_samples (int): Number of sample pairs to generate.
            frac_subset (float): Fraction of samples that should be true subsets.
            max_points_c (int): Maximum number of points in complete clouds.
            min_subset_ratio (float): Minimum ratio of points for subsets.
            max_subset_ratio (float): Maximum ratio of points for subsets.
            noise_level (float): Standard deviation of Gaussian noise to add to subsets.
            transform (callable, optional): Optional transform to be applied to samples.
        """
        # Initialize base class attributes but don't call its __init__
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.frac_subset = frac_subset
        self.max_points_c = max_points_c
        self.min_subset_ratio = min_subset_ratio
        self.max_subset_ratio = max_subset_ratio
        self.noise_level = noise_level
        self.transform = transform
        
        # Load ShapeNet taxonomy and models (same as parent class)
        with open(os.path.join(root_dir, 'taxonomy.json'), 'r') as f:
            self.taxonomy = json.load(f)
            
        if categories is not None:
            self.taxonomy = [item for item in self.taxonomy if item['synsetId'] in categories]
        
        self.id_to_name = {item['synsetId']: item['name'] for item in self.taxonomy}
        
        self.models = []
        for item in self.taxonomy:
            synset_id = item['synsetId']
            category_path = os.path.join(root_dir, synset_id)
            if os.path.exists(category_path):
                model_ids = os.listdir(category_path)
                for model_id in model_ids:
                    model_path = os.path.join(category_path, model_id, 'models', 'model_normalized.obj')
                    if os.path.exists(model_path):
                        self.models.append((synset_id, model_id, model_path))
        
        print(f"Found {len(self.models)} models across {len(self.taxonomy)} categories")
        
        # Generate dataset with variable sizes
        self.data = self._generate_variable_data()
    
    def _generate_variable_data(self):
        """Generate dataset with varying point cloud and subset sizes."""
        data = []
        inv_frac = math.ceil(1 / self.frac_subset)
        
        # Minimum number of points to consider
        min_points_c = 128
        
        # Step size for efficiency
        step_size = 64
        
        # Generate varying cloud sizes
        cloud_sizes = list(range(min_points_c, self.max_points_c + 1, step_size))
        
        # Calculate samples per cloud size
        samples_per_size = self.num_samples // len(cloud_sizes)
        residual_samples = self.num_samples - samples_per_size * len(cloud_sizes)
        
        for c_idx, cloud_size in enumerate(tqdm(cloud_sizes, desc="Generating variable size pairs")):
            # Add residual samples to the last size
            is_last_size = c_idx == len(cloud_sizes) - 1
            samples_for_this_size = samples_per_size + (residual_samples if is_last_size else 0)
            
            for i in range(samples_for_this_size):
                # Randomly determine subset ratio for this sample
                subset_ratio = random.uniform(self.min_subset_ratio, self.max_subset_ratio)
                subset_size = max(32, int(cloud_size * subset_ratio))  # Ensure minimum subset size
                
                # Select a random model for C1
                idx1 = random.randint(0, len(self.models) - 1)
                _, _, path1 = self.models[idx1]
                
                # Load the point cloud C1
                C1_complete = self._sample_points_from_mesh(path1, self.max_points_c)
                C1_points = C1_complete[:cloud_size]
                
                # Create subset S1 from C1
                indices = torch.randperm(cloud_size)[:subset_size]
                S1_points = C1_points[indices].clone()
                
                # Add Gaussian noise to S1
                noise = torch.randn_like(S1_points) * self.noise_level
                S1_points = S1_points + noise
                
                if i % inv_frac == 0:
                    # Positive example: S1 (noisy subset) paired with C1
                    C_points = C1_points
                    label = 1  # S1 is derived from C1
                else:
                    # Negative example: S1 paired with different point cloud C2
                    idx2 = random.randint(0, len(self.models) - 1)
                    while idx2 == idx1:  # Ensure we get a different model
                        idx2 = random.randint(0, len(self.models) - 1)
                    
                    _, _, path2 = self.models[idx2]
                    C2_complete = self._sample_points_from_mesh(path2, self.max_points_c)
                    C_points = C2_complete[:cloud_size]
                    label = 0  # S1 is not derived from C2
                
                # Pad to maximum sizes
                S1_padding = torch.zeros(self.max_points_c - subset_size, 3)
                C_padding = torch.zeros(self.max_points_c - cloud_size, 3)
                
                S1_padded = torch.cat((S1_points, S1_padding), dim=0)
                C_padded = torch.cat((C_points, C_padding), dim=0)
                
                # Create masks
                S1_mask = torch.zeros(self.max_points_c, dtype=torch.int)
                C_mask = torch.zeros(self.max_points_c, dtype=torch.int)
                S1_mask[:subset_size] = 1
                C_mask[:cloud_size] = 1
                
                # Apply transform if provided
                if self.transform:
                    S1_padded = self.transform(S1_padded)
                    C_padded = self.transform(C_padded)
                
                data.append((S1_padded, S1_mask, C_padded, C_mask, label))
        
        # Shuffle the data
        random.shuffle(data)
        assert len(data) == self.num_samples, f"Generated {len(data)} samples, expected {self.num_samples}"
        
        return data

def download_modelnet(version=40):
    """
    Download and extract ModelNet dataset
    
    Args:
        version: 10 or 40 (for ModelNet10 or ModelNet40)
        target_dir: Directory to save the dataset
    """
    raw_datadir = os.path.join(os.getcwd(), 'raw_datasets')
    processed_datadir = os.path.join(os.getcwd(), 'processed_datasets')
    
    if version not in [10, 40]:
        raise ValueError("Version must be either 10 or 40")
    
    # Create target directory if it doesn't exist
    if not os.path.exists(raw_datadir): os.makedirs(raw_datadir, exist_ok=True)
    
    # URL for the dataset
    url = f"https://modelnet.cs.princeton.edu/ModelNet{version}.zip"
    zip_path = os.path.join(raw_datadir, f"ModelNet{version}.zip")
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(zip_path):
        print(f"Downloading ModelNet{version} dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f, tqdm(
            desc=f"ModelNet{version}", 
            total=total_size, 
            unit='B', 
            unit_scale=True, 
            unit_divisor=1024
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    
    # Extract the dataset if the directory doesn't exist
    extract_dir = os.path.join(processed_datadir, f"ModelNet{version}")
    if not os.path.exists(extract_dir):
        print(f"Extracting ModelNet{version} dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(processed_datadir)
    
    print(f"ModelNet{version} dataset is ready at {extract_dir}")
    return extract_dir

class ModelNetSubsetDataset(Dataset):
    """
    Dataset generates:
    - Noisy positives: S1 is a noisy subset of C, label=1
    - True negatives: S1 is from a different shape, label=0
    """
    def __init__(self, modelnet_path, version=40, s2_size=1024, s1_size = 512, noise_level=0.02, pos_neg_ratio = 0.1, split='train'):
        
        self.modelnet_path = modelnet_path
        self.version = version
        self.s2_size = s2_size
        self.s1_size = s1_size
        self.noise_level = noise_level
        self.split = split
        self.pos_neg_ratio = pos_neg_ratio
        
        self.shape_paths = []
        categories_dir = self.modelnet_path
        if not os.path.exists(categories_dir):
            raise ValueError(f"ModelNet{version} directory not found at {categories_dir}")
        
        categories = sorted([d for d in os.listdir(categories_dir) 
                            if os.path.isdir(os.path.join(categories_dir, d))])
        
        print(f"Found {len(categories)} categories in ModelNet{version}")
        
        # Get all shape paths
        for category in categories:
            category_split_dir = os.path.join(categories_dir, category, split)
            if not os.path.exists(category_split_dir):
                print(f"Warning: Split directory {category_split_dir} doesn't exist")
                continue
                
            # Get all .off files in this category and split
            off_files = glob.glob(os.path.join(category_split_dir, '*.off'))
            for off_file in off_files:
                self.shape_paths.append((off_file, category))
        
        print(f"Found {len(self.shape_paths)} shapes in ModelNet{version} {split} split")
    
    def __len__(self):
        return len(self.shape_paths) * 2  # Each shape produces positive and negative examples
    
    def __getitem__(self, idx):
        # Determine if this is a positive or negative example
        is_positive = idx % 2 == 0
        shape_idx = idx // 2
        
        # Load the first shape (complete point cloud)
        obj_path, category = self.shape_paths[shape_idx]
        complete_points = self._sample_points_from_mesh(obj_path)
        #print(f"complete points shape: {complete_points.shape}")
        
        subset_size = self.s1_size
        if is_positive:
            # Create a subset from the same point cloud
            indices = np.random.choice(self.s2_size, subset_size, replace=False)
            subset_points = complete_points[indices].copy()
            
            # Add noise to the subset
            noise = np.random.normal(0, self.noise_level, subset_points.shape)
            subset_points = subset_points + noise
            
            label = 1.0  # Is a subset
        else:
            # Use a different shape as the "subset" (not actually a subset)
            neg_idx = (shape_idx + 1) % len(self.shape_paths)
            neg_obj_path, _ = self.shape_paths[neg_idx]
            
            subset_points = self._sample_points_from_mesh(neg_obj_path)
            indices = np.random.choice(self.s2_size, subset_size, replace=False)
            subset_points = subset_points[indices]
            
            label = 0.0  # Not a subset
        
        # Create fixed-size tensors with masks
        S1 = np.zeros((self.s1_size, 3), dtype=np.float32)
        S1_mask = np.zeros(self.s1_size, dtype=np.float32)
        
        subset_size = len(subset_points)
        S1[:subset_size] = subset_points
        S1_mask[:subset_size] = 1.0
        
        C = complete_points
        C_mask = np.ones(self.s2_size, dtype=np.float32)
        
        return torch.FloatTensor(S1), torch.FloatTensor(S1_mask), torch.FloatTensor(C), torch.FloatTensor(C_mask), label
    
    def _sample_points_from_mesh(self, off_path):
        """Sample points from a ModelNet .off file"""
        try:
            mesh = trimesh.load(off_path)
            points, _ = trimesh.sample.sample_surface(mesh, self.s2_size)
            
            # Center and normalize
            centroid = points.mean(axis=0)
            points = points - centroid
            max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
            points = points / max_dist
            
            return points.astype(np.float32)
        except Exception as e:
            print(f"Error loading mesh {off_path}: {e}")
            # Return random points if loading fails
            return np.random.randn(self.s2_size, 3).astype(np.float32)
            
    def visualize_sample(self, idx):
        """Visualize a sample from the dataset"""
        S1, S1_mask, C, C_mask, label = self[idx]
        S1, S1_mask = S1.cpu().numpy(), S1_mask.cpu().numpy()
        C, C_mask = C.cpu().numpy(), C_mask.cpu().numpy()
        
        # Extract valid points based on masks
        valid_S1 = S1[S1_mask > 0.5]
        valid_C = C[C_mask > 0.5]
        
        fig = plt.figure(figsize=(12, 5))
        
        # Plot subset points
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(valid_S1[:, 0], valid_S1[:, 1], valid_S1[:, 2], c='r', s=20, alpha=0.6)
        ax1.set_title(f"Potential Subset Points")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim([-1.2, 1.2])
        ax1.set_ylim([-1.2, 1.2])
        ax1.set_zlim([-1.2, 1.2])
        
        # Plot complete points
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(valid_C[:, 0], valid_C[:, 1], valid_C[:, 2], c='b', s=20, alpha=0.6)
        ax2.set_title(f"Complete Points")
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_xlim([-1.2, 1.2])
        ax2.set_ylim([-1.2, 1.2])
        ax2.set_zlim([-1.2, 1.2])
        
        plt.suptitle(f"Sample {idx}, Label: {'Is Subset' if label > 0.5 else 'Not Subset'}")
        plt.tight_layout()
        plt.show()
        
        return fig
    
class FastModelNetSubsetDataset(Dataset):
    """
    Dataset generates:
    - Noisy positives: S1 is a noisy subset of C, label=1
    - True negatives: S1 is from a point cloud of a different category, label=0
    
    Supports pre-generating a fixed number of samples and saving to disk for fast loading.
    """
    def __init__(self, modelnet_path, cache_dir=None, num_samples=10000, version=40, s2_size=1024, s1_size=512, 
                 noise_level=0.02, pos_neg_ratio=0.5, split='train', force_regenerate=False, device = 0):
        """
        Initialize the ModelNetSubsetDataset.
        
        Args:
            modelnet_path: Path to the ModelNet directory
            cache_dir: Directory to save/load pre-generated samples (if None, uses modelnet_path/cache)
            num_samples: Total number of samples to generate
            version: ModelNet version (40 or 10)
            s2_size: Number of points in the complete point cloud
            s1_size: Number of points in the subset
            noise_level: Standard deviation of Gaussian noise added to positive examples
            pos_neg_ratio: Ratio of positive samples to total samples (e.g., 0.5 means 50% positive, 50% negative)
            split: Dataset split ('train' or 'test')
            force_regenerate: If True, regenerate cached data even if it exists
        """
        
        self.modelnet_path = modelnet_path
        self.version = version
        self.s2_size = s2_size
        self.s1_size = s1_size
        self.noise_level = noise_level
        self.split = split
        self.pos_neg_ratio = pos_neg_ratio
        self.num_samples = num_samples
        self.force_regenerate = force_regenerate
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.path.join(modelnet_path, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        
        # Create cache filename based on parameters
        cache_params = f"v{version}_s2_{s2_size}_s1_{s1_size}_P2N_{pos_neg_ratio}_ns_{num_samples}_{split}"
        self.cache_file = os.path.join(cache_dir, f"modelnet_subset_{cache_params}.pkl")
        
        # For reproducibility
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        
        # Initialize dataset info
        self._load_dataset_info()
        
        # Load or generate data
        if os.path.exists(self.cache_file) and not force_regenerate:
            print(f"Loading pre-generated data from {self.cache_file}")
            self._load_cached_data()
        else:
            print(f"Generating {num_samples} samples and saving to {self.cache_file}")
            self._generate_and_cache_data()
        
    
    def _load_dataset_info(self):
        """Load information about the ModelNet dataset (categories, file paths)"""
        
        self.shape_paths = []
        categories_dir = self.modelnet_path
        if not os.path.exists(categories_dir):
            raise ValueError(f"ModelNet{self.version} directory not found at {categories_dir}")
        
        categories = sorted([d for d in os.listdir(categories_dir) 
                            if os.path.isdir(os.path.join(categories_dir, d))])
        
        print(f"Found {len(categories)} categories in ModelNet{self.version}")
        
        # Get all shape paths
        self.categories = categories
        self.shapes_by_category = {category: [] for category in categories}
        
        for category in categories:
            category_split_dir = os.path.join(categories_dir, category, self.split)
            if not os.path.exists(category_split_dir):
                print(f"Warning: Split directory {category_split_dir} doesn't exist")
                continue
                
            # Get all .off files in this category and split
            off_files = glob.glob(os.path.join(category_split_dir, '*.off'))
            for off_file in off_files:
                self.shape_paths.append((off_file, category))
                self.shapes_by_category[category].append(off_file)
                
        # Log statistics about categories
        for category in categories:
            count = len(self.shapes_by_category[category])
            #print(f"  - Category '{category}': {count} shapes")
        
        print(f"Found {len(self.shape_paths)} shapes in ModelNet{self.version} {self.split} split")
    
    def _load_cached_data(self):
        """Load pre-generated data from disk cache"""
        start_time = time.time()
        
        # Load the cached data
        with open(self.cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Unpack the data
        self.all_S1 = cached_data['S1']
        self.all_S1_mask = cached_data['S1_mask']
        self.all_C = cached_data['C']
        self.all_C_mask = cached_data['C_mask']
        self.all_labels = cached_data['labels']
        
        # Convert to tensors if they're not already
        if not isinstance(self.all_S1, torch.Tensor):
            self.all_S1 = torch.FloatTensor(self.all_S1)
            self.all_S1_mask = torch.FloatTensor(self.all_S1_mask)
            self.all_C = torch.FloatTensor(self.all_C)
            self.all_C_mask = torch.FloatTensor(self.all_C_mask)
            self.all_labels = torch.FloatTensor(self.all_labels)
            
        noise = torch.randn_like(self.all_S1) * self.noise_level
        num_pos = torch.sum(self.all_labels).item()
        p2n_data = num_pos/(len(self.all_labels) - num_pos)
        self.all_S1 += noise
        
        elapsed = time.time() - start_time
        print(f"Loaded {len(self.all_labels)} samples from cache with P2N: {p2n_data} in {elapsed:.2f} seconds")
    
    def _generate_and_cache_data(self):
        """Generate samples and save them to disk"""
        start_time = time.time()
        
        # Load all complete point clouds first
        print("Loading point clouds...")
        complete_point_clouds = []
        categories = []
        shape_paths = self.shape_paths[:self.num_samples]
        print(f"Length of shape paths: {len(shape_paths)}")
        for obj_path, category in tqdm(shape_paths):
            points = self._sample_points_from_mesh(obj_path)
            # Convert to PyTorch tensor immediately
            points_tensor = torch.FloatTensor(points)
            complete_point_clouds.append(points_tensor)
            categories.append(category)
        
        # Calculate number of positive and negative samples
        num_positives = int(self.num_samples * self.pos_neg_ratio/(1 + self.pos_neg_ratio))
        num_negatives = self.num_samples - num_positives
        
        print(f"Generating {num_positives} positive samples and {num_negatives} negative samples")
        
        # Pre-allocate tensors
        all_S1 = torch.zeros((self.num_samples, self.s1_size, 3), dtype=torch.float32).to(self.device)
        all_S1_mask = torch.zeros((self.num_samples, self.s1_size), dtype=torch.float32).to(self.device)
        all_C = torch.zeros((self.num_samples, self.s2_size, 3), dtype=torch.float32).to(self.device)
        all_C_mask = torch.ones((self.num_samples, self.s2_size), dtype=torch.float32).to(self.device)
        all_labels = torch.zeros(self.num_samples, dtype=torch.float32).to(self.device)
        
        # Generate positive samples
        print("Generating positive samples...")
        num_shapes = len(complete_point_clouds)
        
        for i in tqdm(range(num_positives)):
            # Select a random shape
            try: 
                shape_idx = torch.randint(0, num_shapes, (1,)).item()
            except:
                print(f"num_shapes: {num_shapes}")
                os._exit(1)
            complete_points = complete_point_clouds[shape_idx]
            
            # Create a subset from the same point cloud
            indices = torch.randperm(self.s2_size)[:self.s1_size]
            subset_points = complete_points[indices].clone()
            
            # Store positive example
            all_S1[i, :len(subset_points)] = subset_points
            all_S1_mask[i, :len(subset_points)] = 1.0
            all_C[i] = complete_points
            all_labels[i] = 1.0
        
        # Generate negative samples
        print("Generating negative samples...")
        for i in tqdm(range(num_negatives)):
            idx = num_positives + i
            
            # Select a random shape as the complete point cloud
            shape_idx = torch.randint(0, num_shapes, (1,)).item()
            complete_points = complete_point_clouds[shape_idx]
            current_category = categories[shape_idx]
            
            # Find shapes from different categories
            different_category_indices = [
                j for j, category in enumerate(categories)
                if category != current_category
            ]
            
            # If we have shapes from different categories, use one of them
            if different_category_indices:
                # Randomly select a shape from a different category
                neg_shape_idx = different_category_indices[torch.randint(0, len(different_category_indices), (1,)).item()]
            else:
                # Fallback: use a different shape (potentially from same category)
                neg_shape_idx = (shape_idx + 1) % num_shapes
                
            neg_complete_points = complete_point_clouds[neg_shape_idx]
            
            # Sample a subset from the negative shape
            neg_indices = torch.randperm(self.s2_size)[:self.s1_size]
            neg_subset_points = neg_complete_points[neg_indices]
            
            # Store negative example
            all_S1[idx, :len(neg_subset_points)] = neg_subset_points
            all_S1_mask[idx, :len(neg_subset_points)] = 1.0
            all_C[idx] = complete_points
            all_labels[idx] = 0.0
        
        # Save the data
        print(f"Saving data to {self.cache_file}...")
        # Convert to numpy arrays for more efficient storage
        cached_data = {
            'S1': all_S1.cpu().numpy(),
            'S1_mask': all_S1_mask.cpu().numpy(),
            'C': all_C.cpu().numpy(),
            'C_mask': all_C_mask.cpu().numpy(),
            'labels': all_labels.cpu().numpy()
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        
        # Assign the data
        self.all_S1 = all_S1
        self.all_S1_mask = all_S1_mask
        self.all_C = all_C
        self.all_C_mask = all_C_mask
        self.all_labels = all_labels
        
        elapsed = time.time() - start_time
        print(f"Generated and cached {self.num_samples} samples in {elapsed:.2f} seconds")
    
    def _sample_points_from_mesh(self, off_path):
        """Sample points from a ModelNet .off file"""
        try:
            mesh = trimesh.load(off_path)
            points, _ = trimesh.sample.sample_surface(mesh, self.s2_size)
            
            # Center and normalize
            centroid = points.mean(axis=0)
            points = points - centroid
            max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
            points = points / max_dist
            
            return points.astype(np.float32)
        except Exception as e:
            print(f"Error loading mesh {off_path}: {e}")
            # Return random points if loading fails
            return np.random.randn(self.s2_size, 3).astype(np.float32)
    
    def __len__(self):
        """Return the total number of samples"""
        return len(self.all_labels)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        return (
            self.all_S1[idx],
            self.all_S1_mask[idx],
            self.all_C[idx],
            self.all_C_mask[idx],
            self.all_labels[idx]
        )
        
    def visualize_sample(self, idx):
        """Visualize a sample from the dataset"""
        S1, S1_mask, C, C_mask, label = self[idx]
        S1, S1_mask = S1.cpu().numpy(), S1_mask.cpu().numpy()
        C, C_mask = C.cpu().numpy(), C_mask.cpu().numpy()
        
        # Extract valid points based on masks
        valid_S1 = S1[S1_mask > 0.5]
        valid_C = C[C_mask > 0.5]
        
        fig = plt.figure(figsize=(12, 5))
        
        # Plot subset points
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(valid_S1[:, 0], valid_S1[:, 1], valid_S1[:, 2], c='r', s=20, alpha=0.6)
        ax1.set_title(f"Potential Subset Points")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim([-1.2, 1.2])
        ax1.set_ylim([-1.2, 1.2])
        ax1.set_zlim([-1.2, 1.2])
        
        # Plot complete points
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(valid_C[:, 0], valid_C[:, 1], valid_C[:, 2], c='b', s=20, alpha=0.6)
        ax2.set_title(f"Complete Points")
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_xlim([-1.2, 1.2])
        ax2.set_ylim([-1.2, 1.2])
        ax2.set_zlim([-1.2, 1.2])
        
        plt.suptitle(f"Sample {idx}, Label: {'Is Subset' if label > 0.5 else 'Not Subset'}")
        plt.tight_layout()
        plt.show()
        plt
        return fig
        
def get_pointcloud_dataset(args):
    pos_neg_ratio = args.P2N
    modelnet_path = os.path.join(os.getcwd(), "processed_datasets/ModelNet40")
    num_samples_train = 10000
    num_samples_test = 1000
    noise_scale = args.noise_scale
    s1_size = args.s1_size
    train_dataset = FastModelNetSubsetDataset(modelnet_path, num_samples=num_samples_train, pos_neg_ratio=pos_neg_ratio, split='train', noise_level=noise_scale, s1_size=s1_size)
    val_dataset = FastModelNetSubsetDataset(modelnet_path, num_samples=num_samples_test, pos_neg_ratio=pos_neg_ratio, split='test', noise_level=noise_scale, s1_size=s1_size)
    test_dataset = FastModelNetSubsetDataset(modelnet_path, num_samples=num_samples_test, pos_neg_ratio=pos_neg_ratio, split='test', noise_level=noise_scale, s1_size=s1_size)
    return train_dataset, val_dataset, test_dataset

def fast_local_sampling(points, center_point, radius, n_samples, k=20):
    """
    Quickly sample a local subset of points with reasonable structure preservation
    
    Args:
        points: Input point cloud (N, 3)
        center_point: Center point of the local region (3,)
        radius: Radius of the local region to consider
        n_samples: Number of points to sample
        k: Number of neighbors to consider
    
    Returns:
        indices of sampled points
    """
    device = points.device
    
    # Step 1: Select points within the local region (fast distance calculation)
    distances_to_center = torch.norm(points - center_point.unsqueeze(0), dim=1)
    local_mask = distances_to_center <= radius
    local_indices = torch.nonzero(local_mask).squeeze()
    
    # If we have fewer points in the region than requested samples, return all points
    if len(local_indices) <= n_samples:
        print("this happening")
        _, topk_indices = torch.topk(distances_to_center, k=min(n_samples, points.shape[0]), largest=False)
        return points[topk_indices]
    
    # Get the local points
    local_points = points[local_indices]
    local_N = local_points.shape[0]
    
    # Step 2: Fast importance sampling
    # Use distance from center as a simple proxy for importance
    # This avoids the expensive kNN computation for all points
    center_importance = 1.0 / (distances_to_center[local_indices] + 1e-10)
    
    # Add some noise to break ties and increase diversity
    noise = torch.rand(local_N, device=device) * 0.1
    sampling_weights = center_importance + noise
    
    # Step 3: Sample the first point (closest to center)
    closest_idx = torch.argmin(distances_to_center[local_indices]).item()
    sampled_local_indices = [closest_idx]
    
    # Step 4: Sample the rest using a hybrid approach:
    # - Some points are sampled proportionally to their importance
    # - Some are sampled as neighbors of already sampled points
    
    # Percentage of points to sample randomly based on importance
    random_sample_percent = 0.3
    n_random = int(n_samples * random_sample_percent)
    n_neighbors = n_samples - 1 - n_random  # -1 for the first point
    
    # Compute k-NN graph only for the selected first point to save computation
    # This is much faster than computing the full k-NN graph
    closest_point = local_points[closest_idx].unsqueeze(0)
    dists = torch.norm(local_points - closest_point, dim=1)
    _, neighbor_indices = torch.topk(dists, k=min(k+1, local_N), largest=False)
    
    # Remove the point itself from its neighbors
    neighbor_indices = neighbor_indices[1:k+1] if len(neighbor_indices) > k else neighbor_indices[1:]
    
    # Sample top neighbors
    sampled_local_indices.extend(neighbor_indices[:min(n_neighbors, len(neighbor_indices))].tolist())
    
    # If we need more points, sample randomly based on importance
    if len(sampled_local_indices) < n_samples:
        remaining_to_sample = n_samples - len(sampled_local_indices)
        
        # Create a mask for already sampled points
        already_sampled = torch.zeros(local_N, dtype=torch.bool, device=device)
        for idx in sampled_local_indices:
            already_sampled[idx] = True
        
        # Zero out the weights for already sampled points
        remaining_weights = sampling_weights.clone()
        remaining_weights[already_sampled] = 0
        
        # Normalize weights to create a probability distribution
        if remaining_weights.sum() > 0:
            remaining_weights = remaining_weights / remaining_weights.sum()
            
            # Sample without replacement
            remaining_indices = torch.multinomial(
                remaining_weights, 
                num_samples=min(remaining_to_sample, (remaining_weights > 0).sum().item()),
                replacement=False
            )
            
            sampled_local_indices.extend(remaining_indices.tolist())
    
    # Convert back to original point cloud indices
    sampled_indices = local_indices[sampled_local_indices]
    
    return points[sampled_indices]

def choose_centres(points, n_regions, target_samples):
    N = points.shape[0]
    device = points.device
    indices = torch.randperm(N, device=device)[:n_regions]
    centers = points[indices]
    volume = torch.prod(points.max(dim=0)[0] - points.min(dim=0)[0])
    point_density = N / volume
    # Target having ~region_size points per region
    target_size = min(target_samples * 2, N)
    radii = (target_size / point_density) ** (1/3)
    radii = torch.full((n_regions,), radii, device=device)
    return centers, radii

def sample_local_subsets(points, n_samples):
    start = time.time()
    centers, radii = choose_centres(points, n_samples, 512)
    samples_list = []
    for i in range(n_samples):
        center = centers[i]
        radius = radii[i]
        
        samples = fast_local_sampling(points, center, radius, 512, 8)
        print(f"one sample shape: {samples.shape}")
        samples_list.append(samples)
    end = time.time()
    print(f"{n_samples} points sampled in time: {end - start}s")
    return samples_list

if __name__=='__main__':
    modelnet_path = os.path.join("/mnt/nas/soutrik/Monotone-Clean/processed_datasets/ModelNet40")
    num_samples_train = 10000
    num_samples_test = 1000
    p2n_ratio = 1.0
    ap = ArgumentParser()
    ap.add_argument('--s1_size', type=int, default=512, help='Subset size S1')
    ap.add_argument('--device', type=int, default=0, help='GPU device ID')
    args = ap.parse_args()
    train_datset = FastModelNetSubsetDataset(modelnet_path, num_samples = num_samples_train, s1_size=args.s1_size, pos_neg_ratio=p2n_ratio, split='train', device=args.device)
    test_datset = FastModelNetSubsetDataset(modelnet_path, num_samples = num_samples_test, s1_size=args.s1_size, pos_neg_ratio=p2n_ratio, split='test', device = args.device)