import math
import pickle
import random
import torch.nn as nn
import os
import csv
import gdown
import zipfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
#from msweb_data_generator import fix_seed
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from lightning.pytorch import seed_everything

base_datadir_name = '/mnt/nas/soutrik/Monotone-Clean/raw_datasets/amazon'

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    seed_everything(seed, workers=True)

def read_from_pickle(filename):
    filename = filename + '.pickle'
    with open(filename, 'rb') as f:
        x = pickle.load(f)
        return x

class Amazon:
    def __init__(self, dataset_name: str, s2_size : int, pos_neg_ratio: float):
        self.dataset_name = dataset_name
        self.s2_size = s2_size
        self.pos_neg_ratio = pos_neg_ratio

    def download_amazon(self):
        url = 'https://drive.google.com/uc?id=1OLbCOTsRyowxw3_AzhxJPVB8VAgjt2Y6'
        
        print(os.getcwd())
        base_data_path = os.path.join(os.getcwd(), base_datadir_name)
        if not os.path.exists(base_data_path):
            os.makedirs(base_data_path)
        zipfile_path = os.path.join(base_data_path, 'amazon_baby_registry.zip')
        if not os.path.exists(zipfile_path):
            gdown.download(url, zipfile_path, quiet=False)
            with zipfile.ZipFile(zipfile_path, 'r') as ziphandler:
                ziphandler.extractall(base_data_path)
        else:
            print('Zipfile already there')
            
        return base_data_path

    def read_real_data(self, data_root):
        pickle_filename = data_root + '/' + self.dataset_name
        dataset_ = read_from_pickle(pickle_filename)
        for i in range(len(dataset_)):
            dataset_[i+1]  = torch.tensor(dataset_[i+1])
        data_ = torch.zeros(len(dataset_), dataset_[1].shape[0])
        for i in range(len(dataset_)):
            data_[i,:] = dataset_[i+1]
        
        csv_filename = data_root + '/' + self.dataset_name + '.csv'
        with open(csv_filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',',  quoting=csv.QUOTE_NONNUMERIC,quotechar='|')
            S = {}
            i=-1
            for row in reader:
                i=i+1
                S[i] = torch.tensor([int(row[x]) for x in range(len(row))]).long()
        return data_ , S
    
    def filter_S(self, data, S, noise_scale = 0.01):
        """ S: list[tensor(int)]-> list of index list (each index list is torch.tensor)"""
        data_list = []
        S_shapes = [S[i].shape[0] for i in S.keys()]
        s1_size_max = min(max(S_shapes), self.s2_size-1)
        running_pos_count, running_neg_count = 0, 0
        pc, nc = 0, 0
        eps = 1e-6
        for i in range(len(S)):
            if S[i].shape[0]>2 and S[i].shape[0] < self.s2_size:
                Svar  = S[i] - 1    # index starts froms 0
                running_pos_neg_ratio = running_pos_count/(running_neg_count + eps)
                actual_subset = running_pos_neg_ratio < self.pos_neg_ratio
                #here is where the set construction stuff happens
                S1, S1_mask, S2, S2_mask, label = self.construct_ground_set(data, Svar, V=self.s2_size, actual_subset=actual_subset, max_S1_size = s1_size_max, noise_scale = noise_scale) 
                if label == 1: 
                    pc += 1
                else:
                    nc += 1
                data_list.append((S1, S1_mask, S2, S2_mask, label))
                if actual_subset: running_pos_count += 1
                else: running_neg_count += 1
        return data_list
    
    def construct_ground_set(self, data, S, V, actual_subset: bool, max_S1_size: int, noise_scale: float = 0.01):
        S_data = data[S]  # tensors corresponding to S indices, shape: (|S|, 768)
        
        # Decide how many elements from S1 will be in S2
        num_S1_in_S2 = S.shape[0] if actual_subset else random.randint(0, S.shape[0]-1)
        
        # Shuffle S and select the first num_S1_in_S2 elements to be the intersection
        perm = torch.randperm(S.shape[0])
        S1_indices_in_S = perm[:num_S1_in_S2]  # These are indices into S
        S1_in_S2 = S[S1_indices_in_S]  # These are the actual indices in the original data
        S1_in_S2_data = data[S1_in_S2]  # The embedding data for those indices
        
        all_indices = torch.ones(data.shape[0], dtype=bool)  # Start with all True
        all_indices[S] = False  # Set indices in S to False
        remaining_indices_tensor = torch.where(all_indices)[0]  # Get indices where mask is True
        remaining_data = data[remaining_indices_tensor]  # Data not in S
        
        # Calculate similarity to mean of S1
        S_mean = S_data.mean(dim=0).unsqueeze(0)
        S_mean_norm = F.normalize(S_mean, dim=-1)
        remaining_data_norm = F.normalize(remaining_data, dim=-1)
        
        # Get cosine similarities
        cos_sim = (S_mean_norm @ remaining_data_norm.T).squeeze(0)
        _, idx = torch.sort(cos_sim)  # Sort by similarity (ascending)
        
        # Select elements for S2 that aren't in S1
        needed_elements = V - num_S1_in_S2
        S2_non_S1_indices = remaining_indices_tensor[idx[:needed_elements]]
        S2_non_S1_data = data[S2_non_S1_indices]
        
        # Create S2 with proper elements
        S2_data = torch.zeros(V, data.shape[1])
        S2_data[:num_S1_in_S2] = S1_in_S2_data
        S2_data[num_S1_in_S2:] = S2_non_S1_data
        
        # Create masks
        S1_mask = torch.zeros(max_S1_size, dtype=int)
        S1_mask[:S.shape[0]] = 1
        S2_mask = torch.ones(V, dtype=int)
        
        # Label: 1 if S1 is a subset of S2, 0 otherwise
        label = 1 if actual_subset else 0
        
        # Pad S_data to match max_S1_size
        S1_padding = torch.zeros(max_S1_size - S_data.shape[0], S_data.shape[1])
        white_noise = torch.randn_like(S_data) * noise_scale
        S_data = torch.cat([S_data + white_noise, S1_padding], axis=0)
        
        return S_data, S1_mask, S2_data, S2_mask, label
    
    def split_into_training_test(self, data_list):
        folds = [0.7, 0.15, 0.15]
        num_elem = len(data_list)
        tr_size = int(folds[0]* num_elem)
        dev_size = int((folds[1]+folds[0])* num_elem)
        test_size = num_elem
        
        random.shuffle(data_list)
        data_train = data_list[0:tr_size]
        data_val = data_list[tr_size:dev_size]
        data_test = data_list[dev_size:test_size]
        
        return data_train, data_val, data_test
    
class AmazonDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def get_amazon_datasets(dataset_name: str, pos_neg_ratio: float = 0.1, s2_size: int = 30, noise_scale: float = 0.01):
    amazon_data = Amazon(dataset_name, s2_size, pos_neg_ratio)
    print("dataset object created")
    base_data_path = amazon_data.download_amazon()
    print("raw data downloaded")
    data, S = amazon_data.read_real_data(base_data_path)
    print("Real Data Read from CSV/Pickle files")
    data_list = amazon_data.filter_S(data, S, noise_scale)
    train_data, val_data, test_data = amazon_data.split_into_training_test(data_list)
    print("Data lists created and split")
    train_ds, test_ds, val_ds = AmazonDataset(train_data), AmazonDataset(val_data), AmazonDataset(test_data)
    return train_ds, val_ds, test_ds
    
def unittest(dataset, dataset_type: str, pos_neg_ratio: float):
    print(f"Dataset Type: {dataset_type}, length: {len(dataset)}")
    testing_size = len(dataset)//1
    testing_idx = np.random.permutation(len(dataset))[:testing_size]
    pos_count= 0
    neg_count = 0
    for idx in tqdm(testing_idx):
        s1, mask1, s2, mask2, label = dataset[idx]
        size1 = torch.sum(mask1, dtype = int)
        size2 = torch.sum(mask2, dtype = int)
        zero_threshold = 1e-10
        matched_in_s2 = [False] * size2
        set_contained = True
        #print(f"Label in testing, idx: {idx}, label: {label}, type: {type(label)}")
        if label == 1:
            pos_count += 1
        else:
            neg_count += 1
        for ind_1 in range(size1):
            element_contained = False
            for ind_2 in range(size2):
                is_a_match = (not matched_in_s2[ind_2]) and (torch.norm(s1[ind_1]-s2[ind_2]) < zero_threshold)
                if is_a_match:
                    element_contained = True
                    matched_in_s2[ind_2] = True
                    break
            set_contained = set_contained and element_contained
            if not set_contained: break
        assert int(set_contained) == label, f"Idx: {idx}, label: {label}, reality: {int(set_contained)}"
    
    assert math.isclose(pos_count/neg_count, pos_neg_ratio, abs_tol=6e-2), f"P2N data: {pos_count/neg_count}, P2N supposed: {pos_neg_ratio}"
    print(f"All label correction tests passed for {dataset_type} data")
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_name',  type=str)
    parser.add_argument('--P2N',           type=float, default = 1.0)
    parser.add_argument('--noise_scale',   type=float, default=0.0)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    train_data, val_data, test_data = get_amazon_datasets(dataset_name, pos_neg_ratio=args.P2N, noise_scale = args.noise_scale)
    #s1, m1, s2, m2, label = train_data[2025]
    
    unittest(train_data, 'train', args.P2N)
    unittest(val_data, 'val', args.P2N)
    unittest(test_data, 'test', args.P2N)
    print(f'checked for {dataset_name}')
    
#media, safety, toys, health, gear, feeding, diaper, bedding, bath, apparel