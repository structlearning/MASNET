import argparse
import os
import random
import gzip
from tqdm import tqdm
import urllib.request
import torch
import numpy as np
import math
from sentence_transformers import SentenceTransformer, models
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = False
import pickle
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from lightning.pytorch import seed_everything
import sys
import json

raw_datadir_base = '/mnt/nas/soutrik/Monotone-Clean/raw_datasets'
raw_datadir_msweb = os.path.join(raw_datadir_base, 'msweb_data')
raw_datadir_msnbc = os.path.join(raw_datadir_base, 'msnbc_data')

processed_datadir_base = '/mnt/nas/soutrik/Monotone-Clean/data/mnt/nas/soutrik/Monotone-Clean/processed_datasets'
processed_datadir_msweb = os.path.join(processed_datadir_base, 'MSWEB')
processed_datadir_msnbc = os.path.join(processed_datadir_base, 'MSNBC')

dataset_sizes_config = lambda av: [av.TRAIN_DATASET_SIZE, av.VAL_DATASET_SIZE, av.TEST_DATASET_SIZE]
data_types = ['train', 'val', 'test']
tensor_file_names = lambda av: [f'{data_types[idx]}_query_tensors_size_{dataset_sizes_config(av)[idx]}_p2n_{av.P2N}.pt' for idx in range(3)]
ind1_ind2_label_file_names = lambda av: [f'{data_types[idx]}_index_label_size_{dataset_sizes_config(av)[idx]}_p2n_{av.P2N}.pt' for idx in range(3)]
corpus_file_name = 'corpus_tensors.pt'
corpus_file_name_np = 'corpus_tensors_as_numpy.npz'
corpus_mask_file_name = 'corpus_masks.pt'

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    seed_everything(seed, workers=True)
    
def makedirs():
  dir_list = [raw_datadir_msweb, raw_datadir_msnbc, processed_datadir_msweb, processed_datadir_msnbc]
  for dir in dir_list:
    if not os.path.exists(dir): os.makedirs(dir)

def issubset_multiset(X, Y):
  #checks if X is saubset of Y for multisets
    return len(Counter(X)-Counter(Y)) == 0

def issubset_set(X, Y):
  #checks if X is saubset of Y
  return X.issubset(Y)

def fetch_all_msweb_data(fname):
    print(f"Fname: {fname}")
    if not os.path.exists(fname):
        urllib.request.urlretrieve("https://kdd.ics.uci.edu/databases/msweb/anonymous-msweb.data.gz", fname)
    fIn =  gzip.open(fname, 'rt', encoding='utf8')
    elements = []
    corpus = []
    curr_corpus_item = set()
    for l in fIn:
        # print(l[0], l)
        if l[0] == 'A':
            pieces = l.split(",")
            elements.append({"id":int(pieces[1]), "text": pieces[3].strip('"'), "url": pieces[4].strip('/"')})
        if l[0] == 'C':
            corpus.append(curr_corpus_item)
            curr_corpus_item = set()
        if l[0] == 'V':
            curr_corpus_item.add(int(l.split(",")[1]))
    
    unique_corpus = []
    for i,c1 in tqdm(enumerate(corpus)):
        isUnique = True
        for j in range(i+1, len(corpus)):
            if c1 == corpus[j]:
                isUnique = False
        if isUnique:
            unique_corpus.append(c1)

    return elements, unique_corpus

def fetch_all_msnbc_data(fname):
    if not os.path.exists(fname):
        urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/msnbc-mld/msnbc990928.seq.gz", fname)
    fIn =  gzip.open(fname, 'rt', encoding='utf8')
    elements = []
    corpus = []
    stringified_corpus = []
    curr_corpus_item = list()
    for l in tqdm(fIn):
        l = l.strip()
        if  l == '' or l[0] == '%':
            continue
        tokens = l.split(" ")
        if tokens[0].isnumeric():
            for t in tokens:
                curr_corpus_item.append(int(t))
            stringified = "@".join([str(e) for e in sorted(curr_corpus_item)])
            if stringified not in stringified_corpus:
                corpus.append(curr_corpus_item)
                stringified_corpus.append(stringified)
            curr_corpus_item = list()
        else:
            elements = [{"id":i+1, "text": t} for i,t in enumerate(tokens)]
    filtered_corpus = [c for c in corpus if len(c) <= 50]

    return elements, filtered_corpus

def load_all_ms_data(dataset_name: str, corpus_frac: float = 1.0):
  fname_raw_msweb = os.path.join(raw_datadir_msweb, "anonymous-msweb.data.gz")
  fname_raw_msnbc = os.path.join(raw_datadir_msnbc, "msnbc990928.seq.gz")
  fname_unboxed_msweb = os.path.join(raw_datadir_msweb, "processed.pkl")
  fname_unboxed_msnbc = os.path.join(raw_datadir_msnbc, "processed.pkl")
  
  if dataset_name == "MSWEB" and not os.path.exists(fname_unboxed_msweb):
    print("raw MSWEB does not exist")
    elements, corpus = fetch_all_msweb_data(fname_raw_msweb)
    data = {'elements': elements, 'corpus': corpus}
    pickle.dump(data, open(fname_unboxed_msweb, "wb"))
    
  elif dataset_name == "MSNBC" and not os.path.exists(fname_unboxed_msnbc):
    print("raw MSNBC does not exist")
    elements, corpus = fetch_all_msnbc_data(fname_raw_msnbc)
    data = {'elements': elements, 'corpus': corpus}
    pickle.dump(data, open(fname_unboxed_msnbc, "wb"))
  
  data = pickle.load(open(fname_unboxed_msweb, "rb")) if dataset_name == 'MSWEB' else pickle.load(open(fname_unboxed_msnbc, "rb"))
  print('raw data loaded')
  full_corpus = data['corpus']
  corpus_len = len(full_corpus)
  num_corpus_selected = int(corpus_frac * corpus_len)
  idx_selected_corpus = np.random.permutation(corpus_len)[:num_corpus_selected]
  corpus = [full_corpus[idx] for idx in idx_selected_corpus]
  return data['elements'], corpus

def fetch_query_ids(unique_corpus, av):
  #unique corpus is a set of sets
    cntQueries = 0
    query_ids = []
    cntCorpus = [0 for _ in range(len(unique_corpus))] #number of sets containing it as subset
    
    if av.DATASET_NAME == 'MSWEB'  and av.SKEW==0:
      a = 8
      b = 560
      fname = os.path.join(processed_datadir_msweb, f"query_ids_first{av.NUM_QUERIES}.pkl")
      #if there's no already existing file, then create it
      if not os.path.exists(fname):
        for i,c1 in tqdm(enumerate(unique_corpus)):
            for c2 in unique_corpus:
                if c1 != c2 and issubset_set(c1, c2): #checks if c1 is a subset of c2
                    cntCorpus[i] += 1
            if cntCorpus[i]  > a and cntCorpus[i]  < b: #if it has > a supersets and < b supersets in the corpus
                cntQueries += 1
                query_ids.append(i)
            if cntQueries >= av.NUM_QUERIES:
                break
        all_d = {"query_ids": query_ids}
        pickle.dump(all_d, open(fname, "wb"))
            
    elif av.DATASET_NAME == 'MSNBC' and av.SKEW==0:
      a = 8
      b = 7000
      fname = os.path.join(processed_datadir_msnbc, f"query_ids_first{av.NUM_QUERIES}.pkl")
      #if there's no already existing file, then create it
      if not os.path.exists(fname):
        for i,c1 in tqdm(enumerate(unique_corpus)):
            for c2 in unique_corpus:
                if c1 != c2 and issubset_multiset(c1, c2): #checks if c1 is a subset of c2 as multisets
                    cntCorpus[i] += 1
            if cntCorpus[i]  > a and cntCorpus[i]  < b: #if it has > a supersets and < b supersets in the corpus
                cntQueries += 1
                query_ids.append(i)
            if cntQueries >= av.NUM_QUERIES:
                break
        all_d = {"query_ids": query_ids}
        pickle.dump(all_d, open(fname, "wb"))
    
    print("query ids generated and dumped")    
    #now load things back 
    all_d = pickle.load(open(fname, "rb"))
    query_ids = all_d['query_ids']
    random.shuffle(query_ids)
    print("Query ids loaded")
    return query_ids #list of indices

def remove_queries_from_corpus(unique_corpus, query_ids):
  queries = [unique_corpus[qid] for qid in query_ids]
  disjoint_corpus = [unique_corpus[cid] for cid in range(len(unique_corpus)) if cid not in query_ids]
  print("queries removed from corpus")
  return queries, disjoint_corpus

def convert_items_to_tensors(items, elements, device, dataset_name: str):
    attid2sent = {e['id']: e['text'] for e in elements}
    sents = [e['text'] for e in elements]
    sent2id = {sent: i for (i,sent) in enumerate(sents)}
    
    model_name ='distilroberta-base'
    max_seq_length = 5 if dataset_name == 'MSWEB' else 10
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length).to(device)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean').to(device)
    sent_model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(device)
    
    s_embeds = torch.tensor(sent_model.encode(sents),device=device)
    """
    num_embeds = s_embeds.shape[0]
    distance_matrix = torch.zeros((num_embeds, num_embeds), device=device)
    for i in range(num_embeds):
        for j in range(num_embeds):
            distance_matrix[i, j] = torch.norm(s_embeds[i] - s_embeds[j]).item()
    print(f"Dist matrix: \n{distance_matrix}\n")
    print(f"Shape of embeddings: ", s_embeds.shape)
    print(f"l2 dist between 9th and 13th: {torch.norm(s_embeds[8]-s_embeds[12]).item()}")
    sys.exit()
    """
    sembed_dim = s_embeds[0].shape[0]
    nmax = max([len(c) for c in items])
    item_tensor = torch.zeros(len(items),nmax,sembed_dim,device=device)
    set_sizes = []
    for i,c in tqdm(enumerate(items)):
        set_sizes.append(len(c))
        j = 0
        for attid in c:
            item_tensor[i,j,:] = s_embeds[sent2id[attid2sent[attid]]]
            j += 1
        for k in range(j,nmax):
            item_tensor[i,j,:] = torch.zeros(sembed_dim,device=device)
    
    mask_ids = torch.arange(item_tensor.size(1)).unsqueeze(0) < torch.FloatTensor(set_sizes).unsqueeze(-1)
    dim_after_inner_net_pass =  sembed_dim
    shape_after_inner_net_pass = (item_tensor.shape[0], nmax, dim_after_inner_net_pass)
    mask_tensor = torch.ones(shape_after_inner_net_pass, device=device)
    mask_tensor[~mask_ids] = 0
    #item_tensor: (num items, max set size, 768)
    #mask tensor: (num items, max set size)
    return item_tensor, np.array(set_sizes), mask_tensor[:, :, 0].squeeze(), nmax
  
def generate_pos_neg_lists(dataset_name: str, corpus, queries):
    """Generates positive and negative pairs and caches them"""
    list_pos = []
    list_neg = []
    subset_func = issubset_set if dataset_name == "MSWEB" else issubset_multiset

    for ind_q in tqdm(range(len(queries))):
        for ind_c, c in enumerate(corpus):
            if subset_func(queries[ind_q], c):
                list_pos.append(((ind_q, ind_c), 1.0))
            else:
                list_neg.append(((ind_q, ind_c), 0.0))

    return list_pos, list_neg #list of paired indices of [query set_i] and [corpus_set_i] for positive and negative
  
def create_dataset_with_p2n(list_pos, list_neg, p2n_ratio, dataset_size):
    """
      Creates shuffled batches while maintaining given ratio
    """
    lpos = list_pos
    lneg = list_neg
    
    random.shuffle(lpos)
    random.shuffle(lneg)

    lpos_pair, lposs = zip(*lpos)
    lposa, lposb = zip(*lpos_pair)

    lneg_pair, lnegs = zip(*lneg)
    lnega, lnegb = zip(*lneg_pair)
    
    npos = math.ceil((p2n_ratio/(1+p2n_ratio))*dataset_size)
    pos_indices = np.random.permutation(len(lpos))[:npos]
    
    nneg = dataset_size - len(pos_indices)
    neg_indices = np.random.permutation(len(lneg))[:nneg]
    
    sampled_posa = [lposa[i] for i in pos_indices]
    sampled_posb = [lposb[i] for i in pos_indices]
    sampled_poss = [lposs[i] for i in pos_indices]
    
    sampled_nega = [lnega[i] for i in neg_indices]
    sampled_negb = [lnegb[i] for i in neg_indices]
    sampled_negs = [lnegs[i] for i in neg_indices]
    
    alists = sampled_posa + sampled_nega #index in query list for S1
    blists = sampled_posb + sampled_negb #index in corpus list for S2
    scores = sampled_poss + sampled_negs #0,1 scores
    
    comb = list(zip(alists, blists, scores))
    random.shuffle(comb)
    alists, blists, scores = zip(*comb)
    alists = torch.tensor(alists, dtype = torch.int)
    blists = torch.tensor(blists, dtype = torch.int)
    scores = torch.tensor(scores, dtype = torch.int)
    
    return alists, blists, scores
  
def create_S1_S2_list(queries, corpus, p2n_ratio: float, dataset_size: int, dataset_name: str):
    list_pos, list_neg = generate_pos_neg_lists(dataset_name, corpus, queries)
    s1_ind_list, s2_ind_list, labels = create_dataset_with_p2n(list_pos, list_neg, p2n_ratio, dataset_size)
    return s1_ind_list, s2_ind_list, labels
  
def data_exists(av):
  #Only for MSWEB or MSNBC, else error
    dataset_name = av.DATASET_NAME
    processed_data_dir = processed_datadir_msnbc if dataset_name == 'MSNBC' else processed_datadir_msweb
    file_names_list = [os.path.join(processed_data_dir, file_name) for file_name in tensor_file_names(av) + ind1_ind2_label_file_names(av)]
    data_exists = True
    for file_name in file_names_list:
      data_exists = data_exists and os.path.exists(file_name)
      if not data_exists: 
        print(f"No data for {av.DATASET_NAME}")
        break
    
    corpus_exists = os.path.exists(os.path.join(processed_data_dir, corpus_file_name)) or os.path.exists(os.path.join(processed_data_dir, corpus_file_name_np))
    data_exists = data_exists and corpus_exists
    return data_exists
    
def return_msweb_dataset(av):
    dataset_name = av.DATASET_NAME
    device = torch.device(f"cuda:{av.DEVICE}") if torch.cuda.is_available() else torch.device("cpu")
    processed_data_dir = processed_datadir_msnbc if dataset_name == 'MSNBC' else processed_datadir_msweb
    print(processed_data_dir)
    query_tensor_mask_files = [os.path.join(processed_data_dir, fname) for fname in tensor_file_names(av)]
    ind_label_files = [os.path.join(processed_data_dir, fname) for fname in ind1_ind2_label_file_names(av)]
    
    if not data_exists(av):
        elements, unique_corpus = load_all_ms_data(dataset_name)
        query_ids = fetch_query_ids(unique_corpus, av) #list of indices, takes in NUM_QUERIES as an argument in av
        query_ids = query_ids[:av.NUM_QUERIES]
        queries, unique_corpus = remove_queries_from_corpus(unique_corpus, query_ids)
        val_queries, test_queries, train_queries = queries[:av.VAL_QUERIES], queries[av.VAL_QUERIES:2 * av.VAL_QUERIES], queries[2 * av.VAL_QUERIES:]
        
        train_query_tensors, train_query_set_sizes, train_query_masks, train_query_nmax = convert_items_to_tensors(train_queries, elements, device, dataset_name)
        print("converted train queries to tensors")
        val_query_tensors, val_query_set_sizes, val_query_masks, val_nmax = convert_items_to_tensors(val_queries, elements, device, dataset_name)
        print("converted val queries to tensors")
        test_query_tensors, test_query_set_sizes, test_query_masks, test_nmax = convert_items_to_tensors(test_queries, elements, device, dataset_name)
        print("converted test queries to tensors")
        common_corpus_tensors, common_corpus_set_sizes, common_corpus_masks, common_corpus_nmax = convert_items_to_tensors(unique_corpus, elements, device, dataset_name)
        print("converted corpus elements to tensors")
        
        s1_ind_list_train, s2_ind_list_train, labels_train = create_S1_S2_list(train_queries, unique_corpus, av.P2N, av.TRAIN_DATASET_SIZE, dataset_name)
        print("S1, S2 list created for train")
        s1_ind_list_val, s2_ind_list_val, labels_val = create_S1_S2_list(val_queries, unique_corpus, av.P2N, av.VAL_DATASET_SIZE, dataset_name)
        print("S1, S2 list created for val")
        s1_ind_list_test, s2_ind_list_test, labels_test = create_S1_S2_list(test_queries, unique_corpus, av.P2N, av.TEST_DATASET_SIZE, dataset_name)
        print("S1, S2 list created for test")
        
        train_query_tensor_mask = {'query_tensors': train_query_tensors, 'query_masks': train_query_masks}
        val_query_tensor_mask = {'query_tensors': val_query_tensors, 'query_masks': val_query_masks}
        test_query_tensor_mask = {'query_tensors': test_query_tensors, 'query_masks': test_query_masks}
        common_corpus_tensor_mask = {'corpus_tensors': common_corpus_tensors, 'corpus_masks': common_corpus_masks}
        
        train_ind_labels = {'s1_ind': s1_ind_list_train, 's2_ind': s2_ind_list_train, 'labels': labels_train}
        val_ind_labels = {'s1_ind': s1_ind_list_val, 's2_ind': s2_ind_list_val, 'labels': labels_val}
        test_ind_labels = {'s1_ind': s1_ind_list_test, 's2_ind': s2_ind_list_test, 'labels': labels_test}
        
        if dataset_name == 'MSNBC':
          train_s1 = [train_queries[idx] for idx in s1_ind_list_train]
          train_s2 = [unique_corpus[idx] for idx in s2_ind_list_train]
          
          val_s1 = [val_queries[idx] for idx in s1_ind_list_val]
          val_s2 = [unique_corpus[idx] for idx in s2_ind_list_val]
          
          test_s1 = [test_queries[idx] for idx in s1_ind_list_test]
          test_s2 = [unique_corpus[idx] for idx in s2_ind_list_test]
          
          train_s1_s2 = {'S1_list': train_s1, 'S2_list': train_s2}
          val_s1_s2 = {'S1_list': val_s1, 'S2_list': val_s2}
          test_s1_s2 = {'S1_list': test_s1, 'S2_list': test_s2}
          train_file_name = os.path.join(processed_datadir_msnbc, f"train_s1s2_size_{av.TRAIN_DATASET_SIZE}_p2n_{av.P2N}.json")
          val_file_name = os.path.join(processed_datadir_msnbc, f"val_s1s2_size_{av.VAL_DATASET_SIZE}_p2n_{av.P2N}.json")
          test_file_name = os.path.join(processed_datadir_msnbc, f"test_s1s2_size_{av.TEST_DATASET_SIZE}_p2n_{av.P2N}.json")
          
          with open(train_file_name, 'w') as file:
            json.dump(train_s1_s2, file)
          with open(val_file_name, 'w') as file:
            json.dump(val_s1_s2, file)
          with open(test_file_name, 'w') as file:
            json.dump(test_s1_s2, file)
        
        torch.save(train_query_tensor_mask, query_tensor_mask_files[0])
        torch.save(train_ind_labels, ind_label_files[0])
        torch.save(val_query_tensor_mask, query_tensor_mask_files[1])
        torch.save(val_ind_labels, ind_label_files[1])
        torch.save(test_query_tensor_mask, query_tensor_mask_files[2])
        torch.save(test_ind_labels, ind_label_files[2])
        print("all train test val stuff saved for queries")
        
        #this shit needed cuz too large space needed to save .pt file. If size available, remove this crap
        if dataset_name == 'MSWEB':
          torch.save(common_corpus_tensor_mask, os.path.join(processed_data_dir, corpus_file_name))
        elif dataset_name == 'MSNBC':
          corpus_tensors = common_corpus_tensor_mask['corpus_tensors']
          corpus_masks = common_corpus_tensor_mask['corpus_masks']
          corpus_tensor_file = os.path.join(processed_data_dir, 'corpus_tensors.pt')
          corpus_masks_file = os.path.join(processed_data_dir, 'corpus_masks.pt')
          torch.save(corpus_tensors, corpus_tensor_file)
          torch.save(corpus_masks, corpus_masks_file)
          #torch.save(common_corpus_tensor_mask, os.path.join(processed_data_dir, corpus_file_name))
          #numpy_dict = {key: value.cpu().numpy() for key, value in common_corpus_tensor_mask.items()}
          #np.savez_compressed(os.path.join(processed_data_dir, corpus_file_name_np), **numpy_dict)
          
        print(f"ALL DATA FOR SET: {dataset_name} SAVED")
        
    else: print(f"DATA FOR SET: {dataset_name} ALREADY EXISTS")
    
    corpus_tensor_and_mask = None
    if dataset_name == 'MSWEB':
        corpus_tensor_and_mask = torch.load(os.path.join(processed_data_dir, corpus_file_name), map_location=device)
    else:
        corpus_tensors = torch.load(os.path.join(processed_data_dir, 'corpus_tensors.pt'), map_location = device)
        corpus_masks = torch.load(os.path.join(processed_data_dir, 'corpus_masks.pt'), map_location = device)
        corpus_tensor_and_mask = {'corpus_tensors': corpus_tensors, 'corpus_masks': corpus_masks}
        #loaded_npz = np.load(os.path.join(processed_data_dir, corpus_file_name_np))
        #corpus_tensor_and_mask = {k: torch.from_numpy(loaded_npz[k]).to(device) for k in loaded_npz.files}
        
    train_query_corpus_ind_labels = torch.load(ind_label_files[0], map_location=device)
    val_query_corpus_ind_labels = torch.load(ind_label_files[1], map_location=device)
    test_query_corpus_ind_labels = torch.load(ind_label_files[2], map_location=device)
    train_query_tensor_and_mask = torch.load(query_tensor_mask_files[0], map_location=device)
    val_query_tensor_and_mask = torch.load(query_tensor_mask_files[1], map_location=device)
    test_query_tensor_and_mask = torch.load(query_tensor_mask_files[2], map_location=device)
    print(f"ALL DATA FOR SET: {dataset_name} LOADED")
    noise_scale = av.noise_scale
    print(f"Noise scale for MSWEB set to: {noise_scale}")
    train_dataset = OurMSwebData(train_query_corpus_ind_labels, train_query_tensor_and_mask, corpus_tensor_and_mask, noise_scale)
    val_dataset = OurMSwebData(val_query_corpus_ind_labels, val_query_tensor_and_mask, corpus_tensor_and_mask, noise_scale)
    test_dataset = OurMSwebData(test_query_corpus_ind_labels, test_query_tensor_and_mask, corpus_tensor_and_mask, noise_scale)
    print(f"Length of train data: {len(train_dataset)}, val data: {len(val_dataset)}, test data: {len(test_dataset)}")
    if av.unittest_dataset:
       unittest(train_dataset, av, 'train')
       unittest(val_dataset, av, 'val')
       unittest(test_dataset, av, 'test')

    return train_dataset, val_dataset, test_dataset
              
class OurMSwebData(Dataset):
  def __init__(self, query_corpus_ind_labels, query_tensor_and_mask, corpus_tensor_and_mask, noise_scale: float = 0.01):
    self.alists, self.blists, self.scores = query_corpus_ind_labels['s1_ind'], query_corpus_ind_labels['s2_ind'], query_corpus_ind_labels['labels']
    self.query_tensors, self.query_mask_tensors = query_tensor_and_mask['query_tensors'], query_tensor_and_mask['query_masks']
    self.corpus_tensors, self.corpus_mask_tensors = corpus_tensor_and_mask['corpus_tensors'], corpus_tensor_and_mask['corpus_masks']
    self.dataset_size = len(self.alists)
    self.noise_scale = noise_scale

  def fetch_batched_data_by_id_optimized(self,i):
    """             
    """
    alist = self.alists[i]
    blist = self.blists[i]
    score = self.scores[i]
    query_tensors = self.query_tensors[alist]
    query_mask_tensors = self.query_mask_tensors[alist]
    corpus_tensors = self.corpus_tensors[blist]
    corpus_mask_tensors = self.corpus_mask_tensors[blist]
    target = torch.tensor(score)
    white_noise = torch.randn_like(query_tensors)*self.noise_scale
    query_tensors = query_tensors + white_noise * query_mask_tensors.unsqueeze(-1)
    return query_tensors, query_mask_tensors, corpus_tensors, corpus_mask_tensors, target
  
  def get_s1s2_idx(self, i):
    s1_idx = self.alists[i]
    s2_idx = self.blists[i]
    return s1_idx, s2_idx

  def __getitem__(self, idx):
    return self.fetch_batched_data_by_id_optimized(idx)

  def __len__(self):
      return self.dataset_size
      
def unittest(dataset, av, type: str):
  dataset_size = {'train': av.TRAIN_DATASET_SIZE, 'val': av.VAL_DATASET_SIZE, 'test': av.TEST_DATASET_SIZE}
  assert len(dataset) == dataset_size[type], f"Given dataset length: {len(dataset)}, supposed length of type: {type} is: {dataset_size[type]}"
  processed_datadir_dataset = processed_datadir_msnbc if av.DATASET_NAME == "MSNBC" else processed_datadir_msweb
  #s1s2_file_name = os.path.join(processed_datadir_dataset, f"{type}_s1s2_size_{dataset_size[type]}_p2n_{av.P2N}.json")
  #s1s2_data = None
  #with open(s1s2_file_name, 'r') as file:
  #  s1s2_data = json.load(file)
  testing_size = dataset.dataset_size//1
  testing_idx = np.random.permutation(dataset.dataset_size)[:testing_size]
  supposed_num_queries = av.VAL_QUERIES if type in ['val', 'test'] else av.NUM_QUERIES - 2 * av.VAL_QUERIES
  assert supposed_num_queries == dataset.query_tensors.shape[0], f"Supposed num queries: {supposed_num_queries}, dataset query tensor shape: {dataset.query_tensors.shape}"
  max_query, max_corpus = dataset.query_tensors.shape[1], dataset.corpus_tensors.shape[1]
  for idx in tqdm(testing_idx):
    s1, mask1, s2, mask2, label = dataset[idx]
    size1 = torch.sum(mask1, dtype = int)
    size2 = torch.sum(mask2, dtype = int)
    zero_threshold = 1e-10
    matched_in_s2 = [False] * size2
    set_contained = True
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
    #s1_idx, s2_idx = dataset.get_s1s2_idx(idx)
    assert int(set_contained) == label, f"idx : {idx}, label: {label}, reality: {int(set_contained)}"
  
  print(f"Label correction tests passed for {type} dataset")
    
  batch_sizes = [16, 32, 64, 128, 256]
  for batch_size in tqdm(batch_sizes):
    dl = DataLoader(dataset, batch_size=batch_size)
    dataset_size = len(dataset)
    num_batches = math.ceil(dataset_size/batch_size)
    for id, batch in enumerate(dl):
      s1, mask1, s2, mask2, label = batch
      if id < num_batches - 1:
        assert s1.shape == torch.Size([batch_size, max_query, 768]), f"Actual shape: {s1.shape}, batch_size: {batch_size}"
        assert mask1.shape == torch.Size([batch_size, max_query]), f"Actual shape: {mask1.shape}, batch_size: {batch_size}"
        assert s2.shape == torch.Size([batch_size, max_corpus, 768]), f"Actual shape: {s2.shape}, batch_size: {batch_size}"
        assert mask2.shape == torch.Size([batch_size, max_corpus]), f"Actual shape: {mask2.shape}, batch_size: {batch_size}"
        assert label.shape == torch.Size([batch_size]), f"Actual shape: {label.shape}, batch_size: {batch_size}"
      
      else:
        assert s1.shape[0] == s2.shape[0] == mask1.shape[0] == mask2.shape[0] == label.shape[0]
        #print(f"Bsize: {batch_size}, last batch size: {s1.shape[0]}")
      
      #if batch_size == 16 and id == 0 : print(f"Sample query mask: {mask1}")
    
  print(f"All tests passed for {type} dataset!")

def create_parser():
  ap = argparse.ArgumentParser()
  ap.add_argument("--SKEW",                           type=int,   default=0)
  ap.add_argument("--VAL_QUERIES",                    type=int,   default=100)
  ap.add_argument("--NUM_QUERIES",                    type=int,   default=500)
  ap.add_argument("--TRAIN_DATASET_SIZE",             type=int,   default = 25000)
  ap.add_argument("--VAL_DATASET_SIZE",               type=int,   default = 10000)
  ap.add_argument("--TEST_DATASET_SIZE",              type=int,   default = 10000)
  ap.add_argument("--P2N",                            type=float,   default=1.0)
  ap.add_argument("--DATASET_NAME",                   type=str,   default="MSWEB", help="MSWEB/MSNBC")
  ap.add_argument("--DEVICE",                         type=int,   default=0)
  ap.add_argument("--frac_corpus",                    type=float, default=1.0)
  ap.add_argument("--noise_scale",                    type=float, default= 0.1)
  ap.add_argument("--unittest_dataset",               type=bool,  default = True)

  av = ap.parse_args()
  return av

if __name__ == '__main__':
  makedirs()
  av = create_parser()
  train_data, val_data, test_data = return_msweb_dataset(av)
  
  
  