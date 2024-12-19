import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.init import xavier_normal_

def padding(hyperedges ,max , pad_idx = 0):
    features = np.full(shape=(len(hyperedges), max) , fill_value=pad_idx , dtype = int)
    for i,hyperedge in enumerate(hyperedges):
        features[i,:len(hyperedge)] = np.sort(np.array(hyperedge)) + 1 # Addding 1 to all the node idx because 0 is reserved for pad_idx
    return features

def process_hyperedges(hyperedges, model_name = 'hsgnn'):
    """ 
     if model_name is hsgnn then return undirected edges.
     else return left and right hyperedge.
    """
    if model_name == 'hsgnn':
        undirected_hyperedges = []
        max_len = 0
        for hyperedge in hyperedges:
            left_hyperedge, right_hyperedge = hyperedge[0][0] , hyperedge[0][1]
            tmp_hyperedge = left_hyperedge + right_hyperedge
            undirected_hyperedges.append(tmp_hyperedge)
            max_len = max(len(tmp_hyperedge) , max_len)
        return undirected_hyperedges , max_len
    else:
        left_undirected_hyperedges , right_undirected_hyperedges = [],[]
        left_max_len , right_max_len =0,0
        for hyperedge in hyperedges:
            left_hyperedge, right_hyperedge = hyperedge[0][0] , hyperedge[0][1]
            left_undirected_hyperedges.append(left_hyperedge)
            right_undirected_hyperedges.append(right_hyperedge)
            left_max_len = max(len(left_hyperedge), left_max_len)
            right_max_len = max(len(right_hyperedge) , right_max_len)
        return left_undirected_hyperedges , right_undirected_hyperedges, left_max_len , right_max_len

    
def gethyperedges(data):
    hyperedges = []
    i = 0
    for i in range(len(data)):
        idx, node, left = data[i]
        if idx >= len(hyperedges):
            hyperedges.append((([], []), idx))
        if left == -1:
            hyperedges[idx][0][0].append(node)
        else:
            hyperedges[idx][0][1].append(node)
    return hyperedges

def processStaticData(path,split):
    p = os.path.join(path, 'splits', str(split))
    with open(os.path.join(p, 'indices.pkl'), 'rb') as f:
        D = pickle.load(f)
    n, m = D['n'], D['m']

    train_pos_hyperedges = gethyperedges(D['pos_train'])
    train_neg_hyperedges = gethyperedges(D['neg_train'])
    test_pos_hyperedges =  gethyperedges(D['pos_test'])
    test_neg_hyperedges = gethyperedges(D['neg_test'])

    return n, train_pos_hyperedges , train_neg_hyperedges , test_pos_hyperedges, test_neg_hyperedges