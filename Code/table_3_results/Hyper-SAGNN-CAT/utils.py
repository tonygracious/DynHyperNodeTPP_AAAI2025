import numpy as np
import torch
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from concurrent.futures import as_completed, ProcessPoolExecutor
import os




def add_padding_idx(vec):
	if len(vec.shape) == 1:
		return np.asarray([np.sort(np.asarray(v) + 1).astype('int')
						 for v in tqdm(vec)])
	else:
		vec = np.asarray(vec) + 1
		vec = np.sort(vec, axis=-1)
		return vec.astype('int')


def np2tensor_hyper(vec, dtype):
	vec = np.asarray(vec)
	if len(vec.shape) == 1:
		return [torch.as_tensor(v, dtype=dtype) for v in vec]
	else:
		return torch.as_tensor(vec, dtype = dtype)


def walkpath2str(walk):
	return [list(map(str, w)) for w in tqdm(walk)]


def roc_auc_cuda(y_true, y_pred):
	try:
		y_true = y_true.cpu().detach().numpy().reshape((-1, 1))
		y_pred = y_pred.cpu().detach().numpy().reshape((-1, 1))
		return roc_auc_score(
			y_true, y_pred), average_precision_score(
			y_true, y_pred)
	except BaseException:
		return 0.0, 0.0

def MRR(ranked_list):
    rr = 0
    for rank in ranked_list:
        rr += 1/(rank + 1.0)
    return rr / len(ranked_list)

def getRank(pos_scores, neg_scores , rel_item_pos):
    '''
        pos_scores : N * 1
        neg_scores : N * Num_of_neg_samples
    '''
    pos_scores = pos_scores.reshape(-1,1)
    scores = np.concatenate((pos_scores, neg_scores),axis = 1)
    temp = (-1 * scores).argsort()
    ranks = np.empty_like(temp)
    for i,t in enumerate(temp):
        ranks[i][t] = np.arange(len(t))
    return list(ranks[:,rel_item_pos])

def accuracy(output, target):
	pred = output >= 0.5
	truth = target >= 0.5
	acc = torch.sum(pred.eq(truth))
	acc = float(acc) * 1.0 / (truth.shape[0] * 1.0)
	return acc


def build_hash(data):
	dict1 = set()

	for datum in data:
		# We need sort here to make sure the order is right
		datum.sort()
		dict1.add(tuple(datum))
	del data
	return dict1


def build_hash2(data):
	dict2 = set()
	for datum in tqdm(data):
		for x in datum:
			for y in datum:
				if x != y:
					dict2.add((x, y))
	return dict2


def build_hash3(data):
	dict2 = set()
	for datum in tqdm(data):
		for i in range(3):
			temp = np.copy(datum).astype('int')
			temp[i] = 0
			dict2.add(tuple(temp))

	return dict2


def parallel_build_hash(data, func, args = None, num = None, initial = None):
	import multiprocessing
	cpu_num = multiprocessing.cpu_count()
	data = np.array_split(data, cpu_num * 3)
	dict1 = initial.copy()
	pool = ProcessPoolExecutor(max_workers=cpu_num)
	process_list = []

	if func == 'build_hash':
		func = build_hash
	if func == 'build_hash2':
		func = build_hash2
	if func == 'build_hash3':
		func = build_hash3

	for datum in data:
		process_list.append(pool.submit(func, datum))

	for p in as_completed(process_list):
		a = p.result()
		dict1.update(a)

	pool.shutdown(wait=True)

	return dict1

def negative_sampling_hyperedge_directed(batch_hyperedge, max_nodes, p , Neg_per_Edge, seed=None):
    """
    :param batch_hyperedge: batch pos hyperedge
    :param p: distribution of the number of nodes on left and right side of hyperedges
    :param Neg_per_Edge:
    :return: batch_neg_hyperedge
    """
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)
    p_right = p[0]
    p_left = p[1]
    k_right = [i for i in range(len(p_right))]
    k_left = [i for i in range(len(p_left))]
    k = (k_right, k_left)
    candidate_nodes = set([i for i in range(max_nodes[0])])
    right_nodes = left_nodes = list(candidate_nodes) #[i for i in range(max_nodes[0])]
    
    neg_batch_hyperedge = []
    for (hyperedge, t) in batch_hyperedge:
        for i, hypernode in enumerate(hyperedge):
            for _ in range(Neg_per_Edge//2):
                if i == 0:
                    right_neg_size = np.random.choice(k[0], 1, p=p[0])[0]
                    while True:
                        right_hypernode_neg = list(np.random.choice(right_nodes, right_neg_size, replace=False))
                        if sorted(right_hypernode_neg) != sorted(hypernode):
                            break
                    neg_batch_hyperedge.append(((right_hypernode_neg, hyperedge[1]),t))
                else:
                    left_neg_size = np.random.choice(k[1], 1, p=p[1])[0]
                    while True:
                        left_hyperedge_neg = list(np.random.choice(left_nodes, left_neg_size, replace=False))
                        if sorted(left_hyperedge_neg) != sorted(hypernode):
                            break
                    neg_batch_hyperedge.append(((hyperedge[0], left_hyperedge_neg),t))
    return neg_batch_hyperedge

