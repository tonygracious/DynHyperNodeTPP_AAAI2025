import numpy as np
from sklearn import metrics

def initialize_state(dataset):
    time_bar = np.zeros((dataset.n_nodes, 1)) + dataset.time_start
    return time_bar

def batching_data(batch, dataset):
    batch_edges = []
    batch_time_cur = []
    batch_time_prev = []
    batch_connectives = []
    for _id in batch:
        edge,  time_cur, time_prev, connectives  = dataset.__getitem__(_id)
        batch_edges.append(edge)
        batch_time_cur.append(time_cur)
        batch_time_prev.append(time_prev)
        batch_connectives.append(connectives)
    return batch_edges, batch_time_cur, batch_time_prev, batch_connectives

def negative_sampling_hyperedge(batch_hyperedges, max_nodes, p=np.ones(2)/2, Neg_per_Edge=10, mode='main'):
    """
    :param batch_hyperedges: batch of hyperedge of type List[List[int]] of size Batch size
    :param max_nodes: max number of nodes int
    :param p: probability of sampling different size of hyperedge
    :param Neg_per_Edge: number of negative samples per positive hyperedge
    :param mode: 'main' for hypededge else for pair wise edges
    return: batch_neg_hyperedges of type List[List[int]] of size Batch size * Neg_per_Edge
    """
    # neg_batch_hyperedges=[]
    # for batch_hyperedge in batch_hyperedges:
    k  = [i for i in range(len(p))]

    neg_batch_hyperedge =[]
    for hyperedge in batch_hyperedges:
        hyperedge_pos_size = len(hyperedge)
        #neg_edge=[]
        for _ in range(Neg_per_Edge):
            
            hyperedge_neg_size = np.random.choice(k, 1,  p=p )[0]
            if mode == 'main':
                number_true_nodes = min(hyperedge_neg_size//2, hyperedge_pos_size )
            else:
                number_true_nodes = 1
            nodes_from_true_hyperedge = list(np.random.choice(hyperedge, number_true_nodes, replace=False))
            number_false_nodes = hyperedge_neg_size - number_true_nodes
            for _ in range(number_false_nodes):
                noise = np.random.randint(0, max_nodes, 1)
                while noise in hyperedge or noise in nodes_from_true_hyperedge:
                    noise = np.random.randint(0, max_nodes, 1)
                nodes_from_true_hyperedge.append(noise[0])
            neg_batch_hyperedge.append(nodes_from_true_hyperedge)
        #neg_batch_hyperedge.append(neg_edge)
    #neg_batch_hyperedges.append(neg_batch_hyperedge)

    return neg_batch_hyperedge


def padding_HyperEdge(batch, PAD_LEN):

    mask = []
    padded_hyperedges   = []
    for hyperedge in batch:
        mask.append( [1] * len(hyperedge) + [0] * (PAD_LEN - len(hyperedge)) )
        padded_hyperedges.append( hyperedge + (PAD_LEN - len(hyperedge)) * [0]  )
    mask = np.array(mask)
    padded_hyperedges = np.array(padded_hyperedges)
    return mask, padded_hyperedges


def padding_Connectives(batch,PAD_LEN, NNodes):
    padding_connectives = np.zeros((len(batch), PAD_LEN, NNodes))
    for i, connective in enumerate(batch):
        padding_connectives[i,:len(connective),1:] = connective
    return padding_connectives

def batching_data_directed(batch, dataset):
    batch_edges = []
    batch_time_bar = []
    batch_time_cur = []
    batch_h_index_left = []
    batch_h_index_right = []
    batch_prev_time = []
    for _id in batch:
        edge, time_bar, time_cur, h_index_right, h_index_left, prev_time = dataset.__getitem__(_id)
        batch_edges.append(edge)
        batch_time_bar.append(list(time_bar))
        batch_time_cur.append(time_cur)
        batch_h_index_left.append(h_index_left)
        batch_h_index_right.append(h_index_right)
        batch_prev_time.append(prev_time)
    return batch_edges, batch_time_bar, batch_time_cur, batch_h_index_right, batch_h_index_left, batch_prev_time

def negative_sampling_hyperedge_directed(batch_hyperedge, max_nodes=(1000, 2000), p=[np.ones(2) / 2, np.ones(2) / 2], degree=(None, None),
                                         Neg_per_Edge=10, g_type='directed'):
    """
    :param batch_hyperedge: batch pos hyperedge
    :param p:
    :param Neg_per_Edge:
    :return: batch_neg_hyperedge
    """
    p_right = p[0]
    p_left = p[1]
    k_right = [i for i in range(len(p_right))]
    k_left = [i for i in range(len(p_left))]
    k = (k_right, k_left)
    assert g_type == 'directed', 'directed have different negative sampling'

    candidate_nodes = set([i for i in range(max_nodes[0])])
    right_nodes = left_nodes = list(candidate_nodes) #[i for i in range(max_nodes[0])]


    neg_batch_hyperedge = []
    for hyperedge in batch_hyperedge:
        for i, hypernode in enumerate(hyperedge):
            for _ in range(Neg_per_Edge//2):
                if i == 0:
                    right_neg_size = np.random.choice(k[0], 1, p=p[0])[0]
                    while True:
                        #right_pos_size = min( len(hypernode), right_neg_size//2 )
                        #right_hypernode_pos = list(np.random.choice(hypernode, right_pos_size, replace=False))
                        right_hypernode_neg = list(np.random.choice(right_nodes, right_neg_size, replace=False, p=degree[0]))
                        #right_hypernode_neg = right_hypernode_pos + right_hypernode_neg
                        if sorted(right_hypernode_neg) != sorted(hypernode):
                            break
                    neg_batch_hyperedge.append((right_hypernode_neg, hyperedge[1]))
                else:
                    left_neg_size = np.random.choice(k[1], 1, p=p[1])[0]
                    while True:
                        #left_pos_size = min( len(hypernode), left_neg_size//2 )
                        #left_hypernode_pos = list(np.random.choice(hypernode, left_pos_size, replace=False))
                        left_hyperedge_neg = list(np.random.choice(left_nodes, left_neg_size, replace=False, p=degree[1]))
                        #left_hyperedge_neg = left_hypernode_pos + left_hyperedge_neg
                        if sorted(left_hyperedge_neg) != sorted(hypernode):
                            break
                    neg_batch_hyperedge.append((hyperedge[0], left_hyperedge_neg))
    return neg_batch_hyperedge

def computeAucRocScore(true,pred, multi_class ='ovo', avg = 'macro'):
    labels = np.unique(true)
    pred = pred[:, labels]
    pred = pred/ pred.sum(1).reshape(-1,1)
    return metrics.roc_auc_score( true, pred, multi_class=multi_class, average=avg, labels = labels )

def save_predictions(filepath, predictions, columns):
    with open(filepath, "w") as f:
        f.write(str(columns)+"\n")
        for pred in predictions:
            f.write(str(pred) +"\n")