import numpy as np
import pandas as pd
import torch
import random

from datetime import datetime
import pickle
def size_distribution(edge_size):
    size_count = np.unique(edge_size, return_counts=True)
    max_size = size_count[0].max()
    p = [size_count[1][size_count[0] == i][0] if i in size_count[0] else 0 for i in range(max_size + 1)]
    p[0] = 0
    normalization = sum(p)
    return [i/normalization for i in p ]
class HyperBiEdgeDataset:
    pass
class HyperDiEdgeDataset:
    """
    dataloader for temporal datasets
    """

    def __init__(self, file, step=None, type=type):
        file_right_node = file + 'p_a_list_train.txt'
        file_left_node = file + 'p_k_list_train.txt'
        file_times = file + 'times.txt'
        edge_right_nodes = open(file_right_node, mode='r').readlines()
        edge_left_nodes = open(file_left_node, mode='r').readlines()
        times = open(file_times, mode='r').readlines()
        edge_right_nodes = [i.strip().split(':') for i in edge_right_nodes]
        edge_left_nodes = [i.strip().split(':') for i in edge_left_nodes]

        if 'arxiv' in file:
            time_start = times[0].strip().split('\t')[1]
            self.time_start = datetime.strptime(time_start,  "%Y-%m-%d %H:%M:%S%z")
            times = [(datetime.strptime(i.strip().split('\t')[1], "%Y-%m-%d %H:%M:%S%z") - self.time_start).total_seconds() for i in
                     times]
        else:
            #time_start = int(times[0].strip().split(':')[1])
            #self.time_start = time_start
            times = [float(i.strip().split('\t')[1])  for i in times]
        edge_right_nodes = [(int(i[0]), [int(j) for j in i[1].split(',')]) for i in edge_right_nodes]
        edge_left_nodes = [(int(i[0]), [int(j) for j in i[1].split(',')]) for i in edge_left_nodes]

        assert len(edge_right_nodes) == len(edge_left_nodes)
        self.n_edges = len(edge_right_nodes)

        self.right_nodes = np.sort(list(set([node for i in range(self.n_edges) for node in edge_right_nodes[i][1]])))
        self.left_nodes = np.sort(list(set([node for i in range(self.n_edges) for node in edge_left_nodes[i][1]])))
        len_right = np.array([len(i[1]) for i in edge_right_nodes])
        len_left = np.array([len(i[1]) for i in edge_left_nodes])
        self.p_right = size_distribution(len_right)
        self.p_left = size_distribution(len_left)
        self.p_undir = size_distribution(len_right + len_left)
        self.n_right_nodes = len(self.right_nodes)
        self.n_left_nodes = len(self.left_nodes)
        self.n_nodes = max( max(self.right_nodes), max(self.left_nodes) ) + 1#len(set(list(self.right_nodes) + list(self.left_nodes)))
        print("No of isolated nodes", self.n_nodes - len(set(list(self.right_nodes) + list(self.left_nodes))) )
        # assert self.n_right_nodes == self.right_nodes[-1] + 1
        # assert self.n_left_nodes  == self.left_nodes[-1] + 1

        self.hyperedges = [((edge_right_nodes[i][1], edge_left_nodes[i][1]), times[i]) for i in range(self.n_edges)]
        self.hyperedges = sorted(self.hyperedges, key=lambda t: t[1])
        self.time_start = self.hyperedges[0][1]
        if step== None:
            self.step = np.median(
                [(self.hyperedges[i + 1][1] - self.hyperedges[i][1]) for i in range(len(self.hyperedges) - 1)])
            if self.step == 0:
                self.step = 1
        else:
            self.step = step
        self.hyperedges = [(edge[0], (edge[1] - self.time_start) / self.step) for edge in self.hyperedges]
        self.time_start = 0
        cts = 0
        prev_time = 0
        events_li = []
        self.num_li = []
        i = 0
        while i != len(self.hyperedges):
            arr = []
            cts = self.hyperedges[i][1]
            prev_ele = set()
            while i != len(self.hyperedges) and self.hyperedges[i][1] == cts and len(arr) <=5 :
                if str((sorted(self.hyperedges[i][0][0]), sorted(self.hyperedges[i][0][1]))) in prev_ele:
                    i += 1
                    continue
                arr.append(self.hyperedges[i][0])
                prev_ele.add(str((sorted(self.hyperedges[i][0][0]), sorted(self.hyperedges[i][0][1]))))
                i += 1
            events_li.append((arr, cts, prev_time))
            self.num_li.append(len(arr))
            prev_time = cts

        self.num_li = np.cumsum(np.array(self.num_li))
        self.all_event_concurrent = events_li

        events_ordered = []
        batch_indexes = []
        batch_ids = []
        last_time = np.zeros(self.n_nodes)
        node_inter_event_t_diff = []
        inter_event_t_diff = []
        i = 0
        if type == 'directed':
            for event in events_li:
                arr, cts,  prev_time = event
                right_connectives_dict = {}
                left_connectives_dict = {}
                cur_nodes=[]
                for right_nodes, left_nodes in arr:
                    cur_nodes += list(right_nodes) + list(left_nodes)
                    for node in right_nodes:
                        if node not in right_connectives_dict.keys():
                            right_connectives_dict[node] = np.zeros(self.n_nodes)
                            left_connectives_dict[node] = np.zeros(self.n_nodes)
                        right_connectives_dict[node][right_nodes] = 1
                        right_connectives_dict[node][node] = 0
                        left_connectives_dict[node][left_nodes] = 1

                right_connectives_list = []
                left_connectives_list = []
                for right_nodes,left_nodes in arr:
                    tmp_right, tmp_left =[], []
                    for node in right_nodes:
                        tmp_right.append(right_connectives_dict[node])
                        tmp_left.append(left_connectives_dict[node])
                    right_connectives_list.append(tmp_right)
                    left_connectives_list.append(tmp_left)

                if len(arr) + len(batch_indexes) > 128:
                    batch_ids.append(batch_indexes)
                    batch_indexes = []
                for idx,j in enumerate(arr):
                    events_ordered.append((j, cts, prev_time,(right_connectives_list[idx], left_connectives_list[idx])))
                    inter_event_t_diff.append(cts-prev_time)
                    batch_indexes.append(i)
                    i = i + 1
                for node in np.unique(cur_nodes):
                    node_inter_event_t_diff.append(cts - last_time[node])
                    last_time[node] = cts
        else:
            for event in events_li:
                arr, cts,  prev_time = event
                connectives_dict = {}
                cur_nodes=[]
                for right_nodes, left_nodes in arr:
                    row = np.unique(list(right_nodes) + list(left_nodes))
                    cur_nodes += list(row)
                    for node in row:
                        if node not in connectives_dict.keys():
                            connectives_dict[node] = np.zeros(self.n_nodes)
                        connectives_dict[node][row] = 1
                        connectives_dict[node][node] = 0
                connectives_list = []
                for right_nodes, left_nodes in arr:
                    row = list(right_nodes) + list(left_nodes)
                    tmp = []
                    for node in row:
                        tmp.append(connectives_dict[node])
                    connectives_list.append(tmp)

                if len(arr) + len(batch_indexes) > 128:
                    batch_ids.append(batch_indexes)
                    batch_indexes = []
                for idx,j in enumerate(arr):
                    events_ordered.append((j, cts, prev_time,connectives_list[idx]))
                    inter_event_t_diff.append(cts-prev_time)
                    batch_indexes.append(i)
                    i = i + 1
                for node in np.unique(cur_nodes):
                    node_inter_event_t_diff.append(cts - last_time[node])
                    last_time[node] = cts

        log_node_inter_event_t_diff = np.log(np.array(node_inter_event_t_diff) + 1e-16)
        log_inter_event_t_diff = np.log(np.array(inter_event_t_diff) + 1e-16)
        self.log_node_inter_event_t_diff_mean, self.log_node_inter_event_t_diff_std = np.mean(log_node_inter_event_t_diff) , np.std(log_node_inter_event_t_diff)
        self.log_inter_event_t_diff_mean,self.log_inter_event_t_diff_std = np.mean(log_inter_event_t_diff) , np.std(log_inter_event_t_diff)
        self.all_events = events_ordered
        self.end_time = self.all_events[-1][1]
        self.batch_ids = batch_ids
        self.neighbors = self.neighborhood(self.all_events, type)  # [[] for i in range(self.n_nodes)]
        self.degree = self.node_degree_calculator(self.all_events)

    def node_degree_calculator(self, all_events):
        degree_right = np.zeros(self.n_nodes)
        #degree_left =  np.zeros(self.n_nodes)
        for event in all_events:
            for i in event[0][0]:
                degree_right[i] += 1
            for i in event[0][1]:
                degree_right[i] += 1
        norm_right = np.sum(degree_right)
        #norm_left = np.sum(degree_left)
        degree_right = degree_right / norm_right
        #degree_left = degree_left / norm_left
        degree = (None, None)
        return degree

    def neighborhood(self, events, type):
        if type == 'undirected':
            neighbors = [[] for i in range(self.n_nodes)]
            for i in range(len(events)):
                hyperedge = events[i][0]
                for hypernode in hyperedge:
                    for node in hypernode:
                        neighbors[node].append((i, events[i][1]))
        else:
            neighbors = ([[] for i in range(self.n_nodes)], [[] for i in range(self.n_nodes)])
            for i in range(len(events)):
                hyperedge = events[i][0]
                for j, hypernode in enumerate(hyperedge):
                    for node in hypernode:
                        neighbors[j][node].append((i, events[i][1]))
        return neighbors


    def __getitem__(self, index):
        event = self.all_events[index]
        hyperedges, time_cur, time_prev, connectives  = event
        assert len(set(hyperedges[0])) == len(hyperedges[0]), "nodes are repeating in left hyperedge"
        assert len(set(hyperedges[1])) == len(hyperedges[1]), "nodes are repeating in right hyperedge"
        assert time_prev <= time_cur, "error in preprocessing"
        return hyperedges, time_cur, time_prev, connectives 



if __name__ == '__main__':
    path = '../../DynDiHyperGraph/directed_datasets/enron/'

    file_right_node = path + 'p_a_list_train.txt'
    file_left_node = path + 'p_k_list_train.txt'
    file_times = path + 'times.txt'
    data = HyperDiEdgeDataset(path, type='directed')
    # print(data.hyperedges)

    print(data.right_nodes.shape, data.right_nodes.max(), data.right_nodes.min())
    print(data.left_nodes.shape, data.left_nodes.max(), data.left_nodes.min())
