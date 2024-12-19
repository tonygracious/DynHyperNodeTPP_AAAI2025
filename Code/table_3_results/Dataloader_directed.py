from datetime import datetime
import numpy as np
import pandas as pd
import math
import os 
def size_distribution(edge_size):
    size_count = np.unique(edge_size, return_counts=True)
    max_size = size_count[0].max()
    p = [size_count[1][size_count[0] == i][0] if i in size_count[0] else 0 for i in range(max_size + 1)]
    p[0] = 0
    normalization = sum(p)
    return [i/normalization for i in p ]

class HyperBiEdgeDataset:
    """
    dataloader for directed hyperedges
    """
    def __init__(self, file_right_node=None, file_left_node=None, file_times=None, time_start=None):
        edge_right_nodes = open(file_right_node, mode='r').readlines()
        edge_left_nodes = open(file_left_node, mode='r').readlines()
        times = open(file_times, mode='r').readlines()

        edge_right_nodes = [i.strip().split(':') for i in edge_right_nodes]
        edge_left_nodes = [i.strip().split(':') for i in edge_left_nodes]

        self.time_start = datetime.strptime(time_start, "%Y-%m-%d")
        times = [(datetime.strptime(i.strip().split('\t')[1], "%Y-%m-%d %H:%M:%S") - self.time_start).days for i in
                 times]

        edge_right_nodes = [(int(i[0]), [int(j) for j in i[1].split(',')]) for i in edge_right_nodes]
        edge_left_nodes = [(int(i[0]), [int(j) for j in i[1].split(',')]) for i in edge_left_nodes]

        assert len(edge_right_nodes) == len(edge_left_nodes)
        self.n_edges = len(edge_right_nodes)

        self.right_nodes = np.sort(list(set([node for i in range(self.n_edges) for node in edge_right_nodes[i][1]])))
        self.left_nodes = np.sort(list(set([node for i in range(self.n_edges) for node in edge_left_nodes[i][1]])))

        self.p_right = size_distribution([len(i[1]) for i in edge_right_nodes])
        self.p_left = size_distribution([len(i[1]) for i in edge_left_nodes])

        self.n_right_nodes = len(self.right_nodes)
        self.n_left_nodes = len(self.left_nodes)
        self.n_nodes = self.n_right_nodes + self.n_left_nodes

        assert self.n_right_nodes == self.right_nodes[-1] + 1
        assert self.n_left_nodes == self.left_nodes[-1] + 1

        self.hyperedges = [
            ((edge_right_nodes[i][1], [node + self.n_right_nodes for node in edge_left_nodes[i][1]]), times[i]) for i in
            range(self.n_edges)]
        self.left_nodes = self.left_nodes + self.n_right_nodes
        self.time_start = 0

        cts = 0
        prev_time = 0
        events_li = []
        self.num_li = []
        i = 0
        self.time_bar = np.zeros((self.n_nodes, 1)) + self.time_start
        self.h_index_left = np.zeros((2, 100))
        self.h_index_left[1, :] = np.arange(0, 100)
        self.h_index_right = np.zeros((2, 100))
        self.h_index_right[1,:] = np.arange(0, 100)
        while i != len(self.hyperedges):
            arr = []
            cts = self.hyperedges[i][1]
            prev_ele = set()
            while i != len(self.hyperedges) and self.hyperedges[i][1] == cts:
                if str((set(self.hyperedges[i][0][0]), set(self.hyperedges[i][0][1]))) in prev_ele:
                    i += 1
                    continue
                arr.append(self.hyperedges[i][0])
                prev_ele.add(str((set(self.hyperedges[i][0][0]), set(self.hyperedges[i][0][1]))))
                i += 1
            time_bar = self.time_bar.copy()
            h_index_left = self.h_index_left.copy()
            h_index_right = self.h_index_right.copy()
            events_li.append((arr, cts, h_index_right,  h_index_left, time_bar, prev_time))
            self.num_li.append(len(arr))
            self.h_index_update(arr)
            for hyperedge in arr:
                for c, j in enumerate(hyperedge[0]):
                    self.time_bar[j] = cts
                for c, j in enumerate(hyperedge[1]):
                    self.time_bar[j] = cts
            prev_time = cts
            assert time_bar.max() <= prev_time, (time_bar.max(), prev_time)
        self.num_li = np.cumsum(np.array(self.num_li))
        self.all_event_concurrent = events_li

        events_ordered  = []
        for event in events_li:
            arr, cts, h_index_right, h_index_left,  time_bar, prev_time = event
            for j in arr:
                events_ordered.append((j, cts, h_index_right,  h_index_left, time_bar, prev_time))
        self.all_events = events_ordered

    def __getitem__(self, index):
        event = self.all_events[index]
        hyperedges, time_cur,  h_index_right, h_index_left, time_bar, prev_time = event
        assert len(set(hyperedges[0])) == len(hyperedges[0]), "nodes are repeating in left hyperedge"
        assert len(set(hyperedges[1])) == len(hyperedges[1]), "nodes are repeating in right hyperedge"
        assert time_bar.max() <= prev_time, (time_bar.max(), prev_time, "error in preprocessing")
        return hyperedges, time_bar, time_cur, h_index_right, h_index_left,  prev_time

    def h_index_update(self, batch_hyperedge_pos):
        h_edges_left = []
        h_edges_right = []
        indexes_left = []
        indexes_right = []
        for i in range(len(batch_hyperedge_pos)):
            h_edge_right, h_edge_left = batch_hyperedge_pos[i]
            index_left = np.ones_like(h_edge_left) * i
            index_right = np.ones_like(h_edge_right) * i
            h_edges_left.append(h_edge_left)
            indexes_left.append(index_left)
            h_edges_right.append(h_edge_right)
            indexes_right.append(index_right)

        h_edges_left = np.concatenate(h_edges_left, axis=0) + 1
        indexes_left = np.concatenate(indexes_left, axis=0)
        h_edges_right = np.concatenate(h_edges_right, axis=0) + 1
        indexes_right = np.concatenate(indexes_right, axis=0)

        h_index_update_left = np.concatenate([h_edges_left.reshape(1,-1), indexes_left.reshape(1,-1)], axis=0)
        h_index_update_right = np.concatenate([h_edges_right.reshape(1,-1), indexes_right.reshape(1,-1)], axis=0)

        self.h_index_left[1,:] = self.h_index_left[1,:] + len(batch_hyperedge_pos)
        self.h_index_right[1,:] = self.h_index_right[1,:] + len(batch_hyperedge_pos)

        self.h_index_left = self.h_index_left[:, self.h_index_left[1,:]<100]
        self.h_index_right = self.h_index_right[:, self.h_index_right[1,:]<100]

        self.h_index_left = np.concatenate([h_index_update_left, self.h_index_left], axis=1)
        self.h_index_right = np.concatenate([h_index_update_right, self.h_index_right], axis=1)


class HyperDiEdgeDataset():
    """
    dataloader for temporal datasets
    """

    def __init__(self, file, step=None):
        file_right_node = os.path.join(file , 'p_a_list_train.txt')
        file_left_node = os.path.join(file , 'p_k_list_train.txt')
        file_times = os.path.join(file , 'times.txt') 
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
            times = [math.floor(float(i.strip().split()[1]))  for i in times]
        edge_right_nodes = [(int(i[0]), [int(j) for j in i[1].split(',')]) for i in edge_right_nodes]
        edge_left_nodes = [(int(i[0]), [int(j) for j in i[1].split(',')]) for i in edge_left_nodes]
        '''
        else:
            with open(file, 'rb') as fb:
                data_obj = pickle.load(fb)
            data = data_obj['data']
            times = [time for _, _, time in data]
            self.time_start = np.min(times)
            times = [i - self.time_start for i in times]
            edge_right_nodes = [i for _, i, _ in data]
            edge_left_nodes = [i for i, _, _ in data]
        '''
        assert len(edge_right_nodes) == len(edge_left_nodes)
        self.n_edges = len(edge_right_nodes)

        self.right_nodes = np.sort(list(set([node for i in range(self.n_edges) for node in edge_right_nodes[i][1]])))
        self.left_nodes = np.sort(list(set([node for i in range(self.n_edges) for node in edge_left_nodes[i][1]])))

        self.p_right = size_distribution([len(i[1]) for i in edge_right_nodes])
        self.p_left = size_distribution([len(i[1]) for i in edge_left_nodes])

        self.n_right_nodes = len(self.right_nodes)
        self.n_left_nodes = len(self.left_nodes)
        self.n_nodes = max( max(self.right_nodes), max(self.left_nodes) ) + 1#len(set(list(self.right_nodes) + list(self.left_nodes)))
        print("No of isolated nodes", self.n_nodes - len(set(list(self.right_nodes) + list(self.left_nodes))) )
        # assert self.n_right_nodes == self.right_nodes[-1] + 1
        # assert self.n_left_nodes  == self.left_nodes[-1] + 1

        self.hyperedges = [((edge_right_nodes[i][1], edge_left_nodes[i][1]), times[i]) for i in range(self.n_edges)]
        self.hyperedges = sorted(self.hyperedges, key=lambda t: t[1])
        self.time_start = self.hyperedges[0][1]
        if step == None:
            self.step = np.median(
                [(self.hyperedges[i + 1][1] - self.hyperedges[i][1]) for i in range(len(self.hyperedges) - 1)])
        else:
            self.step = step
        if self.step == 0:
            self.step = 1
            
        self.hyperedges = [(edge[0], (edge[1] - self.time_start) / self.step) for edge in self.hyperedges]
        self.time_start = 0
        cts = 0
        prev_time = 0
        events_li = []
        self.num_li = []
        i = 0
        self.time_bar = np.zeros((self.n_nodes, 1)) + self.time_start
        self.h_index_left = np.zeros((2, 128))
        self.h_index_left[1, :] = np.arange(0, 128)
        self.h_index_right = np.zeros((2, 128))
        self.h_index_right[1, :] = np.arange(0, 128)
        while i != len(self.hyperedges):
            arr = []
            cts = self.hyperedges[i][1]
            prev_ele = set()
            while i != len(self.hyperedges) and self.hyperedges[i][1] == cts and len(arr) <=5 :
                if str((set(self.hyperedges[i][0][0]), set(self.hyperedges[i][0][1]))) in prev_ele:
                    i += 1
                    continue
                arr.append(self.hyperedges[i][0])
                prev_ele.add(str((set(self.hyperedges[i][0][0]), set(self.hyperedges[i][0][1]))))
                i += 1
            time_bar = self.time_bar.copy()
            h_index_left = self.h_index_left.copy()
            h_index_right = self.h_index_right.copy()
            events_li.append((arr, cts, h_index_right, h_index_left, time_bar, prev_time))
            self.num_li.append(len(arr))
            self.h_index_update(arr)
            for hyperedge in arr:
                for c, j in enumerate(hyperedge[0]):
                    self.time_bar[j] = cts
                for c, j in enumerate(hyperedge[1]):
                    self.time_bar[j] = cts
            prev_time = cts
            assert time_bar.max() <= prev_time, (time_bar.max(), prev_time)
        self.num_li = np.cumsum(np.array(self.num_li))
        self.all_event_concurrent = events_li

        events_ordered = []
        batch_indexes = []
        batch_ids = []
        i = 0
        for event in events_li:
            arr, cts, h_index_right, h_index_left, time_bar, prev_time = event
            if len(arr) + len(batch_indexes) > 128:
                batch_ids.append(batch_indexes)
                batch_indexes = []
            for j in arr:
                events_ordered.append((j, cts, h_index_right, h_index_left, time_bar, prev_time))
                batch_indexes.append(i)
                i = i + 1
        self.all_events = events_ordered
        self.end_time = self.all_events[-1][1]
        self.batch_ids = batch_ids

    def __getitem__(self, index):
        event = self.all_events[index]
        hyperedges, time_cur,  h_index_right, h_index_left, time_bar, prev_time = event
        assert len(set(hyperedges[0])) == len(hyperedges[0]), "nodes are repeating in left hyperedge"
        assert len(set(hyperedges[1])) == len(hyperedges[1]), "nodes are repeating in right hyperedge"
        assert time_bar.max() <= prev_time, (time_bar.max(), prev_time, "error in preprocessing")
        return hyperedges, time_bar, time_cur, h_index_right, h_index_left,  prev_time

    def h_index_update(self, batch_hyperedge_pos):
        h_edges_left = []
        h_edges_right = []
        indexes_left = []
        indexes_right = []
        for i in range(len(batch_hyperedge_pos)):
            h_edge_right, h_edge_left = batch_hyperedge_pos[i]
            index_left = np.ones_like(h_edge_left) * i
            index_right = np.ones_like(h_edge_right) * i
            h_edges_left.append(h_edge_left)
            indexes_left.append(index_left)
            h_edges_right.append(h_edge_right)
            indexes_right.append(index_right)

        h_edges_left = np.concatenate(h_edges_left, axis=0) + 1
        indexes_left = np.concatenate(indexes_left, axis=0)
        h_edges_right = np.concatenate(h_edges_right, axis=0) + 1
        indexes_right = np.concatenate(indexes_right, axis=0)

        h_index_update_left = np.concatenate([h_edges_left.reshape(1,-1), indexes_left.reshape(1,-1)], axis=0)
        h_index_update_right = np.concatenate([h_edges_right.reshape(1,-1), indexes_right.reshape(1,-1)], axis=0)

        self.h_index_left[1,:] = self.h_index_left[1,:] + len(batch_hyperedge_pos)
        self.h_index_right[1,:] = self.h_index_right[1,:] + len(batch_hyperedge_pos)

        self.h_index_left = self.h_index_left[:, self.h_index_left[1,:]<128]
        self.h_index_right = self.h_index_right[:, self.h_index_right[1,:]<128]

        self.h_index_left = np.concatenate([h_index_update_left, self.h_index_left], axis=1)
        self.h_index_right = np.concatenate([h_index_update_right, self.h_index_right], axis=1)



if __name__ == '__main__':
    path = '../../DynDiHyperGraph/directed_datasets/arxiv_25/'

    file_right_node = path + 'p_a_list_train.txt'
    file_left_node = path + 'p_k_list_train.txt'
    file_times = path + 'times.txt'
    data = HyperDiEdgeDataset(path)
    # print(data.hyperedges)

    print(data.right_nodes.shape, data.right_nodes.max(), data.right_nodes.min())
    print(data.left_nodes.shape, data.left_nodes.max(), data.left_nodes.min())
