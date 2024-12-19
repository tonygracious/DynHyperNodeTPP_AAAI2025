import numpy as np
import pandas as pd
import torch
import random
import os
import itertools

class HyperEdgeDataset:
    """
    dataloader for temporal datasets
    """
    def __init__(self, simplex_size_file=None, simplex_file=None, times_file=None, labels_file=None, step =None, normalize_time=True, node_map=True):
        """

        :param simplex_size_file: filename
        :param simplex_file: filename
        :param times_file: filename
        :param labels_file: filname
        :param step: float
        :param normalize_time: boolean
        :param node_map:  boolean
        """
        if labels_file:
            Labels = pd.read_csv(labels_file, delimiter=' ', header=None)
            self.n_nodes = Labels.shape[0]
        simplex_size = pd.read_csv(simplex_size_file, header=None)
        simplex = pd.read_csv(simplex_file, header=None)
        times = pd.read_csv(times_file, header=None)

        simplex_size = simplex_size.values.reshape(-1)
        simplex = simplex.values.reshape(-1) - 1
        times = times.values.reshape(-1)
        size_count = np.unique(simplex_size, return_counts=True)
        max_size = size_count[0].max()
        p  = [size_count[1][size_count[0]==i][0] if i in size_count[0] else 0 for i in range(max_size+1)]
        p[0] = 0
        p[1] = 0
        self.p  =[i/sum(p) for i in p ]

        if not labels_file:
            self.n_nodes = max(simplex) + 1
        assert simplex_size.sum() == simplex.shape[0], (simplex_size.sum(), simplex.shape[0] )
        assert  times.shape[0] == simplex_size.shape[0], (times.shape[0], simplex_size.shape[0])

        all_edges = []
        simplex_size_cum = simplex_size.cumsum(axis=0).astype(int)

        #self.MAX_HyperEdge_Length = MAX_HyperEdge_Length

        for i in range(simplex_size.shape[0]):
            if i == 0:
                all_edges.append((list(simplex[: simplex_size_cum[0]]), times[i]))
            else:
                all_edges.append((list(simplex[simplex_size_cum[i - 1]: simplex_size_cum[i]]), times[i]))

        all_edges = [e if len(e[0]) > 1 else (e[0] + [self.n_nodes], e[1]) for e in all_edges]
        self.n_nodes = self.n_nodes + 1
        if node_map:
            self.node_id_map = {}
            counter = 0
            for event in all_edges:
                for e in event[0]:
                    if e not in self.node_id_map:
                        self.node_id_map[e] = counter
                        counter = counter + 1

            self.n_nodes = counter

            all_edges = [([self.node_id_map[n] for n in e[0]], e[1]) for e in all_edges]


        all_edges = sorted(all_edges, key=lambda t: t[1])

        self.n_events = len(all_edges)

        if not step:
            self.step = np.median([(all_edges[i + 1][1] - all_edges[i][1]) for i in range(len(all_edges) - 1)])
        else:
            self.step = step

        self.time_start = all_edges[0][1]
        if normalize_time:
            self.all_events = [(edge[0], (edge[1] - self.time_start) / self.step) for edge in all_edges]
            self.time_start = 0
        else:
            self.all_events = all_edges

        cts=0
        prev_time=0
        events_list=[]
        self.num_list=[]
        i=0
        while i!=len(self.all_events):
            arr=[]
            cts=self.all_events[i][1]
            prev_ele=set()
            while i!=len(self.all_events) and self.all_events[i][1]==cts and len(arr)<=5:  # For avoid memory errors
                if str(sorted(self.all_events[i][0])) in prev_ele :
                    i+=1
                    continue
                arr.append(self.all_events[i][0])
                prev_ele.add(str(sorted(self.all_events[i][0])))
                i+=1

            events_list.append((arr,cts, prev_time))
            self.num_list.append(len(arr))
            prev_time=cts
            
        self.num_list=np.cumsum(np.array(self.num_list))
        
        events_list_new=[]
        batch_ids = []
        batch_indexes = []
        last_time = np.zeros(self.n_nodes)
        node_inter_event_t_diff = []
        inter_event_t_diff = []
        i = 0
        for event in events_list:
            arr,cts, prev_time=event
            connectives_dict = {}
            for row in arr:
                for node in row:
                    if node not in connectives_dict.keys():
                        connectives_dict[node] = np.zeros(self.n_nodes)
                    connectives_dict[node][row] = 1
                    connectives_dict[node][node] = 0
            connectives_list = []
            for row in arr:
                tmp = []
                for node in row:
                    tmp.append(connectives_dict[node])
                connectives_list.append(tmp)
                
            if len(arr) + len(batch_indexes) > 128:  #batch size is fixed at 128
                batch_ids.append(batch_indexes)
                batch_indexes = []
            for idx,j in enumerate(arr):
                events_list_new.append((j,cts, prev_time, connectives_list[idx]))
                inter_event_t_diff.append(cts-prev_time)
                batch_indexes.append(i)
                i= i + 1
            for node in np.unique(list(itertools.chain(*arr))):
                node_inter_event_t_diff.append(cts - last_time[node])
                last_time[node] = cts
        log_node_inter_event_t_diff = np.log(np.array(node_inter_event_t_diff) + 1e-16)
        log_inter_event_t_diff = np.log(np.array(inter_event_t_diff) + 1e-16)

        self.log_node_inter_event_t_diff_mean, self.log_node_inter_event_t_diff_std = np.mean(log_node_inter_event_t_diff) , np.std(log_node_inter_event_t_diff)
        self.log_inter_event_t_diff_mean,self.log_inter_event_t_diff_std = np.mean(log_inter_event_t_diff) , np.std(log_inter_event_t_diff)
        self.all_events=events_list_new

        self.end_time = self.all_events[-1][1]
        self.batch_ids  = batch_ids

        self.neighbors = self.neighborhood(self.all_events)#[[] for i in range(self.n_nodes)]


    def neighborhood(self, events):
        neighbors = [[] for i in range(self.n_nodes)]
        for i in range(len(events)):
            for node in events[i][0]:
                neighbors[node].append((i, events[i][1]))
        return neighbors

    def __getitem__(self, index):

        event = self.all_events[index]
        hyperedges, time_cur, prev_time, connectives = event

        #for hyperedge in hyperedges:
        assert len(set(hyperedges))== len(hyperedges), "nodes repeat in hyperedge"
        assert prev_time <= time_cur, (prev_time, time_cur)
        return hyperedges, time_cur, prev_time, connectives


if __name__ == '__main__':
    path = "../../DynHyperGraph/"
    file_names_size_file = ["email-Enron/email-Enron-nverts.txt", "contact-high-school/contact-high-school-nverts.txt", \
                            "threads-math-sx/threads-math-sx-nverts.txt", "congress-bills/congress-bills-nverts.txt", \
                            "email-Eu/email-Eu-nverts.txt", "NDC-classes/NDC-classes-nverts.txt",
                            "NDC-substances/NDC-substances-nverts.txt"]
    file_names_simplex_file = ["email-Enron/email-Enron-simplices.txt",
                               "contact-high-school/contact-high-school-simplices.txt", \
                               "threads-math-sx/threads-math-sx-simplices.txt",
                               "congress-bills/congress-bills-simplices.txt", \
                               "email-Eu/email-Eu-simplices.txt", "NDC-classes/NDC-classes-simplices.txt",
                               "NDC-substances/NDC-substances-simplices.txt"]
    file_names_times_file = ["email-Enron/email-Enron-times.txt", "contact-high-school/contact-high-school-times.txt", \
                             "threads-math-sx/threads-math-sx-times.txt", "congress-bills/congress-bills-times.txt", \
                             "email-Eu/email-Eu-times.txt", "NDC-classes/NDC-classes-times.txt",
                             "NDC-substances/NDC-substances-times.txt"]
    file_names_labels = ["email-Enron/email-Enron-node-labels.txt", None,
                         "threads-math-sx/threads-math-sx-simplex-labels.txt", None, None, None, None]

    file_Id = int(0)
    simplex_size_file = path + file_names_size_file[file_Id]
    simplex_file = path + file_names_simplex_file[file_Id]
    times_file = path + file_names_times_file[file_Id]

    if file_Id in [1, 3, 4, 5, 6]:
        labels_file = None
        step = 1
        if file_Id == 5:
            step = 86400000.0
        if file_Id == 4:
            step = None
        if file_Id == 6:
            step = 86400000.0
    else:
        step = None
        labels_file = path + file_names_labels[file_Id]

    data = HyperEdgeDataset(simplex_size_file, simplex_file, times_file, labels_file, step=step, normalize_time=True)
    print(len(data.all_events))
