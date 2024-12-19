import numpy as np
import math
class NeighborFinder:
  def __init__(self, events, adj_list, uniform=False, seed=None):
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []
    self.events = events
    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[1])
      self.node_to_edge_idxs.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[1] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.
    Returns 3 lists: neighbors, edge_idxs, timestamps
    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, hyperedges, timestamps, n_neighbors=20, PAD_LEN = 10, mode='u'):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.
    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    PAD_LEN: int
    mode: str u for undirected and d for directed
    """
    assert (len(hyperedges) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    batch_hypergraph = []
    batch_mask = []
    batch_times = []
    for index_h, (hyperedge, timestamp) in enumerate(zip(hyperedges, timestamps)):
      mask = np.zeros((1, tmp_n_neighbors * PAD_LEN))
      times = np.zeros((1, tmp_n_neighbors * PAD_LEN))
      hyperedge_index = []
      hyperedge_nodes = []
      index = 0
      for index_n, node in enumerate(hyperedge):
        source_edge_idxs_i, source_edge_times_i = self.find_before(node, timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time
        for hyperede_id in source_edge_idxs_i[-tmp_n_neighbors:]:
          event = self.events[hyperede_id]
          if mode == 'd':
            hyperedge_h = event[0][0] + event[0][1]
          else:
            hyperedge_h = event[0]
          hyperedge_index = hyperedge_index + [index] * len(hyperedge_h)
          hyperedge_nodes = hyperedge_nodes + list([node_h_i+1 for node_h_i in hyperedge_h])
          times[0, index] = event[1]
          index = index + 1
        mask[0, index: (index_n+1) * tmp_n_neighbors] = 1
        hyperedge_index = hyperedge_index + [i for i in range(index, (index_n+1)*tmp_n_neighbors)]
        hyperedge_nodes = hyperedge_nodes + [0 for i in range(index, (index_n+1)*tmp_n_neighbors)]
        index = (index_n+1) * tmp_n_neighbors

      hyperedge_nodes = hyperedge_nodes + [0 for i in range(index, tmp_n_neighbors * PAD_LEN)]
      hyperedge_index = hyperedge_index + [i for i in range(index, tmp_n_neighbors * PAD_LEN)]
      mask[0, index:] = 1
      hypergraph_i = np.stack([hyperedge_nodes, hyperedge_index])
      hypergraph_i[1] = hypergraph_i[1]  +  index_h * (tmp_n_neighbors * PAD_LEN)
      batch_hypergraph.append(hypergraph_i)
      batch_times.append(times)
      batch_mask.append(mask)
    batch_times = np.concatenate(batch_times, axis=0)
    batch_mask = np.concatenate(batch_mask, axis=0)
    return batch_hypergraph, batch_times, batch_mask

  def get_temporal_neighbor_directed(self, hyperedges, timestamps, n_neighbors=20, PAD_LEN = 10):
    """
    Given a list of hyperedges, extracts a sampled temporal neighborhood of each node in each hyperedge.
    Params
    ------
    hyperedges: List[List[int]]
    timestamps: List[float],
    num_neighbors: int
    PAD_LEN: int
    """
    assert (len(hyperedges) == len(timestamps))
    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    batch_hypergraph_right = []
    batch_hypergraph_left = []
    batch_mask = []
    batch_times = []
    for index_h, (hyperedge, timestamp) in enumerate(zip(hyperedges, timestamps)):
      mask = np.zeros((1, tmp_n_neighbors * PAD_LEN))
      times = np.zeros((1, tmp_n_neighbors * PAD_LEN))
      hyperedge_index_right = []
      hyperedge_index_left = []
      hyperedge_nodes_right = []
      hyperedge_nodes_left = []
      index = 0
      for index_n, node in enumerate(hyperedge):
        source_edge_idxs_i, source_edge_times_i = self.find_before(node, timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time
        for hyperede_id in source_edge_idxs_i[-tmp_n_neighbors:]:
          event = self.events[hyperede_id]
          hyperedge_index_right = hyperedge_index_right + [index] * len(event[0][0])
          hyperedge_index_left = hyperedge_index_left + [index] * len(event[0][1])

          hyperedge_nodes_right = hyperedge_nodes_right + [node_h+1 for node_h  in list(event[0][0])]
          hyperedge_nodes_left = hyperedge_nodes_left + [node_h+1 for node_h in list(event[0][1])]

          times[0, index] = event[1]
          index = index + 1
        mask[0, index: (index_n + 1) * tmp_n_neighbors] = 1

        hyperedge_index_right = hyperedge_index_right + [i for i in range(index, (index_n + 1) * tmp_n_neighbors)]
        hyperedge_nodes_right = hyperedge_nodes_right + [0 for i in range(index, (index_n + 1) * tmp_n_neighbors)]

        hyperedge_index_left = hyperedge_index_left + [i for i in range(index, (index_n + 1) * tmp_n_neighbors)]
        hyperedge_nodes_left = hyperedge_nodes_left + [0 for i in range(index, (index_n + 1) * tmp_n_neighbors)]

        index = (index_n + 1) * tmp_n_neighbors

      hyperedge_nodes_right = hyperedge_nodes_right + [0 for i in range(index, tmp_n_neighbors * PAD_LEN)]
      hyperedge_index_right = hyperedge_index_right + [i for i in range(index, tmp_n_neighbors * PAD_LEN)]

      hyperedge_nodes_left = hyperedge_nodes_left + [0 for i in range(index, tmp_n_neighbors * PAD_LEN)]
      hyperedge_index_left = hyperedge_index_left + [i for i in range(index, tmp_n_neighbors * PAD_LEN)]

      mask[0, index:] = 1

      hypergraph_i = np.stack([hyperedge_nodes_right, hyperedge_index_right])
      hypergraph_i[1] = hypergraph_i[1] + index_h * (tmp_n_neighbors * PAD_LEN)
      batch_hypergraph_right.append(hypergraph_i)
      hypergraph_i = np.stack([hyperedge_nodes_left, hyperedge_index_left])
      hypergraph_i[1] = hypergraph_i[1] + index_h * (tmp_n_neighbors * PAD_LEN)
      batch_hypergraph_left.append(hypergraph_i)

      batch_times.append(times)
      batch_mask.append(mask)
    batch_times = np.concatenate(batch_times, axis=0)
    batch_mask = np.concatenate(batch_mask, axis=0)
    return (batch_hypergraph_right, batch_hypergraph_left), batch_times, batch_mask










