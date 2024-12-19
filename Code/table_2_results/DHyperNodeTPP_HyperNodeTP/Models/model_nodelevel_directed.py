from torch import nn
import torch

import numpy as np
import math

from Modules.encoder import HGCNEmbedddingDiContinuous
from Modules.decoder import DirectedClassifier

from Utils.utils import padding_HyperEdge, padding_Connectives, negative_sampling_hyperedge_directed, batching_data
import torch.nn.functional as F
from Modules.message_function import get_message_function
from Modules.message_aggregator import get_message_aggregator
from collections import defaultdict

class NodeDiHyperlink(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, num_of_nodes,data_p, log_inter_event_t_diff_mean, log_inter_event_t_diff_std, arch='sxs', diag_mask=False, device='cpu', factor=[1000, 10], alpha =0.3):
        """
        :param n_head:
        :param d_model:
        :param d_k:
        :param d_v:
        :param num_of_nodes:
        :param arch:
        :param diag_mask:
        :param dyn:
        :param factor:
        """
        super().__init__()
        self.n = num_of_nodes + 1
        self.dim = d_model
        self.device = device
        self.encoder = HGCNEmbedddingDiContinuous(num_of_nodes + 1, d_model, device=device, factor=factor)
        self.decoder = DirectedClassifier(n_head, d_model, d_k, d_v, arch, diag_mask, d_model, softplus_layer=False)
        self.decoder_time_right = nn.Sequential( nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.decoder_time_left =   nn.Sequential( nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.message_aggregator = get_message_aggregator("last", device)
        self.message_function = get_message_function("mlp", d_model * 8, d_model)
        self.log_inter_event_t_diff_mean = log_inter_event_t_diff_mean
        self.log_inter_event_t_diff_std = log_inter_event_t_diff_std
        self.size_dist_right = torch.FloatTensor(data_p[0]).to(device)
        self.min_size_right , self.max_size_right = 1 , len(self.size_dist_right) -1
        self.size_layer_right = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.Tanh(),
            nn.Linear(self.dim , self.max_size_right - self.min_size_right + 1),
            nn.Softmax(dim = -1)
        )
        self.size_dist_left = torch.FloatTensor(data_p[1]).to(device)
        self.min_size_left , self.max_size_left = 1 , len(self.size_dist_left) -1
        self.size_layer_left = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.Tanh(),
            nn.Linear(self.dim , self.max_size_left - self.min_size_left + 1),
            nn.Softmax(dim = -1)
        )
        self.size_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.memory_transform_right= nn.Sequential( nn.Linear(d_model, d_model * 2), nn.LayerNorm(d_model * 2), nn.Tanh(), nn.Linear(d_model * 2, d_model))
        self.memory_transform_left = nn.Sequential( nn.Linear(d_model, d_model * 2), nn.LayerNorm(d_model * 2), nn.Tanh(), nn.Linear(d_model * 2, d_model))
        self.connectivity_prob = nn.Sigmoid()

    def forward(self, memory, batch_hyperedge,  batch_h_index, cur_time,  batch_h_index_times, batch_h_index_mask):
        """
        :param batch_hyperedge: batch size x |e|
        :param time_delta:  batch size x |e|
        :param batch_h_index: batch size 2 x 100
        :return: \lambda(t)
        """
        x = self.encoder(memory, batch_hyperedge,  batch_h_index, cur_time, batch_h_index_times, batch_h_index_mask)
        lbda, embed = self.decoder(batch_hyperedge, x)
        return lbda, embed, x
    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps =  self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        updated_memory, updated_last_update = self.encoder.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)

        return updated_memory, updated_last_update

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        # Update the memory with the aggregated messages
        self.encoder.memory_updater.update_memory(unique_nodes, unique_messages,
                                          timestamps=unique_timestamps)

    def default_value(self):
        return (torch.zeros(self.dim*4 ).to(self.device), torch.tensor(0).to(self.device) )
    def messages_creation(self, batch_hyperedge_pos, time_delta_pos, edge_embeddings, cur_time_pos, agg='mean', side=1):
        """

        :param batch_hyperedge_pos:
        :param time_delta_pos:
        :param edge_embeddings:
        :param cur_time_pos:
        :return:
        """
        batch_hyperedge_pos_stack = batch_hyperedge_pos.view(-1)
        mask = batch_hyperedge_pos_stack != 0
        batch_hyperedge_pos_stack = batch_hyperedge_pos_stack[mask]
        time_delta_pos = time_delta_pos.view(-1).unsqueeze(1)
        time_delta_pos = time_delta_pos[mask, :]
        edge_embeddings_stack = edge_embeddings.view(-1, edge_embeddings.size(2))
        edge_embeddings_stack = edge_embeddings_stack[mask, :]
        time_features = self.encoder.time_embeddings(time_delta_pos.float()).squeeze(dim=1)
        features = torch.cat([edge_embeddings_stack, time_features], dim=1)
        messages = defaultdict(self.default_value)
        cur_time_pos = cur_time_pos.view(-1).repeat_interleave(batch_hyperedge_pos.size(1), dim=0)
        cur_time_pos = cur_time_pos[mask]
        if agg == 'mean':
            labels_value = batch_hyperedge_pos_stack.view(-1)
            labels_value_unique = labels_value.unique(dim=0)
            index = ((labels_value - labels_value_unique.unsqueeze(0).T) == 0).float()
            res_value = index @ features / (index.sum(dim=1, keepdim=True))
            cur_time_pos, _ = torch.max(index * cur_time_pos.view(1, -1), dim=1)
            for i in range(len(labels_value_unique)):
                messages[labels_value_unique[i].item()] =(res_value[i], cur_time_pos[i])
        else:
            for i in range(len(batch_hyperedge_pos_stack)):
                messages[batch_hyperedge_pos_stack[i].item()] = (features[i], cur_time_pos[i])
        return messages

    def hyperlink_neglikelihood(self, hyperedge_neg_sample_padded, hyperedge_pos_sample_padded, batch_right_connectives, \
                                                 cur_time_neg, cur_time_pos, time_prev, prev_time_neg, \
                                                 batch_h_index_neg, batch_h_index_pos, batch_pos_h_index_times, batch_neg_h_index_times, \
                                                batch_pos_h_index_mask, batch_neg_h_index_mask, mode='train'):
        """
        :param batch_hyperedge_neg:
        :param batch_hyperedge_pos:
        :param batch_node_prev_time_neg:
        :param batch_node_prev_time_pos:
        :param cur_time_neg:
        :param cur_time_pos:
        :param batch_h_index:
        :param time_prev:
        :param mode:
        :return:
        """

        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(list(range(self.n)),
                                                      self.encoder.memory.messages)
        n_pos = hyperedge_pos_sample_padded[0].shape[0]

        lbda_neg, _, _ = self.forward(memory, hyperedge_neg_sample_padded, batch_h_index_neg, cur_time_neg.reshape(-1, 1), batch_neg_h_index_times, batch_neg_h_index_mask)
        log_intensity_neg = torch.log(1 - lbda_neg+ 1e-7).view(n_pos, -1).sum(dim=1, keepdim=True)

        lbda_pos, edge_embeddings, node_embeddings_t_cur = self.forward(memory, hyperedge_pos_sample_padded, batch_h_index_pos, cur_time_pos.reshape(-1, 1), batch_pos_h_index_times, batch_pos_h_index_mask)
        log_intensity_pos = torch.log(lbda_pos+ 1e-7)
        loglikelihood = (log_intensity_pos + log_intensity_neg).reshape(-1)

        # Computing the node embeddings at prev time stamp
        node_embeddings_t_prev = self.encoder(memory, hyperedge_pos_sample_padded, batch_h_index_pos, time_prev.reshape(-1, 1), batch_pos_h_index_times, batch_pos_h_index_mask)
        node_embeddings_neg_t_prev = self.encoder(memory, hyperedge_neg_sample_padded,  batch_h_index_neg, prev_time_neg.reshape(-1,1), batch_neg_h_index_times, batch_neg_h_index_mask) 
        right_neg_nodes_mask = hyperedge_neg_sample_padded[0].ne(0)
        time_loss, size_loss, connectivity_loss , time_estimation_mae, time_estimation_scaled_mse,\
        time_delta_truth, time_estimation, size_pred, conn_pred = self.supervision(memory, hyperedge_pos_sample_padded, batch_right_connectives,
                                                                          node_embeddings_t_prev , node_embeddings_t_cur, node_embeddings_neg_t_prev[0], right_neg_nodes_mask, cur_time_pos, time_prev)
        
        lbda_neg = lbda_neg.view(n_pos, -1)
        lbda_neg = torch.cat((lbda_pos, lbda_neg), dim=1)
        mrr = 1 / (torch.where(torch.argsort(-1 * lbda_neg, dim=1) == 0)[1] + 1)

        batch_hyperedge_pos = hyperedge_pos_sample_padded
        time_delta_pos = (cur_time_pos.reshape(-1, 1) - last_update[batch_hyperedge_pos[0]].to(self.device),\
                          cur_time_pos.reshape(-1, 1) - last_update[batch_hyperedge_pos[1]].to(self.device))

        batch_hyperedge_pos_stack = torch.cat([batch_hyperedge_pos[0].view(-1), batch_hyperedge_pos[1].view(-1)], dim=0)
        mask = batch_hyperedge_pos_stack != 0
        batch_hyperedge_pos_stack = batch_hyperedge_pos_stack[mask]
        with torch.no_grad():
            self.update_memory(batch_hyperedge_pos_stack.cpu().numpy(), self.encoder.memory.messages)
            self.encoder.memory.clear_messages(batch_hyperedge_pos_stack.cpu().numpy())

        message_right = self.messages_creation(batch_hyperedge_pos[0], time_delta_pos[0], edge_embeddings[1],
                                               cur_time_pos, agg='last',  side=1)
        message_left = self.messages_creation(batch_hyperedge_pos[1], time_delta_pos[1], edge_embeddings[0],
                                              cur_time_pos, agg='last', side=-1)
        messages = defaultdict(list)
        batch_hyperedge_pos_stack = batch_hyperedge_pos_stack.unique()
        for i in range(len(batch_hyperedge_pos_stack)):
            message_right_i, t_right_i = message_right[batch_hyperedge_pos_stack[i].cpu().item()]
            message_left_i, t_left_i = message_left[batch_hyperedge_pos_stack[i].cpu().item()]
            '''
            if t_right_i.item() == -1 and t_left_i.item() >=0:
                message_i = message_left_i
                t_i = t_left_i
            elif t_left_i.item() == -1 and t_right_i.item() >=0:
                message_i = message_right_i
                t_i = t_right_i
            elif (t_left_i.item()>=0) and (t_right_i.item()>=0):
                message_i = (message_left_i + message_right_i)/2
                t_i = max(t_right_i, t_left_i)
            else:
                raise ValueError('Wrong message creation')
            '''
            if t_right_i >= t_left_i:
               #message_i = message_right_i
               t_i = t_right_i
            else:
               #message_i = message_left_i
               t_i = t_left_i
            message_i = torch.cat([message_right_i, message_left_i], dim=-1)
            messages[batch_hyperedge_pos_stack[i].cpu().item()].append((message_i, t_i))

        self.encoder.memory.store_raw_messages(batch_hyperedge_pos_stack.cpu().numpy(), messages)
        labels = torch.zeros_like(lbda_neg)
        labels[:, 0] = 1
        return ((loglikelihood) *(-1), time_loss, size_loss, connectivity_loss),last_update[batch_hyperedge_pos[0]], mrr, time_estimation_mae, time_estimation_scaled_mse, time_delta_truth, time_estimation, lbda_neg.reshape(-1), labels.reshape(-1), size_pred, conn_pred

    def history_stacking(self, batch_hyperedge_history):
        """
        :param batch_hyperedge_history:
        :return: batch_h_index
        """
        batch_hyperedge_history_right, batch_hyperedge_history_left = batch_hyperedge_history
        batch_h_index_right = []
        batch_h_index_left = []
        for i in range(len(batch_hyperedge_history_right)):
            batch_h_index_right.append( torch.LongTensor(batch_hyperedge_history_right[i]).to(self.device) )
            batch_h_index_left.append( torch.LongTensor(batch_hyperedge_history_left[i]).to(self.device) )
        batch_h_index_right = torch.cat(batch_h_index_right, dim=1)
        batch_h_index_left = torch.cat(batch_h_index_left, dim=1)
        batch_h_index = (batch_h_index_right, batch_h_index_left)
        return batch_h_index
    def history_extraction(self, hyperedge, time, tmb_ngbrs, PAD_LEN):
        """

        :param hyperedge:
        :param time:
        :param tmb_ngbrs:
        :param PAD_LEN:
        :return: batch_h_index, batch_h_index_times, batch_h_index_mask
        """
        batch_right_hyperedge_history, batch_right_hyperedge_history_times, batch_right_hyperedge_history_mask = self.tnbrs_right.get_temporal_neighbor_directed(
            hyperedge, time, n_neighbors=tmb_ngbrs, PAD_LEN=PAD_LEN)
        batch_left_hyperedge_history, batch_left_hyperedge_history_times, batch_left_hyperedge_history_mask = self.tnbrs_left.get_temporal_neighbor_directed(
            hyperedge, time, n_neighbors=tmb_ngbrs, PAD_LEN=PAD_LEN)

        batch_h_index_right = self.history_stacking(batch_right_hyperedge_history)
        batch_h_index_left  = self.history_stacking(batch_left_hyperedge_history)
        batch_h_index = (batch_h_index_right, batch_h_index_left)

        batch_h_index_times = (torch.FloatTensor(batch_right_hyperedge_history_times).to(self.device), torch.FloatTensor(batch_left_hyperedge_history_times).to(self.device))
        batch_h_index_mask  = (torch.FloatTensor(batch_right_hyperedge_history_mask).to(self.device), torch.FloatTensor(batch_left_hyperedge_history_mask).to(self.device))

        return batch_h_index, batch_h_index_times, batch_h_index_mask

    def batch_preprocessing(self, hyperedge_pos, hyperedge_neg, cur_time_pos, cur_time_neg, tmb_ngbrs, right_connectives= None , left_connectives = None, is_right = True):
        """

        :param hyperedge_pos: List[ batch_size List[] ]
        :param hyperedge_neg: List[ batch_size * Neg_Per_Edge List [] ]
        :param cur_time_pos: np array  of batch size \times 1
        :param cur_time_neg: np array of batch size * Neg_per_Edge \times 1
        :param tmb_ngbrs: Int number of previous interactions per node
        :return:
            (
               hyperedge_pos_sample_padded: Batch size \times PAD LEN
               hyperedge_neg_sample_padded :Batch size * Neg_per_Edge \times PAD LEN
               batch_pos_h_index_times:  Tuple[ LongTensor[Batch size \times tmb_ngbrs * PADLEN], LongTensor[Batch size \times tmb_ngbrs * PADLEN]  ]
               batch_neg_h_index_times:  Tuple[ LongTensor[Batch size * Neg_per_Edge \times tmb_ngbrs * PADLEN], LongTensor[Batch size * Neg_per_Edge \times tmb_ngbrs * PADLEN]  ]
               batch_h_index_pos: Tuple[ Batch size \times List[ LongTensor[ 2 \times >tmb_ngbrs*PADLEN ], Batch size \times List[ LongTensor[2 \times >tmb_ngbrs*PADLEN   ]  ]
               batch_h_index_neg: Tuple[ Batch size * Neg_per_Edge  \times List[ LongTensor[ 2 \times >tmb_ngbrs*PADLEN ], Batch size * Neg_per_Edge  \times List[ LongTensor[2 \times >tmb_ngbrs*PADLEN   ]  ]
               batch_pos_h_index_mask: Tuple[ LongTensor[Batch size \times tmb_ngbrs * PADLEN], LongTensor[Batch size \times tmb_ngbrs * PADLEN]  ]
               batch_neg_h_index_mask: Tuple[ LongTensor[Batch size * Neg_per_Edge \times tmb_ngbrs * PADLEN], LongTensor[Batch size * Neg_per_Edge \times tmb_ngbrs * PADLEN]  ]
            )
        """
        PAD_LEN = max([len(sample) for sample in (hyperedge_neg + hyperedge_pos)])  # padding to maximum length edge
        mask_neg, hyperedge_neg_sample_padded = padding_HyperEdge(hyperedge_neg, PAD_LEN)
        mask_pos, hyperedge_pos_sample_padded = padding_HyperEdge(hyperedge_pos, PAD_LEN)
        hyperedge_neg_sample_padded = np.array(hyperedge_neg_sample_padded)
        hyperedge_pos_sample_padded = np.array(hyperedge_pos_sample_padded)
        mask_neg = np.array(mask_neg)
        mask_pos = np.array(mask_pos)
        min_size = self.min_size_right if is_right else self.min_size_left
        ## Computing true size of hyperedge 
        true_size = torch.LongTensor(mask_pos.sum(axis = 1) - min_size).to(self.device)

        batch_h_index_pos, batch_pos_h_index_times, batch_pos_h_index_mask = self.history_extraction(hyperedge_pos, cur_time_pos, tmb_ngbrs, PAD_LEN)
        batch_h_index_neg, batch_neg_h_index_times, batch_neg_h_index_mask = self.history_extraction(hyperedge_neg, cur_time_neg, tmb_ngbrs, PAD_LEN)

        hyperedge_neg_sample_padded = torch.LongTensor(hyperedge_neg_sample_padded)
        hyperedge_pos_sample_padded = torch.LongTensor(hyperedge_pos_sample_padded)

        mask_neg = torch.LongTensor(mask_neg)
        mask_pos = torch.LongTensor(mask_pos)

        hyperedge_neg_sample_padded = (hyperedge_neg_sample_padded + 1) * mask_neg
        hyperedge_pos_sample_padded = (hyperedge_pos_sample_padded + 1) * mask_pos

        hyperedge_neg_sample_padded = hyperedge_neg_sample_padded.to(self.device)
        hyperedge_pos_sample_padded = hyperedge_pos_sample_padded.to(self.device)
        right_connectives_padded =right_connectives
        if right_connectives is not None:
            right_connectives_padded = padding_Connectives(right_connectives, PAD_LEN, self.n)
            right_connectives_padded = torch.LongTensor(right_connectives_padded).to(self.device)
        left_connectives_padded = left_connectives
        if left_connectives is not None:
            left_connectives_padded = padding_Connectives(left_connectives, PAD_LEN, self.n)
            left_connectives_padded = torch.LongTensor(left_connectives_padded).to(self.device)

        return  hyperedge_pos_sample_padded,  hyperedge_neg_sample_padded, batch_pos_h_index_times, batch_neg_h_index_times,\
                                    batch_h_index_pos, batch_h_index_neg, \
                                    batch_pos_h_index_mask, batch_neg_h_index_mask, right_connectives_padded, left_connectives_padded, true_size

    def train_batch_continuous_directed(self, batch_pos_hyperedge, batch_time_cur, batch_prev_time, batch_connectives, Neg_per_Edge,  max_nodes, p, degree=(None, None), mode='train',  g_type= 'bipartite'):

        """
        :param batch_pos_hyperedge:
        :param batch_time_cur:
        :param batch_prev_time:
        :param Neg_per_Edge:
        :param max_nodes:
        :param p:
        :param mode:
        :param g_type:
        :return:
        """
        self.Neg_per_Edge = Neg_per_Edge
        tmb_ngbrs = 10 #len(p[0]) + len(p[1])
        hyperedge_pos = batch_pos_hyperedge
        cur_time_pos = np.array(batch_time_cur)
        prev_time_pos = np.array(batch_prev_time)
        hyperedge_neg = negative_sampling_hyperedge_directed(hyperedge_pos, max_nodes, p, degree=degree, Neg_per_Edge=Neg_per_Edge, g_type=g_type)
        cur_time_neg = cur_time_pos.repeat(Neg_per_Edge, axis=0)
        prev_time_neg = prev_time_pos.repeat(Neg_per_Edge, axis=0)

        hyperedge_right_neg = [sample[0] for sample in hyperedge_neg]
        hyperedge_left_neg = [sample[1] for sample in hyperedge_neg]

        hyperedge_right_pos = [sample[0] for sample in hyperedge_pos]
        hyperedge_left_pos = [sample[1] for sample in hyperedge_pos]

        right_right_connectives = [sample[0] for sample in batch_connectives]
        right_left_connectives = [sample[1] for sample in batch_connectives]

        time_prev = torch.Tensor(prev_time_pos).to(self.device)
        prev_time_neg = torch.Tensor(prev_time_neg).to(self.device)
        hyperedge_right_pos_sample_padded, hyperedge_right_neg_sample_padded, batch_right_pos_h_index_times, batch_right_neg_h_index_times, \
        batch_right_h_index_pos, batch_right_h_index_neg, batch_right_pos_h_index_mask, batch_right_neg_h_index_mask,\
        batch_right_right_connectives, batch_right_left_connectives, hyper_right_size = self.batch_preprocessing(hyperedge_right_pos, hyperedge_right_neg,\
                                                                                    cur_time_pos,  cur_time_neg, tmb_ngbrs, right_right_connectives, right_left_connectives, True)
        
        hyperedge_left_pos_sample_padded, hyperedge_left_neg_sample_padded, batch_left_pos_h_index_times, batch_left_neg_h_index_times, \
        batch_left_h_index_pos, batch_left_h_index_neg, batch_left_pos_h_index_mask, batch_left_neg_h_index_mask, \
                                                               _, _, hyper_left_size = self.batch_preprocessing(hyperedge_left_pos, hyperedge_left_neg, 
                                                                                    cur_time_pos, cur_time_neg, tmb_ngbrs, is_right = False)

        cur_time_pos = torch.Tensor(cur_time_pos).to(self.device)
        cur_time_neg = torch.Tensor(cur_time_neg).to(self.device)

        hyperedge_neg_sample_padded = (hyperedge_right_neg_sample_padded, hyperedge_left_neg_sample_padded)
        hyperedge_pos_sample_padded = (hyperedge_right_pos_sample_padded, hyperedge_left_pos_sample_padded)

        batch_neg_h_index_times = (batch_right_neg_h_index_times, batch_left_neg_h_index_times)
        batch_pos_h_index_times = (batch_right_pos_h_index_times, batch_left_pos_h_index_times)

        batch_neg_h_index_mask = (batch_right_neg_h_index_mask, batch_left_neg_h_index_mask)
        batch_pos_h_index_mask = (batch_right_pos_h_index_mask, batch_left_pos_h_index_mask)

        batch_right_connectives = (batch_right_right_connectives, batch_right_left_connectives)

        batch_h_index_neg = (batch_right_h_index_neg, batch_left_h_index_neg)
        batch_h_index_pos = (batch_right_h_index_pos, batch_left_h_index_pos)

        loss, last_update, mrr, mae, scaled_mse, t_true, t_estimate, scores, labels, size_pred, conn_pred = self.hyperlink_neglikelihood(hyperedge_neg_sample_padded, hyperedge_pos_sample_padded, batch_right_connectives, \
                                cur_time_neg, cur_time_pos, time_prev, prev_time_neg,\
                                 batch_h_index_neg, batch_h_index_pos, batch_pos_h_index_times, batch_neg_h_index_times, \
                                batch_pos_h_index_mask, batch_neg_h_index_mask, mode='train')
        
        # Removing prediction for padded node on right side of hyperedge 
        true_size_right = hyper_right_size.repeat_interleave(hyperedge_right_pos_sample_padded.shape[-1])[hyperedge_right_pos_sample_padded.view(-1)!=0]
        true_size_left = hyper_left_size.repeat_interleave(hyperedge_right_pos_sample_padded.shape[-1])[hyperedge_right_pos_sample_padded.view(-1)!=0]

        size_pred_right = size_pred[0].view(-1,size_pred[0].shape[-1])[hyperedge_right_pos_sample_padded.view(-1)!=0]
        size_pred_left = size_pred[1].view(-1,size_pred[1].shape[-1])[hyperedge_right_pos_sample_padded.view(-1)!=0]
        predictions  = []
        for j in range(len(hyperedge_pos)):
            predictions.append((cur_time_pos[j].cpu().item(),
                                len(hyperedge_pos[j][0]), len(hyperedge_pos[j][1]), 
                                mrr[j].cpu().item(), 
                                loss[0][j].cpu().item() + loss[1][j].cpu().item()+loss[2][j].cpu().item()+ loss[3][j].cpu().item(), 
                                mae[j].cpu().item(), scaled_mse[j].cpu().item(), 
                                t_true[j].cpu().item(), 
                                ' '.join(map(str,t_estimate[j].detach().cpu().numpy())),
                                ' '.join(map(str,last_update[j].detach().cpu().numpy())),
                                ' '.join(map(str,conn_pred[0][j].detach().cpu().numpy())),
                                ' '.join(map(str,conn_pred[1][j].detach().cpu().numpy())),
                                ))
            
        return (loss[0].sum(), loss[1].sum(), loss[2].sum(), loss[3].sum()), mrr.sum(), mae.sum(), scaled_mse.sum(), predictions, scores.cpu().tolist(), labels.cpu().tolist(), (true_size_right.cpu().tolist(), true_size_left.cpu().tolist()), (size_pred_right.cpu().numpy(), size_pred_left.cpu().numpy() )

    def testing_continuous_directed(self, iteration, batch_ids, data, Neg_per_Edge, g_type='bipartite'):
        with torch.no_grad():
            self.eval()
            epoch_loss_te = 0
            epoch_hyper_loss_te = 0
            epoch_time_loss_te = 0
            epoch_size_loss_te = 0
            epoch_conn_loss_te = 0
            epoch_mae_loss_te = 0
            epoch_scaled_mse_loss_te = 0
            epoch_mrr_loss_te = 0
            N_test = 0
            predictions_test = []
            auc_scores_test = []
            auc_labels_test = []
            size_right_true_test = []
            size_left_true_test = []
            size_right_pred_test = []
            size_left_pred_test = []
            p_right = data.p_right
            p_left = data.p_left
            p = [p_right, p_left]
            if g_type == 'bipartite':
                max_nodes = (data.n_right_nodes, data.n_nodes)
            else:
                max_nodes = (data.n_nodes, data.n_nodes)

            for i in iteration:
                batch_pos_hyperedge, batch_time_cur, batch_time_prev, batch_connectives = batching_data(batch_ids[i], data)

                loss, mrr, t_mae, t_scaled_mse,predictions_test_batch, auc_scores_batch, auc_labels_batch,size_true_batch, size_pred_batch = self.train_batch_continuous_directed(batch_pos_hyperedge, \
                                                                       batch_time_cur, batch_time_prev, batch_connectives,
                                                                       Neg_per_Edge,
                                                                       max_nodes, p, mode='test',
                                                                       g_type=g_type)

                N_test = N_test + len(batch_pos_hyperedge)
                total_loss = loss[0] + loss[1] + loss[2] +loss[3]
                epoch_loss_te = epoch_loss_te + total_loss.item()
                epoch_hyper_loss_te = epoch_hyper_loss_te + loss[0].item()
                epoch_time_loss_te = epoch_time_loss_te + loss[1].item()
                epoch_size_loss_te = epoch_size_loss_te + loss[2].item()
                epoch_conn_loss_te = epoch_conn_loss_te + loss[3].item()
                epoch_mrr_loss_te = epoch_mrr_loss_te + mrr.item()
                epoch_mae_loss_te = epoch_mae_loss_te + t_mae.item()
                epoch_scaled_mse_loss_te = epoch_scaled_mse_loss_te + t_scaled_mse.item()
                predictions_test = predictions_test + predictions_test_batch
                auc_scores_test = auc_scores_test + auc_scores_batch
                auc_labels_test = auc_labels_test + auc_labels_batch
                size_right_true_test = size_right_true_test + size_true_batch[0]
                size_left_true_test = size_left_true_test + size_true_batch[1]
                size_right_pred_test.append(size_pred_batch[0])
                size_left_pred_test.append(size_pred_batch[1])

        size_true_test = (size_right_true_test, size_left_true_test)
        size_pred_test = (np.squeeze(np.concatenate(size_right_pred_test, axis =0)), np.squeeze(np.concatenate(size_left_pred_test, axis =0)))
        return epoch_loss_te, epoch_hyper_loss_te, epoch_time_loss_te , epoch_size_loss_te, epoch_conn_loss_te, epoch_mae_loss_te, \
        epoch_scaled_mse_loss_te, epoch_mrr_loss_te, N_test, predictions_test, auc_scores_test, auc_labels_test, size_true_test, size_pred_test


    def time_modeling(self, emb, right_neg_emb, mask, right_neg_nodes_mask, delta_t):

        # preparing negative node embeddings for survival function. We have sampled negative nodes in such a way that
        # first half negative samples is formed by replacing right side of positive hyperedge with randomly sampled nodes. 
        # We are choosing such randomly sampled nodes.
        right_neg_node_emb = right_neg_emb.view(emb[0].shape[0],-1,*right_neg_emb[0].shape[-2:])[:,:self.Neg_per_Edge//2,:,:]
        right_neg_nodes_mask = right_neg_nodes_mask.view(emb[0].shape[0],-1,right_neg_nodes_mask.shape[-1])[:,:self.Neg_per_Edge//2,:]
        right_neg_node_emb = right_neg_node_emb.view(right_neg_node_emb.shape[0],-1,right_neg_node_emb.shape[-1])
        right_neg_nodes_mask = right_neg_nodes_mask.view(right_neg_nodes_mask.shape[0],-1)

        # Using a log normal distribution to model time, we are estimating when a hyperedge will occur given the right node of the hyperedge.
        log_delta_t = (torch.log(delta_t+1e-16) - self.log_inter_event_t_diff_mean) / self.log_inter_event_t_diff_std
        meu_pos = self.decoder_time_right(emb[0].float()).squeeze(dim=2)
        meu_neg = self.decoder_time_right(right_neg_node_emb.float()).squeeze(dim=2)
        log_delta_t_neg = torch.repeat_interleave(log_delta_t,meu_neg.shape[1],dim = -1)

        time_mse = (0.5 * torch.pow(meu_pos - log_delta_t, 2) * mask[0]).sum(dim = 1)/ mask[0].sum(dim=1)
        survial = (1e-16 + 1 - (0.5 * (1+torch.erf((log_delta_t_neg - meu_neg)/math.sqrt(2))))).log()
        survial = (survial*right_neg_nodes_mask).sum(dim = 1)/right_neg_nodes_mask.sum(dim=1)

        time_loss = time_mse - survial
        time_estimation_right = torch.exp(meu_pos * self.log_inter_event_t_diff_std + self.log_inter_event_t_diff_mean ) * mask[0]

        time_estimation_mae = (abs(delta_t - time_estimation_right) * mask[0]).sum(dim=-1)/mask[0].sum(dim=1)        
        time_estimation_scaled_mse = ((((delta_t - time_estimation_right) / torch.clamp(delta_t,min=1)) ** 2) * mask[0]).sum(dim = -1)/mask[0].sum(dim=1)
        time_estimation = time_estimation_right
        return time_loss , time_estimation, time_estimation_mae , time_estimation_scaled_mse
    
    def size_modeling(self, node_emb, k_true, mask):
        # Given the right node of a hyperedge, we are predicting the size of the right side and left side of the hyperedge.
        node_emb_right, _ = node_emb
        size_prob = (self.size_layer_right(node_emb_right) ,self.size_layer_left(node_emb_right))
        k_true_onehot = (nn.functional.one_hot(k_true[0],num_classes = self.max_size_right - self.min_size_right + 1).float(),
                         nn.functional.one_hot(k_true[1],num_classes = self.max_size_left - self.min_size_left + 1).float())
        size_loss = -(torch.bmm((size_prob[0]+1e-16).log(), k_true_onehot[0].unsqueeze(dim = -1)).squeeze(dim = -1) * mask[0]).sum(dim = 1)/(mask[0].sum(dim=1))\
                    -(torch.bmm((size_prob[1]+1e-16).log(), k_true_onehot[1].unsqueeze(dim = -1)).squeeze(dim = -1) * mask[0]).sum(dim = 1)/(mask[0].sum(dim=1))
        
        return size_loss, (size_prob[0].detach(), size_prob[1].detach())
    
    def get_connectivity_perf_measurement(self, gen_hedge, true_connectivies, n_neighbors, n_non_neighbors, mask, v_mask):
        # Calculate the average prediction probability for neighbors and non-neighbors and the performance metrics for the right nodes of every hyperedge.
        Conn_Pred_labels = (gen_hedge > 0.5).int()
        true_pos = (true_connectivies * Conn_Pred_labels)
        false_pos = (Conn_Pred_labels - true_pos).sum(axis=1).view(mask.shape) * mask
        false_neg = (true_connectivies - true_pos).sum(axis=1).view(mask.shape) * mask
        true_pos = true_pos.sum(axis=1).view(mask.shape) * mask

        pred_pos_prob_sum = (true_connectivies * gen_hedge).sum(axis = -1) / (torch.clamp(n_neighbors,min = 1))
        pred_pos_prob_sum = pred_pos_prob_sum.view(mask.shape) * mask

        pred_neg_prob_sum = ((1-true_connectivies) * gen_hedge * v_mask).sum(axis = -1)/n_non_neighbors
        pred_neg_prob_sum = pred_neg_prob_sum.view(mask.shape) * mask

        return true_pos, false_pos, false_neg, pred_pos_prob_sum, pred_neg_prob_sum

    def connectivity_modeling(self, node_emb, memory, pos_hyper, mask, right_connectives):
        # Modeling left connectivity and right connectivity for right nodes of every hyperedge.

        node_emb_right, _ = node_emb
        right_right_connectives, right_left_connectives = right_connectives
        v_right , _ = pos_hyper
        v_right = v_right.view(-1)
        right_right_connectives = right_right_connectives.view(-1, right_right_connectives.shape[-1])
        right_left_connectives = right_left_connectives.view(-1, right_left_connectives.shape[-1])
        node_emb_right = node_emb_right.view(-1,self.dim)

        # Computing right connectivity probability conditioned on right node of hyperedges
        gen_hedge_right = self.connectivity_prob(torch.matmul(node_emb_right, self.memory_transform_right(memory).T))
        # Computing left connectivity probability conditioned on right node of hyperedges
        gen_hedge_left = self.connectivity_prob(torch.matmul(node_emb_right, self.memory_transform_left(memory).T))
        
        # v_mask to mask the right nodes while computing loss for right connectivity of right nodes
        v_mask = 1-nn.functional.one_hot(v_right,num_classes = self.n)
        # zero_mask to mask the padded nodes.
        zero_mask = torch.ones_like(v_mask).to(v_mask.device)
        zero_mask[:,0] = 0
        
        # Computing Normalized positive likelihood
        n_right_neighbors = right_right_connectives.sum(axis = -1) 
        ## right_connectives_masking is masking those those nodes which don't have any right neighbors. We are avoiding such nodes to model connectivity.
        right_connectives_masking = n_right_neighbors.ne(0).int()
        n_left_neighbors = right_left_connectives.sum(axis = -1) 
        positive_likelihood = (right_right_connectives * (gen_hedge_right+1e-16).log()).sum(axis = -1) * right_connectives_masking / (torch.clamp(n_right_neighbors,min = 1)) + \
                              (right_left_connectives * (gen_hedge_left+1e-16).log()).sum(axis = -1) / (torch.clamp(n_left_neighbors,min = 1))     #Clamp avoid 0 in denominator
        
        # Computing  Normalized Negative likelihood 
        n_right_non_neighbors = self.n - 2 - n_right_neighbors # removing neighbors, a padded node and the conditioned node
        n_left_non_neighbors = self.n - 1 - n_left_neighbors  # removing neighbors and a padded node 
        negative_likelihood = ((1-right_right_connectives) *(1- gen_hedge_right +1e-16).log() * v_mask * zero_mask).sum(axis = -1) * right_connectives_masking / (torch.clamp(n_right_non_neighbors,min = 1)) + \
                              ((1-right_left_connectives) * (1- gen_hedge_left +1e-16).log() * zero_mask).sum(axis = -1) / (torch.clamp(n_left_non_neighbors,min = 1))     #Clamp avoid 0 in denominator
        
        # self.alpha is used in computing weightage average of positive_likelihood and negative_likelihood. 
        # Since the number of non-neighbors is very large compared to neighbors, hence we are giving more weightage to negative_likelihood. 
        connectivity_loss = -(self.alpha * positive_likelihood + (1 - self.alpha) * negative_likelihood)
        connectivity_loss = connectivity_loss.view(mask[0].shape) * mask[0]
        connectivity_loss = connectivity_loss.sum(dim = 1) / (mask[0].sum(dim = 1))
        gen_hedge_right = gen_hedge_right * v_mask * zero_mask # masking input and padded nodes from predicted connectivity
        gen_hedge_left = gen_hedge_left * zero_mask # masking padded nodes

        return connectivity_loss, (gen_hedge_right.view(pos_hyper[0].shape[0], -1).detach(), gen_hedge_left.view(pos_hyper[0].shape[0], -1).detach()) 
    
    def supervision(self, memory, pos_hyperedge_padded, right_connectives, node_embeddings_t_prev , node_embeddings_t_cur, right_node_embeddings_neg_t_prev, right_neg_nodes_mask, cur_ts, prev_ts):
        # Supervising the model by predicting the time, size and connectivity of hyperedges conditioned on right nodes of hyperedges.
        mask = (pos_hyperedge_padded[0].ne(0).int() , pos_hyperedge_padded[1].ne(0).int())
        delta_t_true  = (cur_ts - prev_ts).reshape(-1,1)
        assert (delta_t_true>=0.0).all()
        time_loss, time_estimation, time_estimation_mae, time_estimation_scaled_mse = self.time_modeling(
            node_embeddings_t_prev, right_node_embeddings_neg_t_prev, mask, right_neg_nodes_mask, delta_t_true)

        k_true = (mask[0].sum(axis = 1) - self.min_size_right , mask[1].sum(axis = 1) - self.min_size_left)
        size_loss, size_pred = self.size_modeling(node_embeddings_t_cur, k_true, mask)

        connectivity_loss, conn_pred = self.connectivity_modeling(node_embeddings_t_cur, memory, pos_hyperedge_padded, mask, right_connectives)
        return time_loss, size_loss, connectivity_loss, time_estimation_mae, time_estimation_scaled_mse, delta_t_true, time_estimation, size_pred, conn_pred