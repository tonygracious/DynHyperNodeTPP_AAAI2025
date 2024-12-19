from Modules.encoder import HGCNEmbeddingContinuous
from torch import nn
import torch
import torch.nn.functional as F
from Modules.decoder import Classifier
import numpy as np
import math
from Utils.utils import negative_sampling_hyperedge, padding_HyperEdge, padding_Connectives,batching_data, negative_sampling_hyperedge_directed

from Modules.message_aggregator import get_message_aggregator
from Modules.message_function import get_message_function
from collections import defaultdict

class NodeHyperlink(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, num_of_nodes, diag_mask, device, data_p, log_inter_event_t_diff_mean, log_inter_event_t_diff_std, factor=[1000, 10], alpha = 0.3):
        """
        :param n_head: Number of heads Int
        :param d_model: Dimension of the model Int
        :param d_k: Dimension of the key Int
        :param d_v: Dimension of the value Int
        :param num_of_nodes: Number of nodes Int
        :param diag_mask: Boolean for diagonal mask of the hyperege prediction
        :param device: Device to run the model cpu/gpu
        :param factor: Factor for the time embedding
        """
        super().__init__()
        self.device = device
        self.n = num_of_nodes + 1 # +1 for the padding
        self.dim = d_model
        self.encoder = HGCNEmbeddingContinuous(num_of_nodes + 1, d_model, factor=factor, device=device)
        self.decoder = Classifier(n_head, d_model, d_k, d_v, diag_mask=diag_mask, bottle_neck=d_model, softplus_layer=False)
        self.decoder_time = nn.Sequential( nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.message_aggregator = get_message_aggregator("last", device)
        self.message_function = get_message_function("mlp", d_model * 3, d_model)
        self.log_inter_event_t_diff_mean = log_inter_event_t_diff_mean
        self.log_inter_event_t_diff_std = log_inter_event_t_diff_std
        # Size Model
        self.size_dist = torch.FloatTensor(data_p).to(device)
        self.min_size , self.max_size = 2 , len(self.size_dist)
        self.size_layer = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.Tanh(),
            nn.Linear(self.dim , self.max_size - self.min_size + 1),
            nn.Softmax(dim = -1)
        )
        self.size_loss = nn.CrossEntropyLoss()
        # Connectivity Model
        self.alpha = alpha
        self.memory_transform = nn.Sequential(
            nn.Linear(self.dim , self.dim *2),
            nn.LayerNorm(self.dim * 2),
            nn.Tanh(),
            nn.Linear(self.dim*2, self.dim)
        )
        self.connectivity_prob = nn.Sigmoid()

    def forward(self, memory, batch_hyperedge,  batch_h_index, time_delta, batch_h_index_mask):
        """

        :param memory: No of nodes \time dim
        :param batch_hyperedge: batch size \times Pad Len
        :param batch_h_index: batch_size List[ 2 \times batch_size  \times tmp_neighbors  ]
        :param batch_h_index_mask:  batch size \times tmnbrs
        :param time_delta: batch size \times tmnbrs
        :return:
        """
        x = self.encoder(memory, batch_hyperedge, batch_h_index, time_delta, batch_h_index_mask)
        (mu, alpha), edge_embeddings, node_embeddings = self.decoder(batch_hyperedge, x)
        return (mu, alpha), edge_embeddings, node_embeddings, x

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

    def hyperlink_neglikelihood(self, hyperedge_neg_sample_padded, hyperedge_pos_sample_padded, \
                                                 cur_time_neg, cur_time_pos, time_prev, \
                                                 batch_h_index_neg, batch_h_index_pos, batch_pos_h_index_times, batch_neg_h_index_times, \
                                                batch_pos_h_index_mask, batch_neg_h_index_mask, connectives_padded, mode='train'):
        """
        :param hyperedge_neg_sample_padded: batch_size * Neg_per_Edge \times PAD_LEN
        :param hyperedge_pos_sample_padded: batch_size \times PAD_LEN
        :param cur_time_neg: batch_size * Neg_per_Edge \times 1
        :param cur_time_pos: batch_size \times 1
        :param time_prev: batch_size \times 1
        :param batch_h_index_neg: batch_size List[ 2 \times batch_size * Neg_per_Edge \times tmp_neighbors  ]
        :param batch_h_index_pos: batch_size List[ 2 \times batch_size  \times tmp_neighbors  ]
        :param batch_pos_h_index_times: batch_size \times tmp_neighbors
        :param batch_neg_h_index_times: batch_size * Neg_per_Edge \times tmp_neighbors
        :param mode: train/test
        :return: 
        """
        memory, last_update = self.get_updated_memory(list(range(self.n)), self.encoder.memory.messages)
        n_pos = hyperedge_pos_sample_padded.shape[0]
        time_delta_neg = cur_time_neg.reshape(-1, 1) - batch_neg_h_index_times
        #batch_node_prev_time_neg = last_update[hyperedge_neg_sample_padded].to(self.device)
        #time_delta_neg = cur_time_neg.reshape(-1, 1) - batch_node_prev_time_neg
        assert (time_delta_neg >= 0).all()
        (lbda_neg, _), _, _, _ = self.forward(memory, hyperedge_neg_sample_padded, batch_h_index_neg, time_delta_neg, batch_neg_h_index_mask)
        log_intensity_neg = torch.log(1- lbda_neg + 1e-7).view(n_pos, -1).sum(dim=1, keepdim=True)

        time_delta_pos = cur_time_pos.reshape(-1, 1) - batch_pos_h_index_times
        #batch_node_prev_time_pos = last_update[hyperedge_pos_sample_padded].to(self.device)
        #time_delta_pos = cur_time_pos.reshape(-1, 1) - batch_node_prev_time_pos
        assert (time_delta_pos >= 0).all()
        (lbda_pos, _), edge_embeddings, node_embeddings, node_embeddings_t_cur = self.forward(memory, hyperedge_pos_sample_padded,  batch_h_index_pos, time_delta_pos, batch_pos_h_index_mask)
        log_intensity_pos = torch.log(lbda_pos + 1e-7)

        loglikelihood = (log_intensity_pos + log_intensity_neg).reshape(-1)

        time_delta_prev = time_prev.reshape(-1, 1) - batch_pos_h_index_times
        assert (time_delta_prev >= 0).all()

        # Computing the node embeddings at prev time stamp
        node_embeddings_t_prev = self.encoder(memory, hyperedge_pos_sample_padded, batch_h_index_pos, time_delta_prev, batch_pos_h_index_mask)
        time_loss, size_loss, connectivity_loss , time_estimation_mae, time_estimation_scaled_mse,\
                    time_delta_truth, time_estimation, size_pred, conn_pred = self.supervision(memory, hyperedge_pos_sample_padded, node_embeddings_t_prev , node_embeddings_t_cur, connectives_padded, cur_time_pos, time_prev)


        lbda_neg = lbda_neg.view(n_pos, -1)
        lbda_neg = torch.cat((lbda_pos, lbda_neg), dim=1)
        mrr = 1 / (torch.where(torch.argsort(-1 * lbda_neg, dim=1) == 0)[1] + 1)

        batch_hyperedge_pos = hyperedge_pos_sample_padded
        batch_node_prev_time_pos = last_update[hyperedge_pos_sample_padded].to(self.device)
        time_delta_pos = cur_time_pos.reshape(-1, 1) - batch_node_prev_time_pos
        assert (time_delta_pos >= 0).all(), 'time_delta_pos is negative: {}'.format(time_delta_pos.min())
        batch_hyperedge_pos_stack = batch_hyperedge_pos.view(-1)  # Batchsize * PADLEN
        mask = batch_hyperedge_pos_stack != 0
        batch_hyperedge_pos_stack = batch_hyperedge_pos_stack[mask]  # Batchsize * PADLEN
        batch_hyperedge_pos_unique = batch_hyperedge_pos_stack.unique().cpu().numpy()
        time_delta_pos = time_delta_pos.view(-1).unsqueeze(1)  # Batchsize x PADLEN
        time_delta_pos = time_delta_pos[mask, :]
        edge_embeddings = torch.cat([edge_embeddings, node_embeddings], dim=2)
        edge_embeddings_stack = edge_embeddings.view(-1, edge_embeddings.size(2))  # Batchsize * PADLEN x d
        edge_embeddings_stack = edge_embeddings_stack[mask, :]
        with torch.no_grad():
            self.update_memory(batch_hyperedge_pos_stack.cpu().numpy(), self.encoder.memory.messages)
            self.encoder.memory.clear_messages(batch_hyperedge_pos_unique)

        time_features = self.encoder.time_embeddings(time_delta_pos.float()).squeeze(dim=1)  # Batchsize * PADLEN * d
        features = torch.cat([edge_embeddings_stack, time_features], dim=1)  # Batchsize * PADLEN * 2d
        messages = defaultdict(list)
        cur_time_pos = cur_time_pos.view(-1).repeat_interleave(batch_hyperedge_pos.size(1), dim=0)  # Batchsize * PADLEN
        cur_time_pos = cur_time_pos[mask]
        for i in range(len(batch_hyperedge_pos_stack)):
            messages[batch_hyperedge_pos_stack[i].item()].append((features[i], cur_time_pos[i]))
        self.encoder.memory.store_raw_messages(batch_hyperedge_pos_unique, messages)
        labels = torch.zeros_like(lbda_neg)
        labels[:, 0] = 1

        return [(loglikelihood)*(-1), time_loss, size_loss, connectivity_loss], last_update[hyperedge_pos_sample_padded], mrr, time_estimation_mae, time_estimation_scaled_mse, time_delta_truth, time_estimation, lbda_neg.reshape(-1), labels.reshape(-1), size_pred, conn_pred

    def train_batch_continuous(self, batch_pos_hyperedge, batch_time_cur, batch_time_prev, batch_connectives, Neg_per_Edge, data_p, mode='train', neg_sampling_mode='main'):

        """
        :param batch_pos_hyperedge: batch_size \times PAD_LEN
        :param batch_time_cur: batch_size \times 1
        :param batch_time_prev: batch_size \times 1
        :param Neg_per_Edge: number of negative samples per edge
        :param data_p: data_p distribution of hyperedge sizes (used for negative sampling) List[float] of size k_max
        :param mode: train/test
        :param neg_sampling_mode: main/edge (main: sample negative nodes from the  hyperedges, edge: sample negative nodes from the edge)
        :return: Tuple[float, float,List[float,float,float]]: (loss, mrr, predictions)
        """
        loss = 0
        mse_loss_batch = 0
        mrr_loss_batch = 0
        predictions = []
        tmb_ngbrs = 10 # number of neighbors for each node in the hyperedge
        hyperedge_pos = batch_pos_hyperedge # batch_size List[List[int]]
        cur_time_pos =  np.array(batch_time_cur) # batch_size List[float]
        hyperedge_neg = negative_sampling_hyperedge(hyperedge_pos, self.n-1, data_p, Neg_per_Edge=Neg_per_Edge, mode=neg_sampling_mode) # minus 1 is padding node
        cur_time_neg = cur_time_pos.repeat(Neg_per_Edge, axis=0)

        PAD_LEN = max([len(sample) for sample in (hyperedge_neg + hyperedge_pos)])

        mask_neg, hyperedge_neg_sample_padded = padding_HyperEdge(hyperedge_neg, PAD_LEN)
        mask_pos, hyperedge_pos_sample_padded = padding_HyperEdge(hyperedge_pos, PAD_LEN)

        batch_pos_hyperedge_history, batch_pos_hyperedge_history_times, batch_pos_hyperedge_history_mask = self.tnbrs.get_temporal_neighbor(
            hyperedge_pos, cur_time_pos, n_neighbors=tmb_ngbrs, PAD_LEN= PAD_LEN)
        batch_neg_hyperedge_history, batch_neg_hyperedge_history_times, batch_neg_hyperedge_history_mask = self.tnbrs.get_temporal_neighbor(
            hyperedge_neg, cur_time_neg, n_neighbors=tmb_ngbrs, PAD_LEN= PAD_LEN)

        hyperedge_neg_sample_padded = np.array(hyperedge_neg_sample_padded)
        hyperedge_pos_sample_padded = np.array(hyperedge_pos_sample_padded)
        mask_neg = np.array(mask_neg)
        mask_pos = np.array(mask_pos)

        ## Computing true size of hyperedge and then expanding it to cover size for each node in hyperedges
        true_size = mask_pos.sum(axis = 1) - self.min_size
        true_size = np.repeat(true_size,PAD_LEN)[mask_pos.reshape(-1) !=0]

        time_prev = torch.Tensor(np.array(batch_time_prev)).to(self.device)

        hyperedge_neg_sample_padded = torch.LongTensor(hyperedge_neg_sample_padded)
        hyperedge_pos_sample_padded = torch.LongTensor(hyperedge_pos_sample_padded)

        mask_neg = torch.LongTensor(mask_neg)
        mask_pos = torch.LongTensor(mask_pos)

        hyperedge_neg_sample_padded = (hyperedge_neg_sample_padded + 1) * mask_neg #padding node is 0
        hyperedge_pos_sample_padded = (hyperedge_pos_sample_padded + 1) * mask_pos #padding node is 0
        connectives_padded = padding_Connectives(batch_connectives, PAD_LEN, self.n) #padding node is 0
        connectives_padded = torch.LongTensor(connectives_padded)
        nonzero_connectives_node_mask = connectives_padded.sum(axis=-1).sum(axis= -1).ne(0)

        hyperedge_neg_sample_padded = hyperedge_neg_sample_padded.to(self.device)
        hyperedge_pos_sample_padded = hyperedge_pos_sample_padded.to(self.device)
        connectives_padded = connectives_padded.to(self.device)

        batch_h_index_pos = []
        for i in range(len(hyperedge_pos)):
            batch_pos_hyperedge_history[i] = torch.LongTensor(batch_pos_hyperedge_history[i])
            batch_h_index_pos.append(batch_pos_hyperedge_history[i].to(self.device))
        batch_h_index_neg = []
        for i in range(len(hyperedge_neg)):
            batch_neg_hyperedge_history[i] = torch.LongTensor(batch_neg_hyperedge_history[i])
            batch_h_index_neg.append(batch_neg_hyperedge_history[i].to(self.device))
        batch_pos_h_index_times = torch.FloatTensor(batch_pos_hyperedge_history_times).to(self.device)
        batch_neg_h_index_times = torch.FloatTensor(batch_neg_hyperedge_history_times).to(self.device)
        batch_pos_h_index_mask = torch.FloatTensor(batch_pos_hyperedge_history_mask).to(self.device)
        batch_neg_h_index_mask = torch.FloatTensor(batch_neg_hyperedge_history_mask).to(self.device)
        cur_time_pos = torch.FloatTensor(cur_time_pos).to(self.device)
        cur_time_neg = torch.FloatTensor(cur_time_neg).to(self.device)
        loss, last_update, mrr,  mae, scaled_mse, t_true, t_estimate, scores, labels, size_pred, conn_pred = self.hyperlink_neglikelihood(hyperedge_neg_sample_padded, hyperedge_pos_sample_padded,
                                                 cur_time_neg, cur_time_pos, time_prev,
                                                 batch_h_index_neg, batch_h_index_pos, batch_pos_h_index_times, batch_neg_h_index_times, batch_pos_h_index_mask, batch_neg_h_index_mask, connectives_padded, mode='train')
        # Removing prediction for padded node
        size_pred = size_pred.view(-1,size_pred.shape[-1])[mask_pos.view(-1)!=0]
        for j in range(len(hyperedge_pos)):
            predictions.append((cur_time_pos[j].cpu().item(), len(hyperedge_pos[j]), mrr[j].cpu().item(),
                                 loss[0][j].cpu().item() + loss[1][j].cpu().item()+ loss[2][j].cpu().item()+ loss[3][j].cpu().item(), mae[j].cpu().item(),
                                scaled_mse[j].cpu().item(),' '.join(map(str,last_update[j].detach().cpu().numpy())), 
                                ' '.join(map(str,t_true[j].detach().cpu().numpy())),
                                ' '.join(map(str,t_estimate[j].detach().cpu().numpy())),
                                ' '.join(map(str,conn_pred[j].detach().cpu().numpy()))
                                ))        
        return [loss[0].sum(), loss[1].sum(), loss[2].sum(), loss[3].sum()], mrr.sum(), mae.sum(), scaled_mse.sum(), predictions, scores.cpu().tolist(), labels.cpu().tolist(), list(true_size), size_pred.cpu().numpy()
    
    def train_batch_continuous_directed(self, batch_pos_hyperedge, batch_time_cur, batch_time_prev, batch_connectives, Neg_per_Edge, max_nodes, p, degree=(None, None), mode='train', g_type='bipartite'):
        """

        :param batch_pos_hyperedge:
        :param batch_time_cur:
        :param batch_time_prev:
        :param Neg_per_Edge:
        :param max_nodes:
        :param p:
        :param mode:
        :param g_type:
        :return:
        """

        loss = 0
        mse_loss_batch = 0
        mrr_loss_batch = 0
        predictions = []
        tmb_ngbrs = 10 #len(p[0]) + len(p[1]) * 2
        hyperedge_pos = batch_pos_hyperedge
        cur_time_pos = np.array(batch_time_cur)
        hyperedge_neg = negative_sampling_hyperedge_directed(hyperedge_pos, max_nodes, p, degree=degree,
                                                             Neg_per_Edge=Neg_per_Edge, g_type=g_type)
        cur_time_neg = cur_time_pos.repeat(Neg_per_Edge, axis=0)
        hyperedge_neg = [sample[0] + sample[1] for sample in hyperedge_neg]
        hyperedge_pos = [sample[0] + sample[1] for sample in hyperedge_pos]

        PAD_LEN = max([len(sample) for sample in (hyperedge_neg + hyperedge_pos)])

        mask_neg, hyperedge_neg_sample_padded = padding_HyperEdge(hyperedge_neg, PAD_LEN)
        mask_pos, hyperedge_pos_sample_padded = padding_HyperEdge(hyperedge_pos, PAD_LEN)

        batch_pos_hyperedge_history, batch_pos_hyperedge_history_times, batch_pos_hyperedge_history_mask = self.tnbrs.get_temporal_neighbor(
            hyperedge_pos, cur_time_pos, n_neighbors=tmb_ngbrs, mode='d',PAD_LEN= PAD_LEN)
        batch_neg_hyperedge_history, batch_neg_hyperedge_history_times, batch_neg_hyperedge_history_mask = self.tnbrs.get_temporal_neighbor(
            hyperedge_neg, cur_time_neg, n_neighbors=tmb_ngbrs, mode='d',PAD_LEN= PAD_LEN)


        hyperedge_neg_sample_padded = np.array(hyperedge_neg_sample_padded)
        hyperedge_pos_sample_padded = np.array(hyperedge_pos_sample_padded)
        mask_neg = np.array(mask_neg)
        mask_pos = np.array(mask_pos)

        ## Computing true size of hyperedge and then expanding it to cover size for each node in hyperedges
        true_size = mask_pos.sum(axis = 1) - self.min_size
        true_size = np.repeat(true_size,PAD_LEN)[mask_pos.reshape(-1) !=0]

        time_prev = torch.Tensor(np.array(batch_time_prev)).to(self.device)

        hyperedge_neg_sample_padded = torch.LongTensor(hyperedge_neg_sample_padded)
        hyperedge_pos_sample_padded = torch.LongTensor(hyperedge_pos_sample_padded)

        mask_neg = torch.LongTensor(mask_neg)
        mask_pos = torch.LongTensor(mask_pos)

        hyperedge_neg_sample_padded = (hyperedge_neg_sample_padded + 1) * mask_neg
        hyperedge_pos_sample_padded = (hyperedge_pos_sample_padded + 1) * mask_pos
        connectives_padded = padding_Connectives(batch_connectives, PAD_LEN, self.n) #padding node is 0
        connectives_padded = torch.LongTensor(connectives_padded)

        hyperedge_neg_sample_padded = hyperedge_neg_sample_padded.to(self.device)
        hyperedge_pos_sample_padded = hyperedge_pos_sample_padded.to(self.device)
        connectives_padded = connectives_padded.to(self.device)

        batch_h_index_pos = []
        for i in range(len(hyperedge_pos)):
            batch_pos_hyperedge_history[i] = torch.LongTensor(batch_pos_hyperedge_history[i])
            batch_h_index_pos.append(batch_pos_hyperedge_history[i].to(self.device))
        batch_h_index_neg = []
        for i in range(len(hyperedge_neg)):
            batch_neg_hyperedge_history[i] = torch.LongTensor(batch_neg_hyperedge_history[i])
            batch_h_index_neg.append(batch_neg_hyperedge_history[i].to(self.device))
        batch_pos_h_index_times = torch.FloatTensor(batch_pos_hyperedge_history_times).to(self.device)
        batch_neg_h_index_times = torch.FloatTensor(batch_neg_hyperedge_history_times).to(self.device)
        batch_pos_h_index_mask = torch.FloatTensor(batch_pos_hyperedge_history_mask).to(self.device)
        batch_neg_h_index_mask = torch.FloatTensor(batch_neg_hyperedge_history_mask).to(self.device)
        cur_time_pos = torch.FloatTensor(cur_time_pos).to(self.device)
        cur_time_neg = torch.FloatTensor(cur_time_neg).to(self.device)

        loss, last_update, mrr, mae, scaled_mse, t_true, t_estimate, scores, labels, size_pred, conn_pred= self.hyperlink_neglikelihood(hyperedge_neg_sample_padded, hyperedge_pos_sample_padded,
                                                 cur_time_neg, cur_time_pos, time_prev,
                                                 batch_h_index_neg, batch_h_index_pos, batch_pos_h_index_times,
                                                 batch_neg_h_index_times, batch_pos_h_index_mask,
                                                 batch_neg_h_index_mask, connectives_padded, mode='train')
        
        size_pred = size_pred.view(-1,size_pred.shape[-1])[mask_pos.view(-1)!=0]
        predictions = []
        for j in range(len(batch_pos_hyperedge)):
            predictions.append((cur_time_pos[j].cpu().item(),
                                len(batch_pos_hyperedge[j][0]), len(batch_pos_hyperedge[j][1]), 
                                mrr[j].cpu().item(), 
                                loss[0][j].cpu().item() + loss[1][j].cpu().item()+loss[2][j].cpu().item()+ loss[3][j].cpu().item(), 
                                mae[j].cpu().item(), scaled_mse[j].cpu().item(), 
                                t_true[j].cpu().item(), 
                                ' '.join(map(str,t_estimate[j].detach().cpu().numpy())),
                                ' '.join(map(str,last_update[j].detach().cpu().numpy())),
                                ' '.join(map(str,conn_pred[j].detach().cpu().numpy()))
                                ))

        return (loss[0].sum(), loss[1].sum(), loss[2].sum(), loss[3].sum()), mrr.sum(), mae.sum(), scaled_mse.sum(), predictions, scores.cpu().tolist(), labels.cpu().tolist(), list(true_size), size_pred.cpu().numpy()
    
    def testing_continuous(self, iteration, batch_ids, data, Neg_per_Edge):

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
            size_true_test = []
            size_pred_test = []
            for i in iteration:
                batch_pos_hyperedge, batch_time_cur, batch_prev_time, batch_connectives = batching_data(
                    batch_ids[i], data)

                loss, mrr, t_mae, t_scaled_mse, predictions_test_batch, auc_scores_batch, auc_labels_batch, true_size, size_pred = self.train_batch_continuous(batch_pos_hyperedge, \
                                                             batch_time_cur, batch_prev_time, batch_connectives,
                                                             Neg_per_Edge, data.p, mode='test')
                total_loss = loss[0] + loss[1] + loss[2] +loss[3]
                epoch_hyper_loss_te = epoch_hyper_loss_te + loss[0].item()
                epoch_time_loss_te = epoch_time_loss_te + loss[1].item()
                epoch_size_loss_te = epoch_size_loss_te + loss[2].item()
                epoch_conn_loss_te = epoch_conn_loss_te + loss[3].item()

                N_test = N_test + len(batch_pos_hyperedge)
                epoch_loss_te = epoch_loss_te + total_loss.item()
                epoch_mae_loss_te = epoch_mae_loss_te + t_mae.item()
                epoch_scaled_mse_loss_te = epoch_scaled_mse_loss_te + t_scaled_mse.item()
                epoch_mrr_loss_te = epoch_mrr_loss_te + mrr.item()
                predictions_test= predictions_test + predictions_test_batch
                auc_scores_test = auc_scores_test + auc_scores_batch
                auc_labels_test = auc_labels_test + auc_labels_batch
                size_true_test = size_true_test + true_size
                size_pred_test.append(size_pred)
        size_pred_test = np.squeeze(np.concatenate(size_pred_test, axis =0))
        return epoch_loss_te, epoch_hyper_loss_te, epoch_time_loss_te , epoch_size_loss_te, epoch_conn_loss_te , epoch_mae_loss_te, epoch_scaled_mse_loss_te, \
        epoch_mrr_loss_te, N_test, predictions_test, auc_scores_test, auc_labels_test, size_true_test, size_pred_test
    
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
            size_true_test = []
            size_pred_test = []
            p_right = data.p_right
            p_left = data.p_left
            p = [p_right, p_left]
            if g_type == 'bipartite':
                max_nodes = (data.n_right_nodes, data.n_nodes)
            else:
                max_nodes = (data.n_nodes, data.n_nodes)

            for i in iteration:
                batch_pos_hyperedge, batch_time_cur, batch_time_prev, batch_connectives = batching_data(batch_ids[i], data)

                loss, mrr, t_mae, t_scaled_mse, predictions_test_batch, auc_scores_batch, auc_labels_batch, true_size, size_pred = self.train_batch_continuous_directed(batch_pos_hyperedge, batch_time_cur, batch_time_prev, batch_connectives, Neg_per_Edge, max_nodes, p, mode='test', g_type=g_type)
                total_loss = loss[0] + loss[1] + loss[2] +loss[3]
                N_test = N_test + len(batch_pos_hyperedge)
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
                size_true_test = size_true_test + true_size
                size_pred_test.append(size_pred)
        size_pred_test = np.squeeze(np.concatenate(size_pred_test, axis =0))
        return epoch_loss_te, epoch_hyper_loss_te, epoch_time_loss_te , epoch_size_loss_te, epoch_conn_loss_te , epoch_mae_loss_te, epoch_scaled_mse_loss_te, \
        epoch_mrr_loss_te, N_test, predictions_test, auc_scores_test, auc_labels_test, size_true_test, size_pred_test


    def time_modeling(self, emb, mask, delta_t):
        # Using a log normal distribution to model time, we are estimating when a hyperedge will occur given the node of the hyperedge.
        log_delta_t = (torch.log(delta_t+1e-16) - self.log_inter_event_t_diff_mean) / self.log_inter_event_t_diff_std
        mu_t_inv = self.decoder_time(emb.float()).squeeze(dim=2)
        time_loss = torch.pow(mu_t_inv - log_delta_t, 2) * mask
        time_loss = time_loss.sum(dim=1)/(mask.sum(dim=1) )
        time_estimation = torch.exp(mu_t_inv * self.log_inter_event_t_diff_std + self.log_inter_event_t_diff_mean )
        time_estimation_mae = abs(delta_t - time_estimation) * mask
        time_estimation_mae = torch.sum(time_estimation_mae , dim = -1)/ (mask.sum(dim = 1))
        time_estimation_scaled_mse = (((delta_t - time_estimation) / torch.clamp(delta_t,min=1)) ** 2) * mask # Bs * PAD_LEN
        time_estimation_scaled_mse = torch.sum(time_estimation_scaled_mse , dim = -1)/ (mask.sum(dim = 1)) # Bs * 1
        time_estimation *= mask
        return time_loss , time_estimation, time_estimation_mae , time_estimation_scaled_mse

    def size_modeling(self, node_emb, k_true, mask):
        # Given a node of a hyperedge, we are predicting the size of the hyperedge.
        size_prob = self.size_layer(node_emb) # Computing probability
        k_true_onehot = nn.functional.one_hot(k_true,num_classes = self.max_size - self.min_size + 1).float()
        size_loss = -torch.bmm((size_prob+1e-16).log(), k_true_onehot.unsqueeze(dim = -1)).squeeze() * mask  # Bs * PAD_LEN
        size_loss = size_loss.sum(dim=1)/(mask.sum(dim=1) ) # Bs * 1
        return size_loss, size_prob.detach()

    def connectivity_modeling(self, node_emb, memory, pos_hyper, connectives_padded):
        v = pos_hyper.view(-1)
        # Since some nodes have zero connectivity and we are ignoring those nodes for connectivity modeling so we will use connectives_padded_mask instead of 
        # pos_hyper.ne(0).int() to avoid those nodes. For undirected dataset, connectivity mask is equivalent to pos_hyper.ne(0).int()
        connectives_padded_mask = connectives_padded.sum(axis = -1).ne(0).int()
        connectives_padded = connectives_padded.view(-1, connectives_padded.shape[-1]) # Bs * PAD_LEN * NNODES --> Bs.PAD_LEN * NNODES
        node_emb = node_emb.view(-1, self.dim)

        # Computing connectivity probability conditioned on a node of every hyperedge.
        gen_hedge = self.connectivity_prob(torch.matmul(node_emb,self.memory_transform(memory).T))
        # v_mask to mask the conditioned and padded nodes while computing loss for right connectivity of right nodes
        v_mask = 1-nn.functional.one_hot(v,num_classes = self.n)
        v_mask[:,0] = 0

        # Computing Normalized positive likelihood
        n_neighbors = connectives_padded.sum(axis = -1) 
        positive_likelihood = (connectives_padded * (1e-16+ gen_hedge).log()).sum(axis = -1) / (torch.clamp(n_neighbors,min = 1))  #Clamp avoid 0 in denominator
        
        # Computing  Normalized Negative likelihood 
        n_non_neighbors = self.n - 2 - n_neighbors 
        negative_likelihood = ((1-connectives_padded) * (1e-16 + 1-gen_hedge).log() * v_mask).sum(axis = -1)/n_non_neighbors

        # self.alpha is used in computing weightage average of positive_likelihood and negative_likelihood. 
        # Since the number of non-neighbors is very large compared to neighbors, hence we are giving more weightage to negative_likelihood. 
        connectivity_loss = -(self.alpha * positive_likelihood + (1-self.alpha) * negative_likelihood) #  Bs.PAD_LEN * 1
        connectivity_loss = connectivity_loss.view(connectives_padded_mask.shape) * connectives_padded_mask  # Bs * PAD_LEN
        connectivity_loss = connectivity_loss.sum(dim=1)/torch.clamp(connectives_padded_mask.sum(dim=1), min=1 ) # Bs * 1
        
        gen_hedge = gen_hedge * v_mask  
        return connectivity_loss, gen_hedge.view(pos_hyper.shape[0], -1).detach() 
    
    def supervision(self, memory, pos_hyperedge_padded, node_embeddings_t_prev , node_embeddings_t_cur, connectives_padded, cur_ts, prev_ts):
        # Supervising the model by predicting the time, size and connectivity of hyperedges conditioned on each node of hyperedge.
        mask = pos_hyperedge_padded.ne(0).int()
        delta_t_true  = (cur_ts - prev_ts).reshape(-1,1)
        assert (delta_t_true>=0.0).all()
        time_loss, time_estimation, time_estimation_mae, time_estimation_scaled_mse = self.time_modeling(
            node_embeddings_t_prev, mask, delta_t_true)

        k_true = mask.sum(axis = 1) - self.min_size
        size_loss, size_pred = self.size_modeling(node_embeddings_t_cur, k_true, mask)

        connectivity_loss, pred_conn_prob = self.connectivity_modeling(node_embeddings_t_cur, memory, pos_hyperedge_padded, connectives_padded)
        
        return time_loss, size_loss, connectivity_loss, time_estimation_mae, time_estimation_scaled_mse, delta_t_true, time_estimation, size_pred, pred_conn_prob