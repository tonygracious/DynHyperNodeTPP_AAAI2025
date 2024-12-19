
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from itertools import combinations
from Modules.hypergraphconv import HyperEdgeEmbed
from Modules.memory import Memory

from Modules.memory_updater import get_memory_updater
class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=[5, 10]):
        super(TimeEncode, self).__init__()

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((factor[0] * torch.from_numpy(1 / factor[1] ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())


    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)


class EmbeddingsTemporal(nn.Module):
    def __init__(self, n,  dim=64, factor=[50, 2], device='cuda'):
        super().__init__()
        self.n = n
        self.dim = dim
        self.memory = Memory(n, memory_dimension=dim, input_dimension=dim, message_dimension=dim, device=device, combination_method='sum')
        self.memory_updater = get_memory_updater("gru", self.memory, dim, dim, device )
        self.W1 = nn.Linear(dim ,dim)
        self.W2 = nn.Linear(dim, dim)
        self.time_embeddings = TimeEncode(dim, factor)


    def forward(self, nodes):
        # embeddings = self.graphconv(self.graph.to(device), self.embeddings)
        # embed_gcn = embeddings[nodes]
        # embed_self = self.embeddings[nodes]
        # embed = self.W1( embed_self ) + self.W2(embed_gcn)
        embed = torch.tanh(self.W1(self.embeddings[nodes]))
        return embed


    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"

        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = timestamps

        updated_memory = self.memory_updater(unique_messages, memory)

        self.memory.set_memory(unique_node_ids, updated_memory)

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"

        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update


class EmbeddingContinuous(EmbeddingsTemporal):
    def __init__(self, n,  dim=64, factor=[50, 2], device='cuda'):
        super().__init__(n, dim, factor, device)

    def forward(self, nodes, delta):
        embed_self = self.embeddings[nodes]
        embed = self.W1( embed_self ) + self.W2(self.time_embeddings(delta) )
        embed = torch.tanh(embed)
        return embed

class HGCNEmbeddingContinuous(EmbeddingsTemporal):
    def __init__(self, n,  dim=64, factor=[50, 2], device='cuda'):
        super().__init__(n, dim, factor, device)
        self.hgcn = HyperEdgeEmbed(dim, dim)
        self.W3 = nn.Sequential(nn.Linear(3*dim, 3*dim//2), nn.LayerNorm(3 * dim//2) ,nn.Tanh(), nn.Linear(3*dim//2, dim) ) #nn.Sequential(nn.Dropout(p=0.3),nn.Linear(3*dim, dim), nn.Tanh(), nn.Linear(dim//2, dim) )  #
        self.attnencoder = nn.MultiheadAttention(embed_dim=2*dim, kdim=dim*2, vdim=dim*2, num_heads=4, dropout=0.3, batch_first=True)
    def forward(self, memory, nodes,  h_index, times, mask):
        """

        :param memory: N x d
        :param nodes:  B  x PAD LEN
        :param h_index: B x 2 * unspecified length
        :param times: B x n_neighbors
        :param mask : B x n_neighbors
        return B x PADLEN x d
        """
        embed_self = memory[nodes] # B * negperedge x PADLEN x d
        bs=len(h_index) #Batchsize
        PADLEN = nodes.shape[1]
        h_index_concatenated=torch.cat(h_index,dim=1) # 2 X unspecified length
        embed_hyperedge  = self.hgcn(memory, h_index_concatenated.long()) #B * N  x d
        embed_hyperedge = embed_hyperedge.squeeze(dim=1).reshape(bs * PADLEN, -1, self.dim)
        source_nodes_time_embeddings = torch.cat( [embed_self, self.time_embeddings(torch.zeros_like(nodes))], dim=2)
        source_nodes_time_embeddings = source_nodes_time_embeddings.reshape(bs * PADLEN, 1, self.dim*2)
        times = times.reshape(bs* PADLEN, -1)
        embed_hyperedge_time_embeddings  = torch.cat( [embed_hyperedge, self.time_embeddings(times)], dim=2)
        mask = mask.reshape(bs * PADLEN, -1)
        mask = mask.bool()
        invalid_neighborhood_mask = mask.all(dim=1, keepdim=True)
        mask[invalid_neighborhood_mask.squeeze(), 0] = False
        attn_output, attn_output_weights = self.attnencoder(query=source_nodes_time_embeddings, key=embed_hyperedge_time_embeddings, \
                                                            value=embed_hyperedge_time_embeddings, key_padding_mask=mask)
        attn_output = attn_output.squeeze(dim=1)
        attn_output_weights = attn_output_weights.squeeze(dim=1)
        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
        attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)
        #embed = torch.tanh(self.W3(torch.cat([embed_self, attn_output], dim=2)))
        attn_output = attn_output.reshape(bs, PADLEN, self.dim*2)
        embed = torch.cat([embed_self, attn_output], dim=2)
        embed = torch.tanh(self.W3(embed))
        #embed = torch.tanh(self.W1(embed_self) + self.W3(attn_output))
        return embed

from Modules.hypergraphconv import BiHypergraphConv

class HGCNEmbedddingDiContinuous(nn.Module):
    def __init__(self, n,  dim=64, device='cuda', factor=[50, 2]):
        super().__init__()
        self.n = n
        self.dim = dim
        self.memory = Memory(n, memory_dimension=dim, input_dimension=dim, message_dimension=dim, device=device,
                             combination_method='sum')
        self.memory_updater = get_memory_updater("gru", self.memory, dim, dim, device)
        self.time_embeddings = TimeEncode(dim, factor)
        self.hgcn_right  = BiHypergraphConv(dim, dim)
        self.hgcn_left = BiHypergraphConv(dim, dim)
        self.W = nn.Sequential(nn.Linear(5*dim, 5*dim//2), nn.LayerNorm(5*dim//2), nn.Tanh(), nn.Linear(5*dim//2, dim) ) #nn.Linear(dim, dim) #
        self.W_right = nn.Sequential(nn.Linear(3*dim, 3*dim//2), nn.Tanh(), nn.Linear(3*dim//2, dim) )  #nn.Linear(2 * dim, dim)
        self.W_left = nn.Sequential(nn.Linear(3*dim, 3*dim//2),  nn.Tanh(), nn.Linear(3*dim//2, dim) )  #nn.Linear(2* dim, dim ) # nn.Sequential(nn.Dropout(p=0.3),nn.Linear(3*dim, dim), nn.Tanh(), nn.Linear(dim//2, dim) )
        self.attnencoder_right = nn.MultiheadAttention(embed_dim=2 * dim, kdim=dim * 2, vdim=dim * 2, num_heads=4,
                                                 dropout=0.3, batch_first=True)
        self.attnencoder_left =  nn.MultiheadAttention(embed_dim=2 * dim, kdim=dim * 2, vdim=dim * 2, num_heads=4,
                                                 dropout=0.3, batch_first=True)

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"

        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = timestamps

        updated_memory = self.memory_updater(unique_messages, memory)

        self.memory.set_memory(unique_node_ids, updated_memory)

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"

        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update
    def attention_encoding(self, memory, nodes, h_index, cur_time, prev_time, mask, right=True):

        embed_self = memory[nodes]
        bs = nodes.shape[0]
        PADLEN = nodes.shape[1]
        mask = mask.reshape(bs * PADLEN, -1)
        mask = mask.bool()
        (right_h_index, left_h_index) = h_index
        #if right:
        embed_hyperedge = self.hgcn_right(memory, (right_h_index.long(), left_h_index.long()))
        #else:
        #    embed_hyperedge = self.hgcn_left(memory, (right_h_index.long(), left_h_index.long()))
        embed_hyperedge = embed_hyperedge.squeeze(dim=1).reshape(bs * PADLEN, -1, self.dim)
        #assert (embed_hyperedge[mask] == 0).all(), "embed_hyperedge_right is not zero"

        source_nodes_time_embeddings = torch.cat([embed_self, self.time_embeddings(torch.zeros_like(nodes))], dim=2)
        source_nodes_time_embeddings = source_nodes_time_embeddings.reshape(bs * PADLEN, 1, self.dim * 2)

        time = (cur_time - prev_time).float()
        assert (prev_time.reshape(bs * PADLEN, -1)[mask.bool()] == 0).all(), "prev_time_left is not zero"
        assert (time>=0).all(), "error in time updating"
        time = time.reshape(bs * PADLEN, -1)
        embed_hyperedge_time_embeddings = torch.cat([embed_hyperedge, self.time_embeddings(time)], dim=2)
        embed_hyperedge_time_embeddings = embed_hyperedge_time_embeddings.reshape(bs * PADLEN, -1, self.dim * 2)

        invalid_neighborhood_mask = mask.all(dim=1, keepdim=True)
        mask[invalid_neighborhood_mask.squeeze(), 0] = False

        if right:
            attn_output, _ = self.attnencoder_right(source_nodes_time_embeddings, embed_hyperedge_time_embeddings, embed_hyperedge_time_embeddings, key_padding_mask=mask)
        else:
            attn_output, _ = self.attnencoder_left(source_nodes_time_embeddings, embed_hyperedge_time_embeddings, embed_hyperedge_time_embeddings, key_padding_mask=mask)
        attn_output = attn_output.squeeze(dim=1)
        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
        attn_output = attn_output.reshape(bs, PADLEN, self.dim * 2)
        return attn_output


    def node_temporal_embeddings(self, memory, nodes, h_index, cur_time, prev_time, mask):
        """"""
        embed_self = memory[nodes]
        attn_output_right = self.attention_encoding(memory, nodes, h_index[0], cur_time, prev_time[0], mask[0], right=True)
        attn_output_left  = self.attention_encoding(memory, nodes, h_index[1], cur_time, prev_time[1], mask[1], right=False)
        #embed = torch.tanh( self.W_right(torch.cat([embed_self, attn_output_right], dim=2) )  + self.W_left(torch.cat([embed_self, attn_output_left], dim=2)) )
        embed= torch.tanh(self.W( torch.cat([embed_self, attn_output_right, attn_output_left], dim=2)) )
        return embed
    def forward(self, memory, hyperedge, h_index, cur_time, prev_time, mask):
        """
        :param hyperedge: batch right nodes, batch left nodes
        :param delta:  delta right nodes, delta left nodes
        :return: embed right, embed left
        """
        embed_right = self.node_temporal_embeddings(memory, hyperedge[0], h_index[0], cur_time, prev_time[0], mask[0])
        embed_left  = self.node_temporal_embeddings(memory, hyperedge[1], h_index[1], cur_time, prev_time[1], mask[1])
        return embed_right, embed_left

if __name__ == '__main__':
    encoder = EmbeddingContinuous(10, factor=[1000, 10])
    print(encoder.n)
