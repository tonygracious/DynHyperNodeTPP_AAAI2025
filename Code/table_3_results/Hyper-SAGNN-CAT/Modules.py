import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm, trange
import copy
import math

dir_cls_flag = False
def setdevice(gpu):
    global device 
    device =  torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
def set_dir_cls_flag(bool_value):
    global dir_cls_flag 
    dir_cls_flag = bool_value


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk
    
    return padding_mask

class PositionalEncoding(nn.Module):
    def __init__(self,d_model, dropout , max_len):
        super(PositionalEncoding, self).__init__()
        print('Using Static Pos Embedding')
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1,max_len,d_model))
        X = torch.arange(0,max_len).reshape(-1,1) / torch.pow(10000,torch.arange(0,d_model,2) / d_model)
        self.P[:,:,0::2] = torch.sin(X)
        self.P[:,:,1::2] = torch.cos(X)
        self.P = self.P.to(device)
    def forward(self,X, pos):
        pos += 1
        X = X + self.P[:,pos,:]
        return self.dropout(X).float()

class LearnablePositionalEncoding(nn.Module):
    def __init__(self,num_pos,d_model, dropout,add):
        super(LearnablePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.add = add
        self.dropout = nn.Dropout(dropout)
        self.P = nn.Embedding(num_pos,d_model)
        self.linear_merger = nn.Linear(d_model * 2, d_model,bias= False)
    def forward(self,X, pos):
        pos = torch.tensor(pos).to(device)
        if self.add:
            X = X + self.P(pos)
        else:
            pos_enc = self.P(pos).repeat(X.shape[0],1)
            X = torch.cat([X , pos_enc] , axis = -1)
            # X = self.linear_merger(X)
        return self.dropout(X).float()

class Wrap_Embedding(torch.nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, *input):
        return super().forward(*input), torch.Tensor([0]).to(device)

# Used only for really big adjacency matrix


class SparseEmbedding(nn.Module):
    def __init__(self, embedding_weight, sparse=True):
        super().__init__()
        print(embedding_weight.shape)
        self.sparse = sparse
        if self.sparse:
            self.embedding = embedding_weight
        else:
            try:
                try:
                    self.embedding = torch.from_numpy(
                        np.asarray(embedding_weight.todense())).to(device)
                except BaseException:
                    self.embedding = torch.from_numpy(
                        np.asarray(embedding_weight)).to(device)
            except Exception as e:
                print("Sparse Embedding Error",e)
                self.sparse = True
                self.embedding = embedding_weight
    
    def forward(self, x):
        
        if self.sparse:
            x = x.cpu().numpy()
            x = x.reshape((-1))
            temp = np.asarray((self.embedding[x, :]).todense())
            
            return torch.from_numpy(temp).to(device)
        else:
            return self.embedding[x, :]


class TiedAutoEncoder(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.Tensor(out, inp))
        self.bias1 = nn.parameter.Parameter(torch.Tensor(out))
        self.bias2 = nn.parameter.Parameter(torch.Tensor(inp))
        
        self.register_parameter('tied weight',self.weight)
        self.register_parameter('tied bias1', self.bias1)
        self.register_parameter('tied bias2', self.bias2)
        
        self.reset_parameters()
        
    

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias1 is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias1, -bound, bound)
        
        if self.bias2 is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            torch.nn.init.uniform_(self.bias2, -bound, bound)

    def forward(self, input):
        encoded_feats = F.linear(input, self.weight, self.bias1)
        encoded_feats = F.tanh(encoded_feats)
        reconstructed_output = F.linear(encoded_feats, self.weight.t(), self.bias2)
        return encoded_feats, reconstructed_output
    
class MultipleEmbedding(nn.Module):
    def __init__(
            self,
            embedding_weights,
            dim,
            sparse=True,
            num_list=None,
            node_type_mapping=None):
        super().__init__()
        print(dim)
        self.num_list = torch.tensor([0] + list(num_list)).to(device)
        print(self.num_list)
        self.node_type_mapping = node_type_mapping
        self.dim = dim
        self.embeddings = []
        for i, w in enumerate(embedding_weights):
            try:
                self.embeddings.append(SparseEmbedding(embedding_weight = w, device = device, sparse = sparse))
            except BaseException as e:
                print ("Conv Embedding Mode")
                self.add_module("ConvEmbedding1", w)
                self.embeddings.append(w)
        
        test = torch.zeros(1, device=device).long()
        self.input_size = []
        for w in self.embeddings:
            self.input_size.append(w(test).shape[-1])
        
        self.wstack = [TiedAutoEncoder(self.input_size[i],self.dim).to(device) for i,w in enumerate(self.embeddings)]
        self.norm_stack =[nn.LayerNorm(self.dim).to(device) for w in self.embeddings]
        for i, w in enumerate(self.wstack):
            self.add_module("Embedding_Linear%d" % (i), w)
            self.add_module("Embedding_norm%d" % (i), self.norm_stack[i])
            
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        
        final = torch.zeros((len(x), self.dim)).to(device)
        recon_loss = torch.Tensor([0.0]).to(device)
        for i in range(len(self.num_list) - 1):
            select = (x >= (self.num_list[i] + 1)) & (x < (self.num_list[i + 1] + 1))
            if torch.sum(select) == 0:
                continue
            adj = self.embeddings[i](x[select] - self.num_list[i] - 1)
            output = self.dropout(adj)
            output, recon = self.wstack[i](output)
            output = self.norm_stack[i](output)
            final[select] = output
            recon_loss += sparse_autoencoder_error(recon, adj)
            
        return final, recon_loss

def sparse_autoencoder_error(y_pred, y_true):
    return torch.mean(torch.sum((y_true.ne(0).type(torch.float) * (y_true - y_pred)) ** 2, dim = -1) / torch.sum(y_true.ne(0).type(torch.float), dim = -1))

class Word2vec_Skipgram(nn.Module):
    def __init__(
            self,
            dict_size,
            embedding_dim,
            window_size,
            u_embedding=None,
            sparse=False):
        super(Word2vec_Skipgram, self).__init__()
        '''
        use context (u) to predict center (v)
        '''
        self.dict_size = dict_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        
        self.u_embedding = u_embedding
        self.sm_w_t = nn.Embedding(
            dict_size,
            embedding_dim,
            sparse=sparse,
            padding_idx=0,
        )
        self.sm_b = nn.Embedding(dict_size, 1, sparse=sparse, padding_idx=0, )
    
    def forward_u(self, u):
        return self.u_embedding(u)
    
    def forward_w_b(self, id):
        return self.sm_w_t(id), self.sm_b(id)

class Classifier(nn.Module):
    def __init__(
            self,
            n_head,
            d_model,
            d_k,
            d_v,
            num_nodes ,
            pad_idx,
            emb_size ,
            diag_mask,
            bottle_neck,
            **args):
        super().__init__()
        
        self.pff_classifier = PositionwiseFeedForward(
            [d_model, 1], reshape=True, use_bias=True)
        
        self.node_embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=emb_size,padding_idx=pad_idx)
        self.encode1 = EncoderLayer(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout_mul=0.3,
            dropout_pff=0.4,
            diag_mask=diag_mask,
            bottle_neck=bottle_neck,
            )
        # self.encode2 = EncoderLayer(n_head, d_model, d_k, d_v, dropout_mul=0.0, dropout_pff=0.0, diag_mask = diag_mask, bottle_neck=bottle_neck)
        self.diag_mask_flag = diag_mask
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def get_node_embeddings(self, x,return_recon = False):
        # shape of x: (b, tuple)
        sz_b, len_seq = x.shape
        # print(torch.max(x), torch.min(x))
        
        x = self.node_embedding(x.view(-1))
        if return_recon:
            return x.view(sz_b, len_seq, -1)
        else:
            return x.view(sz_b, len_seq, -1)
    
    def get_embedding(self, x, slf_attn_mask, non_pad_mask,return_recon = False):
        if return_recon:
            x, recon_loss = self.get_node_embeddings(x,return_recon)
        else:
            x = self.get_node_embeddings(x, return_recon)
        dynamic, static, attn = self.encode1(x, x, slf_attn_mask, non_pad_mask)
        # dynamic, static1, attn = self.encode2(dynamic, static,slf_attn_mask, non_pad_mask)
        if return_recon:
            return dynamic, static, attn, recon_loss
        else:
            return dynamic, static, attn
    
    def get_embedding_static(self, x):
        if len(x.shape) == 1:
            x = x.view(-1, 1)
            flag = True
        else:
            flag = False
        slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        non_pad_mask = get_non_pad_mask(x)
        x = self.get_node_embeddings(x)
        dynamic, static, attn = self.encode1(x, x, slf_attn_mask, non_pad_mask)
        # dynamic, static, attn = self.encode2(dynamic, static,slf_attn_mask, non_pad_mask)
        if flag:
            return static[:, 0, :]
        return static
    
    def forward(self, x, mask=None, get_outlier=None, return_recon = False):
        x = x.long()
        slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        non_pad_mask = get_non_pad_mask(x)
        
        if return_recon:
            dynamic, static, attn, recon_loss = self.get_embedding(x, slf_attn_mask, non_pad_mask,return_recon)
        else:
            dynamic, static, attn = self.get_embedding(x, slf_attn_mask, non_pad_mask, return_recon)
        dynamic = self.layer_norm1(dynamic)
        static = self.layer_norm2(static)
        sz_b, len_seq, dim = dynamic.shape
        
        if self.diag_mask_flag == 'True':
            output = (dynamic - static) ** 2
        else:
            output = dynamic
        
        output = self.pff_classifier(output)
        output = torch.sigmoid(output)
        
        
        if get_outlier is not None:
            k = get_outlier
            outlier = (
                    (1 -
                     output) *
                    non_pad_mask).topk(
                k,
                dim=1,
                largest=True,
                sorted=True)[1]
            return outlier.view(-1, k)
        
        mode = 'sum'
        
        if mode == 'min':
            output, _ = torch.max(
                (1 - output) * non_pad_mask, dim=-2, keepdim=False)
            output = 1 - output
        
        elif mode == 'sum':
            output = torch.sum(output * non_pad_mask, dim=-2, keepdim=False)
            mask_sum = torch.sum(non_pad_mask, dim=-2, keepdim=False)
            output /= mask_sum
        elif mode == 'first':
            output = output[:, 0, :]
            
        if return_recon:
            return output, recon_loss
        else:
            return output


# A custom position-wise MLP.
# dims is a list, it would create multiple layer with tanh between them
# If dropout, it would add the dropout at the end. Before residual and
# layer-norm


class PositionwiseFeedForward(nn.Module):
    def __init__(
            self,
            dims,
            dropout=None,
            reshape=False,
            use_bias=True,
            residual=False,
            layer_norm=False):
        super(PositionwiseFeedForward, self).__init__()
        self.w_stack = []
        self.dims = dims
        for i in range(len(dims) - 1):
            self.w_stack.append(nn.Conv1d(dims[i], dims[i + 1], 1, bias = use_bias))
            self.add_module("PWF_Conv%d" % (i), self.w_stack[-1])
        self.reshape = reshape
        self.layer_norm = nn.LayerNorm(dims[-1])
        
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.residual = residual
        self.layer_norm_flag = layer_norm
    
    def forward(self, x):
        output = x.transpose(1, 2)
        
        
        for i in range(len(self.w_stack) - 1):
            output = self.w_stack[i](output)
            output = torch.tanh(output)
            if self.dropout is not None:
                output = self.dropout(output)
        
        output = self.w_stack[-1](output)
        output = output.transpose(1, 2)
        
        if self.reshape:
            output = output.view(output.shape[0], -1, 1)
        
        if self.dims[0] == self.dims[-1]:
            # residual
            if self.residual:
                output += x

            if self.layer_norm_flag:
                output = self.layer_norm(output)
        
        return output


# A custom position wise MLP.
# dims is a list, it would create multiple layer with torch.tanh between them
# We don't do residual and layer-norm, because this is only used as the
# final classifier


class FeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    
    def __init__(self, dims, dropout=None, reshape=False, use_bias=True):
        super(FeedForward, self).__init__()
        self.w_stack = []
        for i in range(len(dims) - 1):
            self.w_stack.append(nn.Linear(dims[i], dims[i + 1], use_bias))
            self.add_module("FF_Linear%d" % (i), self.w_stack[-1])
        
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.reshape = reshape
    
    def forward(self, x):
        output = x
        for i in range(len(self.w_stack) - 1):
            output = self.w_stack[i](output)
            output = torch.tanh(output)
            if self.dropout is not None:
                output = self.dropout(output)
        output = self.w_stack[-1](output)
        
        if self.reshape:
            output = output.view(output.shape[0], -1, 1)
        
        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
    
    def masked_softmax(self, vector: torch.Tensor,
                       mask: torch.Tensor,
                       dim: int = -1,
                       memory_efficient: bool = False,
                       mask_fill_value: float = -1e32) -> torch.Tensor:
        
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside
                # the mask, we zero these out.
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill(
                    (1 - mask).bool(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)
        return result
    
    def forward(self, q, k, v, diag_mask, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        
        ## dir_cls_flag is global variable set in Directed_Classifier
        if dir_cls_flag:
            combined_mask = 1 - ( ( (1- diag_mask)  + mask.float() ) > 0 ).float()
            attn = torch.nn.functional.softmax(attn * combined_mask, dim=-1)
            attn = attn * combined_mask
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-13)
        else:
            if mask is not None:
                attn = attn.masked_fill(mask, -float('inf'))
            
            attn = self.masked_softmax(
                attn, diag_mask, dim=-1, memory_efficient=True)
        
        
        output = torch.bmm(attn, v)
        
        return output, attn


# class MultiHeadAttention(nn.Module):
#     ''' Multi-Head Attention module '''
    
#     def __init__(
#             self,
#             n_head,
#             d_model,
#             d_k,
#             d_v,
#             dropout,
#             diag_mask,
#             input_dim,
#             ):
#         super().__init__()
        
#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v
        
#         self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
#         self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
#         self.w_vs = nn.Linear(input_dim, n_head * d_v, bias=False)
        
#         nn.init.normal_(self.w_qs.weight, mean=0,
#                         std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.w_ks.weight, mean=0,
#                         std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.w_vs.weight, mean=0,
#                         std=np.sqrt(2.0 / (d_model + d_v)))
        
#         self.attention = ScaledDotProductAttention(
#             temperature=np.power(d_k, 0.5))
        
#         self.fc1 = FeedForward([n_head * d_v, d_model], use_bias=False)
#         self.fc2 = FeedForward([n_head * d_v, d_model], use_bias=False)
        
#         self.layer_norm1 = nn.LayerNorm(input_dim)
#         self.layer_norm2 = nn.LayerNorm(input_dim)
#         self.layer_norm3 = nn.LayerNorm(input_dim)
        
#         if dropout is not None:
#             self.dropout = nn.Dropout(dropout)
#         else:
#             self.dropout = dropout
        
#         self.diag_mask_flag = diag_mask
#         self.diag_mask = None
    
#     def pass_(self, inputs):
#         return inputs
    
#     def forward(self, q, k, v,mask=None):
#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        
#         residual_dynamic = q
#         residual_static = v
        
#         q = self.layer_norm1(q)
#         k = self.layer_norm2(k)
#         v = self.layer_norm3(v)
        
#         sz_b, len_q, _ = q.shape
#         sz_b, len_k, _ = k.shape
#         sz_b, len_v, _ = v.shape
        
#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
#         q = q.permute(2, 0, 1, 3).contiguous(
#         ).view(-1, len_q, d_k)  # (n*b) x lq x dk
#         k = k.permute(2, 0, 1, 3).contiguous(
#         ).view(-1, len_k, d_k)  # (n*b) x lk x dk
#         v = v.permute(2, 0, 1, 3).contiguous(
#         ).view(-1, len_v, d_v)  # (n*b) x lv x dv
        
#         n = sz_b * n_head
        
#         if self.diag_mask is not None:
#             if (len(self.diag_mask) <= n) or (
#                     self.diag_mask.shape[1] != len_v):
#                 self.diag_mask = torch.ones((len_v, len_v), device=device)
#                 if self.diag_mask_flag == 'True':
#                     self.diag_mask -= torch.eye(len_v, len_v, device=device)
#                 self.diag_mask = self.diag_mask.repeat(n, 1, 1)
#                 diag_mask = self.diag_mask
#             else:
#                 diag_mask = self.diag_mask[:n]
        
#         else:
#             self.diag_mask = (torch.ones((len_v, len_v), device=device))
#             if self.diag_mask_flag == 'True':
#                 self.diag_mask -= torch.eye(len_v, len_v, device=device)
#             self.diag_mask = self.diag_mask.repeat(n, 1, 1)
#             diag_mask = self.diag_mask
        
#         if mask is not None:
#             mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        
#         dynamic, attn = self.attention(q, k, v, diag_mask, mask=mask)
        
#         dynamic = dynamic.view(n_head, sz_b, len_q, d_v)
#         dynamic = dynamic.permute(
#             1, 2, 0, 3).contiguous().view(
#             sz_b, len_q, -1)  # b x lq x (n*dv)
#         static = v.view(n_head, sz_b, len_q, d_v)
#         static = static.permute(
#             1, 2, 0, 3).contiguous().view(
#             sz_b, len_q, -1)  # b x lq x (n*dv)
        
#         dynamic = self.dropout(self.fc1(dynamic)) if self.dropout is not None else self.fc1(dynamic)
#         static = self.dropout(self.fc2(static)) if self.dropout is not None else self.fc2(static)
        
        
#         return dynamic, static, attn

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout, diag_mask, input_dim, static_flag= True):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.static_flag = static_flag

        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(input_dim, n_head * d_v, bias=False)

        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.fc1 = FeedForward([n_head * d_v, d_model], use_bias=False)
        if self.static_flag:
            self.fc2 = FeedForward([n_head * d_v, d_model], use_bias=False)

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

        self.diag_mask_flag = diag_mask
        self.diag_mask = None

    def pass_(self, inputs):
        return inputs

    def forward(self, q, k, v, diag_mask = None , mask=None):
        # pdb.set_trace()
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        residual_dynamic = q
        residual_static = v
        q = self.layer_norm1(q)
        k = self.layer_norm2(k)
        v = self.layer_norm3(v)

        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        n = sz_b * n_head
        """change masking matrix from len_v to len_q for cross attentions"""
        self.diag_mask = (torch.ones((len_q, len_v), device=device))
        if self.diag_mask_flag == 'True':
            self.diag_mask -= torch.eye(len_q, len_v, device=device)
        self.diag_mask = self.diag_mask.repeat(n, 1, 1)
        diag_mask = self.diag_mask
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        dynamic, attn = self.attention(q, k, v, diag_mask, mask=mask)
        dynamic = dynamic.view(n_head, sz_b, len_q, d_v)
        dynamic = dynamic.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        dynamic = self.dropout(self.fc1(dynamic)) if self.dropout is not None else self.fc1(dynamic)
        if self.static_flag:

            static = v.view(n_head, sz_b, len_k, d_v)
            static = static.permute(1, 2, 0, 3).contiguous().view(sz_b, len_k, -1)  # b x lq x (n*dv)
            static = self.dropout(self.fc2(static)) if self.dropout is not None else self.fc2(static)
            return dynamic, static, attn
        else:

            return dynamic, attn


class EncoderLayer(nn.Module):
    '''A self-attention layer + 2 layered pff'''
    
    def __init__(
            self,
            n_head,
            d_model,
            d_k,
            d_v,
            dropout_mul,
            dropout_pff,
            diag_mask,
            bottle_neck,
            ):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.mul_head_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout_mul,
            diag_mask=diag_mask,
            input_dim=bottle_neck,
            )
        self.pff_n1 = PositionwiseFeedForward(
            [d_model, d_model, d_model], dropout=dropout_pff, residual=True, layer_norm=True)
        self.pff_n2 = PositionwiseFeedForward(
            [bottle_neck, d_model, d_model], dropout=dropout_pff, residual=False, layer_norm=True)
    
    # self.dropout = nn.Dropout(0.2)
    
    def forward(self, dynamic, static, slf_attn_mask, non_pad_mask):
        dynamic, static1, attn = self.mul_head_attn(
            dynamic, dynamic, static, mask = slf_attn_mask)
        dynamic = self.pff_n1(dynamic * non_pad_mask) * non_pad_mask
        static1 = self.pff_n2(static * non_pad_mask) * non_pad_mask
        
        return dynamic, static1, attn


#### Modules for CATSETMET
class OnlyCrossAttention(nn.Module):
    """A self-attention layer + 2 layered pff"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout_mul, dropout_pff, diag_mask, bottle_neck):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.cross_attn_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                               diag_mask=diag_mask, input_dim=bottle_neck, static_flag=True)

        self.cross_attn_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                               diag_mask=diag_mask, input_dim=bottle_neck, static_flag=True)

        self.pff_U1 = PositionwiseFeedForward([d_model, d_model, d_model],
                                              dropout=dropout_pff, residual=True, layer_norm=True)

        self.pff_U2 = PositionwiseFeedForward([bottle_neck, d_model, d_model],
                                              dropout=dropout_pff, residual=False, layer_norm=True)

        self.pff_V1 = PositionwiseFeedForward([d_model, d_model, d_model],
                                              dropout=dropout_pff, residual=True, layer_norm=True)

        self.pff_V2 = PositionwiseFeedForward([bottle_neck, d_model, d_model],
                                              dropout=dropout_pff, residual=False, layer_norm=True)

    def forward(self, dynamic_1, dynamic_2, static_1, static_2, crs_attn_mask1, crs_attn_mask2, slf_attn_mask1,
                slf_attn_mask2, non_pad_mask1, non_pad_mask2):
        """here the static_1 refer to the input embeddings of U_side while static_2 relates the embeddings of V sides
        and dynamic_1 refer to query embedding of u side(input) and  dynamic_2 refers to embeddings of V sides """

        """only change is now self attention mask and non pad_mask"""  ########
        # pdb.set_trace()
        dynamic2u, static2, cr_attn_u = self.cross_attn_u(dynamic_1, static_2, static_2, diag_mask=None,
                                                          mask=crs_attn_mask1)

        dynamic2v, static1, cr_attn_v = self.cross_attn_v(dynamic_2, static_1, static_1, diag_mask=None,
                                                          mask=crs_attn_mask2)

        output_attn = [cr_attn_u, cr_attn_v]

        dynamic1 = self.pff_U1(dynamic2u * non_pad_mask1) * non_pad_mask1
        static1 = self.pff_U2(static1 * non_pad_mask1) * non_pad_mask1
        dynamic2 = self.pff_V1(dynamic2v * non_pad_mask2) * non_pad_mask2
        static2 = self.pff_V2(static2 * non_pad_mask2) * non_pad_mask2
        return dynamic1, static1, dynamic2, static2, output_attn


class CrossAttention(nn.Module):
    """A self-attention layer + 2 layered pff"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout_mul, dropout_pff, diag_mask, bottle_neck):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.slf_attn_lv1_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                 diag_mask=diag_mask, input_dim=bottle_neck, static_flag=True)
        self.slf_attn_lv1_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                 diag_mask=diag_mask, input_dim=bottle_neck, static_flag=True)

        self.cross_attn_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                               diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)

        self.cross_attn_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                               diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)

        self.slf_attn_lv2_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                 diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)

        self.slf_attn_lv2_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                 diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)

        self.pff_U1 = PositionwiseFeedForward([d_model, d_model, d_model],
                                              dropout=dropout_pff, residual=True, layer_norm=True)

        self.pff_U2 = PositionwiseFeedForward([bottle_neck, d_model, d_model],
                                              dropout=dropout_pff, residual=False, layer_norm=True)

        self.pff_V1 = PositionwiseFeedForward([d_model, d_model, d_model],
                                              dropout=dropout_pff, residual=True, layer_norm=True)

        self.pff_V2 = PositionwiseFeedForward([bottle_neck, d_model, d_model],
                                              dropout=dropout_pff, residual=False, layer_norm=True)

    # self.dropout = nn.Dropout(0.2)

    def forward(self, dynamic_1, dynamic_2, static_1, static_2, crs_attn_mask1, crs_attn_mask2, slf_attn_mask1,
                slf_attn_mask2, non_pad_mask1, non_pad_mask2):
        """here the static_1 refer to the input embeddings of U_side while static_2 relates the embeddings of V sides
        and dynamic_1 refer to query embedding of u side(input) and  dynamic_2 refers to embeddings of V sides """

        """only change is now self attention mask and non pad_mask"""  ########
        # pdb.set_trace()
        dynamic1u, static1, attn_lv1u = self.slf_attn_lv1_u(dynamic_1, static_1, static_1, diag_mask=None,
                                                            mask=slf_attn_mask1)
        dynamic1v, static2, attn_lv1v = self.slf_attn_lv1_v(dynamic_2, static_2, static_2, diag_mask=None,
                                                            mask=slf_attn_mask2)
        dynamic2u, cr_attn_u = self.cross_attn_u(dynamic1u, dynamic1v, dynamic1v, diag_mask=None, mask=crs_attn_mask1)
        dynamic2v, cr_attn_v = self.cross_attn_v(dynamic1v, dynamic1u, dynamic1u, diag_mask=None, mask=crs_attn_mask2)
        dynamic3u, attn_lv2u = self.slf_attn_lv2_u(dynamic2u, dynamic2u, dynamic2u, diag_mask=None, mask=slf_attn_mask1)
        dynamic3v, attn_lv2v = self.slf_attn_lv2_v(dynamic2v, dynamic2v, dynamic2v, diag_mask=None, mask=slf_attn_mask2)
        output_attn = [attn_lv1u, attn_lv1v, attn_lv2u, attn_lv2v, cr_attn_u, cr_attn_v]

        # dynamic1, cr_attn1 = self.mul_head_attn_forward(dynamic_1, static_2, static_2, diag_mask=None,
        #                                                 mask=crs_attn_mask1)
        # dynamic2, cr_attn2 = self.mul_head_attn_backward(dynamic_2, static_1, static_1, diag_mask=None,
        #                                                  mask=crs_attn_mask2)
        # static1, slf_attn1 = self.mul_head_attn_selfU(dynamic_1, static_1, static_1, diag_mask=None,
        #                                               mask=slf_attn_mask1)
        # static2, slf_attn2 = self.mul_head_attn_selfV(dynamic_2, static_2, static_2, diag_mask=None,
        #                                               mask=slf_attn_mask2)

        dynamic1 = self.pff_U1(dynamic3u * non_pad_mask1) * non_pad_mask1
        static1 = self.pff_U2(static1 * non_pad_mask1) * non_pad_mask1
        dynamic2 = self.pff_V1(dynamic3v * non_pad_mask2) * non_pad_mask2
        static2 = self.pff_V2(static2 * non_pad_mask2) * non_pad_mask2
        return dynamic1, static1, dynamic2, static2, output_attn


class CrossAttentionSimple(nn.Module):
    """A self-attention layer + 2 layered pff"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout_mul, dropout_pff, diag_mask, bottle_neck):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.slf_attn_lv1_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                 diag_mask=diag_mask, input_dim=bottle_neck, static_flag=True)
        self.slf_attn_lv1_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                 diag_mask=diag_mask, input_dim=bottle_neck, static_flag=True)

        self.cross_attn_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                               diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)
        self.cross_attn_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                               diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)

        # self.slf_attn_lv2_u = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
        #                                          diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)
        #
        # self.slf_attn_lv2_v = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
        #                                          diag_mask=diag_mask, input_dim=bottle_neck, static_flag=False)

        self.pff_U1 = PositionwiseFeedForward([d_model, d_model, d_model],
                                              dropout=dropout_pff, residual=True, layer_norm=True)
        self.pff_U2 = PositionwiseFeedForward([bottle_neck, d_model, d_model],
                                              dropout=dropout_pff, residual=False, layer_norm=True)
        self.pff_V1 = PositionwiseFeedForward([d_model, d_model, d_model],
                                              dropout=dropout_pff, residual=True, layer_norm=True)
        self.pff_V2 = PositionwiseFeedForward([bottle_neck, d_model, d_model],
                                              dropout=dropout_pff, residual=False, layer_norm=True)

    # self.dropout = nn.Dropout(0.2)

    def forward(self, dynamic_1, dynamic_2, static_1, static_2, crs_attn_mask1, crs_attn_mask2, slf_attn_mask1,
                slf_attn_mask2, non_pad_mask1, non_pad_mask2):
        """here the static_1 refer to the input embeddings of U_side while static_2 relates the embeddings of V sides
        and dynamic_1 refer to query embedding of u side(input) and  dynamic_2 refers to embeddings of V sides """

        """only change is now self attention mask and non pad_mask"""  ########
        # pdb.set_trace()
        dynamic1u, static1, attn_lv1u = self.slf_attn_lv1_u(dynamic_1, static_1, static_1, diag_mask=None,
                                                            mask=slf_attn_mask1)

        dynamic1v, static2, attn_lv1v = self.slf_attn_lv1_v(dynamic_2, static_2, static_2, diag_mask=None,
                                                            mask=slf_attn_mask2)

        dynamic2u, cr_attn_u = self.cross_attn_u(dynamic1u, dynamic1v, dynamic1v, diag_mask=None, mask=crs_attn_mask1)

        dynamic2v, cr_attn_v = self.cross_attn_v(dynamic1v, dynamic1u, dynamic1u, diag_mask=None, mask=crs_attn_mask2)

        # dynamic3u, attn_lv2u = self.slf_attn_lv2_u(dynamic2u, dynamic2u, dynamic2u, diag_mask=None,
        #                                            mask=slf_attn_mask1)
        # dynamic3v, attn_lv2v = self.slf_attn_lv2_v(dynamic2v, dynamic2v, dynamic2v, diag_mask=None,
        #                                            mask=slf_attn_mask2)

        output_attn = [attn_lv1u, attn_lv1v, cr_attn_u, cr_attn_v]

        # dynamic1, cr_attn1 = self.mul_head_attn_forward(dynamic_1, static_2, static_2, diag_mask=None,
        #                                                 mask=crs_attn_mask1)
        # dynamic2, cr_attn2 = self.mul_head_attn_backward(dynamic_2, static_1, static_1, diag_mask=None,
        #                                                  mask=crs_attn_mask2)
        # static1, slf_attn1 = self.mul_head_attn_selfU(dynamic_1, static_1, static_1, diag_mask=None,
        #                                               mask=slf_attn_mask1)
        # static2, slf_attn2 = self.mul_head_attn_selfV(dynamic_2, static_2, static_2, diag_mask=None,
        #                                               mask=slf_attn_mask2)

        dynamic1 = self.pff_U1(dynamic2u * non_pad_mask1) * non_pad_mask1
        static1 = self.pff_U2(static1 * non_pad_mask1) * non_pad_mask1
        dynamic2 = self.pff_V1(dynamic2v * non_pad_mask2) * non_pad_mask2
        static2 = self.pff_V2(static2 * non_pad_mask2) * non_pad_mask2
        return dynamic1, static1, dynamic2, static2, output_attn


class CatClassifier(nn.Module):
    """a classifier is the main model for embeddings"""

    def __init__(self, n_head, d_model, d_k, d_v, num_nodes , pad_idx,emb_size, diag_mask, bottle_neck, **args):
        super().__init__()

        print(args)
        self.pff_classifier1 = PositionwiseFeedForward([d_model, 1], reshape=True, use_bias=True)
        self.pff_classifier2 = PositionwiseFeedForward([d_model, 1], reshape=True, use_bias=True)
        self.pff_classifier3 = PositionwiseFeedForward([1, 1], reshape=True, use_bias=True)
        # self.pff_classifier3 = PositionwiseFeedForward([d_model, 1], reshape=True, use_bias=True)

        """remove positional embedding"""  ###########

        self.node_embedding1 =  nn.Embedding(num_embeddings=num_nodes, embedding_dim=emb_size,padding_idx=pad_idx)
        self.node_embedding2 =  nn.Embedding(num_embeddings=num_nodes, embedding_dim=emb_size,padding_idx=pad_idx)
        
        if args['cross_attn_type'] == 'x':
            model_init = OnlyCrossAttention
        elif args['cross_attn_type'] == 'sx':
            model_init = CrossAttentionSimple
        elif args['cross_attn_type'] == 'sxs':
            model_init = CrossAttention
        else:
            raise Exception('No Cross Attention Type specified.')
        self.encode1 = model_init(n_head, d_model, d_k, d_v, dropout_mul=0.4, dropout_pff=0.4,
                                      diag_mask=diag_mask, bottle_neck=bottle_neck)

        self.diag_mask_flag = diag_mask
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.layer_norm4 = nn.LayerNorm(d_model)

    def get_node_embeddings(self, x, mode, return_recon=False):

        # shape of x: (b, tuple)
        sz_b, len_seq = x.shape
        # print(torch.max(x), torch.min(x))
        if mode == 1:
            x = self.node_embedding1(x.view(-1))
        else:
            x = self.node_embedding2(x.view(-1))
        
        return x.view(sz_b, len_seq, -1)

    def get_embedding(self, x, y, crs_attn_mask1, crs_attn_mask2, slf_attn_mask1, slf_attn_mask2, non_pad_mask1,
                      non_pad_mask2, return_recon=False):
        if return_recon:
            x, recon_loss1 = self.get_node_embeddings(x, 1, return_recon)
            y, recon_loss2 = self.get_node_embeddings(y, 2, return_recon)
        else:
            x = self.get_node_embeddings(x, 1, return_recon)
            y = self.get_node_embeddings(y, 2, return_recon)
            recon_loss1, recon_loss2 = None, None
        dynamic1, static1, dynamic2, static2, output_attn = self.encode1(x, y, x, y, crs_attn_mask1, crs_attn_mask2,
                                                                         slf_attn_mask1, slf_attn_mask2, non_pad_mask1,
                                                                         non_pad_mask2)
        if return_recon:
            return dynamic1, static1, dynamic2, static2, output_attn, recon_loss1, recon_loss2
        else:
            return dynamic1, static1, dynamic2, static2, output_attn

    def forward(self, x, y, mask=None, get_outlier=None, return_recon=False):
        x = x.long()
        y = y.long()
        # pdb.set_trace()
        cr_attn_mask1 = get_attn_key_pad_mask(seq_k=y, seq_q=x)
        slf_attn_mask1 = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        non_pad_mask1 = get_non_pad_mask(x)
        cr_attn_mask2 = get_attn_key_pad_mask(seq_k=x, seq_q=y)
        slf_attn_mask2 = get_attn_key_pad_mask(seq_k=y, seq_q=y)
        non_pad_mask2 = get_non_pad_mask(y)
        if return_recon:
            dynamic1, static1, dynamic2, static2, \
                recon_loss1, recon_loss2 = self.get_embedding(x, y, cr_attn_mask1, cr_attn_mask2,
                                                              slf_attn_mask1, slf_attn_mask2,
                                                              non_pad_mask1, non_pad_mask2, return_recon)
        else:
            dynamic1, static1, dynamic2, static2, \
                output_attn = self.get_embedding(x, y, cr_attn_mask1, cr_attn_mask2, slf_attn_mask1, slf_attn_mask2,
                                                 non_pad_mask1, non_pad_mask2, return_recon)

        dynamic1 = self.layer_norm1(dynamic1)
        static1 = self.layer_norm2(static1)
        dynamic2 = self.layer_norm3(dynamic2)
        static2 = self.layer_norm4(static2)
        sz_b, len_seq, dim = dynamic1.shape
        # pdb.set_trace()
        # output=torch.cat([((dynamic1-static1)**2),((dynamic2-static2)**2)],dim=1)
        output1 = self.pff_classifier1((dynamic1 - static1) ** 2)
        output2 = self.pff_classifier2((dynamic2 - static2) ** 2)
        # output = dynamic1**2+dynamic2**2
        # output1 = self.pff_classifier(dynamic1**2)
        # output2 = self.pff_classifier(dynamic2**2)
        output = torch.cat([output1, output2], axis=1)
        output = self.pff_classifier3(output)
        # pdb.set_trace()
        # output = torch.sigmoid(torch.cat([output1,output2],axis=1))
        output = torch.sigmoid(output)

        # embedding_after_attn = [[static1,dynamic1],[static2,dynamic2], output_attn]
        embedding_after_attn = None
        non_pad_mask = torch.cat([non_pad_mask1, non_pad_mask2], axis=1)
        if get_outlier is not None:
            k = get_outlier
            outlier = ((1 - output) * non_pad_mask).topk(k, dim=1, largest=True, sorted=True)[1]
            return outlier.view(-1, k)

        mode = 'first'
        if mode == 'min':
            output, _ = torch.max(
                (1 - output) * non_pad_mask, dim=-2, keepdim=False)
            output = 1 - output

        elif mode == 'sum':
            output = torch.sum(output * non_pad_mask, dim=-2, keepdim=False)
            mask_sum = torch.sum(non_pad_mask, dim=-2, keepdim=False)
            output /= mask_sum

        elif mode == 'first':
            output = output[:, 0, :]

        if return_recon:
            return output, None, embedding_after_attn
        else:
            return output, embedding_after_attn

    def return_embeddings(self, x, mode):
        # x must be tensor of elements (index)
        if mode == 1:
            return self.node_embedding1[x]
        else:
            return self.node_embedding2[x]

    def save_trained_embeddings(self, file_path):
        file = {"first_set_graph": self.node_embedding1, "second_set_graph": self.node_embedding2}
        torch.save(file, file_path)




# Decoder for HGBDHE model



class DirectedClassifier(nn.Module):
    """a classifier is the main model for embeddings"""

    def __init__(self, n_head, d_model, d_k, d_v, num_nodes, pad_idx, 
            emb_size, diag_mask, bottle_neck, softplus_layer,  **args):
        super().__init__()
        # global dir_cls_flag
        # dir_cls_flag = True
        print(args)

        self.softplus_layer = softplus_layer
        self.pff_classifier1 =  PositionwiseFeedForward([d_model, 1], reshape=True, use_bias=True)
        self.pff_classifier2 = PositionwiseFeedForward([d_model, 1], reshape=True, use_bias=True)
        self.pff_classifier3 = PositionwiseFeedForward([d_model, 1], reshape=True, use_bias=True)
        self.pff_classifier4 = PositionwiseFeedForward([d_model, 1], reshape=True, use_bias=True)
        self.pff_classifier5 = PositionwiseFeedForward([2, 1], reshape=True, use_bias=True)
        self.pff_classifier6 = PositionwiseFeedForward([2, 1], reshape=True, use_bias=True)
        
        
        self.node_embedding1 =  nn.Embedding(num_embeddings=num_nodes, embedding_dim=emb_size,padding_idx=pad_idx)
        self.node_embedding2 = self.node_embedding1
        # self.node_embedding2 =  nn.Embedding(num_embeddings=num_nodes, embedding_dim=emb_size,padding_idx=pad_idx)
        """remove positional embedding"""  ###########

        self.pff_u = PositionwiseFeedForward([d_model, d_model, d_model],
                                  dropout=0.4, residual=False, layer_norm=True)
        self.pff_v = PositionwiseFeedForward([d_model, d_model, d_model],
                                              dropout=0.4, residual=False, layer_norm=True)
        if args['cross_attn_type']  == 'x':
            model_init = OnlyCrossAttention
        elif args['cross_attn_type'] == 'sx':
            model_init = CrossAttentionSimple
        elif args['cross_attn_type'] == 'sxs':
            model_init = CrossAttention
        else:
            raise Exception('No Cross Attention Type specified.')
        self.encode1 = model_init(n_head, d_model, d_k, d_v, dropout_mul=0.4, dropout_pff=0.4,
                                      diag_mask=diag_mask, bottle_neck=bottle_neck)

        self.encode2 = EncoderLayer(
            4,
            d_model,
            d_model // 4,
            d_model // 4,
            dropout_mul=0.3,
            dropout_pff=0.4,
            diag_mask=True,
            bottle_neck=d_model)
        self.encode3 = EncoderLayer(
            4,
            d_model,
            d_model // 4,
            d_model // 4,
            dropout_mul=0.3,
            dropout_pff=0.4,
            diag_mask=True,
            bottle_neck=d_model)

        self.diag_mask_flag = diag_mask
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.layer_norm4 = nn.LayerNorm(d_model)
        self.layer_norm5 = nn.LayerNorm(d_model)
        self.layer_norm6 = nn.LayerNorm(d_model)

    def get_node_embeddings(self, x, mode, return_recon=False):

        # shape of x: (b, tuple)
        sz_b, len_seq = x.shape
        # print(torch.max(x), torch.min(x))
        if mode == 1:
            x = self.node_embedding1(x.view(-1))
        else:
            x = self.node_embedding2(x.view(-1))
        
        return x.view(sz_b, len_seq, -1)


    def get_embedding(self, x, y, crs_attn_mask1, crs_attn_mask2, slf_attn_mask1, slf_attn_mask2, non_pad_mask1,
                      non_pad_mask2, return_recon=False):

        if return_recon:
            x, recon_loss1 = self.get_node_embeddings(x, 1, return_recon)
            y, recon_loss2 = self.get_node_embeddings(y, 2, return_recon)
            recon_loss1 = None
            recon_loss2 = None
        else:
            x = self.get_node_embeddings(x, 1, return_recon)
            y = self.get_node_embeddings(y, 2, return_recon)
            recon_loss1, recon_loss2 = None, None
        dynamic1, static1, dynamic2, static2, output_attn = self.encode1(x, y, x, y, crs_attn_mask1, crs_attn_mask2,
                                                                         slf_attn_mask1, slf_attn_mask2, non_pad_mask1,
                                                                         non_pad_mask2)
        if return_recon:
            return x, y, dynamic1, static1, dynamic2, static2, output_attn, recon_loss1, recon_loss2
        else:
            return x, y, dynamic1, static1, dynamic2, static2, output_attn

    def forward(self, x, y, mask=None, get_outlier=None, return_recon=False):
        x = x.long()
        y = y.long()
        # pdb.set_trace()
        cr_attn_mask1 = get_attn_key_pad_mask(seq_k=y, seq_q=x)
        slf_attn_mask1 = get_attn_key_pad_mask(seq_k=x, seq_q=x)
        non_pad_mask1 = get_non_pad_mask(x)
        cr_attn_mask2 = get_attn_key_pad_mask(seq_k=x, seq_q=y)
        slf_attn_mask2 = get_attn_key_pad_mask(seq_k=y, seq_q=y)
        non_pad_mask2 = get_non_pad_mask(y)
        if return_recon:
            x_embed, y_embed, dynamic1, static1, dynamic2, static2, \
                recon_loss1, recon_loss2 = self.get_embedding(x , y, cr_attn_mask1, cr_attn_mask2,
                                                              slf_attn_mask1, slf_attn_mask2,
                                                              non_pad_mask1, non_pad_mask2, return_recon)
        else:
            x_embed, y_embed, dynamic1, static1, dynamic2, static2, \
                output_attn = self.get_embedding(x, y, cr_attn_mask1, cr_attn_mask2, slf_attn_mask1, slf_attn_mask2,
                                                 non_pad_mask1, non_pad_mask2, return_recon)
        dynamic1 = self.layer_norm1(dynamic1)
        dynamic2 = self.layer_norm3(dynamic2)
        dynamic1_2, static_, attn_ = self.encode2(x_embed + dynamic1, x_embed + dynamic1, slf_attn_mask1, non_pad_mask1)
        dynamic2_2, static_, attn_ = self.encode3(y_embed + dynamic2, y_embed + dynamic2, slf_attn_mask2, non_pad_mask2)

        static1 = self.layer_norm2(static1)
        static2 = self.layer_norm4(static2)
        dynamic1_2 =  self.layer_norm5(dynamic1_2 + dynamic1)
        dynamic2_2 =  self.layer_norm6(dynamic2_2 + dynamic2)
        sz_b, len_seq, dim = dynamic1.shape
        # pdb.set_trace()
        # output=torch.cat([((dynamic1-static1)**2),((dynamic2-static2)**2)],dim=1)
        dynamic1 = self.pff_u(dynamic1 * non_pad_mask1) * non_pad_mask1
        dynamic2 = self.pff_v(dynamic2 * non_pad_mask2) * non_pad_mask2
        output1 = self.pff_classifier1((dynamic1_2  - x_embed) **2 )
        output2 = self.pff_classifier2((dynamic2_2  - y_embed) **2 )
        output3 = self.pff_classifier3((dynamic1  - static1) ** 2)
        output4 = self.pff_classifier4((dynamic2  - static2) ** 2)
        # output = dynamic1**2+dynamic2**2
        # output1 = self.pff_classifier(dynamic1**2)
        # output2 = self.pff_classifier(dynamic2**2)
        #output = torch.cat([output1, output2], axis=1)
        #output = self.pff_classifier3(output)
        # pdb.set_trace()
        #output = torch.cat([torch.cat([output1, output2], axis=1), torch.cat([output3, output4], axis=1)], axis=2)
        output1 = self.pff_classifier5(torch.cat([output1, output3], axis=2))
        output2 = self.pff_classifier6(torch.cat([output2, output4], axis=2))
        if self.softplus_layer:
            output1 = F.softplus(output1)#torch.sigmoid(output)
            output2 = F.softplus(output2)#torch.sigmoid(output)
            #output3 = F.softplus(output3)#torch.sigmoid(output)
            #output4 = F.softplus(output4)#torch.sigmoid(output)
            #output = F.softplus(output)#torch.sigmoid(output)
        else:
            output1 = torch.sigmoid(output1)
            output2 = torch.sigmoid(output2)
            #output3 = torch.sigmoid(output3)
            #output4 = torch.sigmoid(output4)
            #output = torch.sigmoid(output)

        # embedding_after_attn = [[static1,dynamic1],[static2,dynamic2], output_attn]
        embedding_after_attn = None
        non_pad_mask = torch.cat([non_pad_mask1, non_pad_mask2], axis=1)
        '''
        if get_outlier is not None:
            k = get_outlier
            outlier = ((1 - output) * non_pad_mask).topk(k, dim=1, largest=True, sorted=True)[1]
            return outlier.view(-1, k)
        
        
        if mode == 'min':
            output, _ = torch.max(
                (1 - output) * non_pad_mask, dim=-2, keepdim=False)
            output = 1 - output
        '''
        mode = 'sum'
        if mode == 'sum':
            output1 = torch.sum(output1 * non_pad_mask1, dim=-2, keepdim=False)
            mask_sum1 = torch.sum(non_pad_mask1, dim=-2, keepdim=False)
            output1 /= mask_sum1
            output2 = torch.sum(output2 * non_pad_mask2, dim=-2, keepdim=False)
            mask_sum2 = torch.sum(non_pad_mask2, dim=-2, keepdim=False)
            output2 /= mask_sum2
            '''
            output3 = torch.sum(output3 * non_pad_mask1, dim=-2, keepdim=False)
            output3 /= mask_sum1
            output4 = torch.sum(output4 * non_pad_mask2, dim=-2, keepdim=False)
            output4 /= mask_sum2
            output = torch.sum(output * non_pad_mask, dim=-2, keepdim=False)
            mask_sum = torch.sum(non_pad_mask, dim=-2, keepdim=False)
            output /= mask_sum
            '''
            output = (output1 + output2)/2
        '''
        elif mode == 'first':
            output = output[:, 0, :]
        '''
        embedding_dynamic = (torch.cat([dynamic2, dynamic2_2, y_embed], dim=-1 ), torch.cat([dynamic1, dynamic1_2, x_embed], dim=-1) )
        if return_recon:
            return output, None, embedding_after_attn
        else:
            return output, embedding_dynamic
