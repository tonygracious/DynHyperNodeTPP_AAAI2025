from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, LayerNorm
from torch_scatter import scatter_add

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax


class BiHypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}
    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W} \in \mathbb{R}^M` is the diagonal hyperedge
    weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.
    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:
    .. code-block:: python
        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])
    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          hyperedge indices :math:`(|\mathcal{V}|, |\mathcal{E}|)`,
          hyperedge weights :math:`(|\mathcal{E}|)` *(optional)*
          hyperedge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin_u = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')
            self.lin_v = Linear(in_channels, out_channels, bias=False,
                                weight_initializer='glorot')

            self.proj_u = Linear(2* out_channels, out_channels, bias=False,
                              weight_initializer='glorot')
            self.proj_v = Linear(2* out_channels, out_channels, bias=False,
                              weight_initializer='glorot')

            self.proj_u_output = Linear(2 * out_channels, out_channels, bias=False,
                                weight_initializer='glorot')
            self.proj_v_output = Linear(2 * out_channels, out_channels, bias=False,
                                 weight_initializer='glorot')
            self.LayerNorm =  LayerNorm(out_channels*2)
        if bias and concat:
            self.bias_u = Parameter(torch.Tensor(heads * out_channels))
            self.bias_v = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias_u = Parameter(torch.Tensor(out_channels))
            self.bias_v = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_u.reset_parameters()
        self.lin_v.reset_parameters()
        self.proj_u.reset_parameters()
        self.proj_v.reset_parameters()
        self.proj_u_output.reset_parameters()
        self.proj_v_output.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias_u)
        zeros(self.bias_v)


    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        hyperedge_index_u, hyperedge_index_v = hyperedge_index
        if hyperedge_index_u.numel() > 0:
            num_edges = int(hyperedge_index_u[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x_u = self.lin_u(x)
        x_v = self.lin_v(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D_u = scatter_add(hyperedge_weight[hyperedge_index_u[1]],
                        hyperedge_index_u[0], dim=0, dim_size=num_nodes)
        D_u = 1.0 / D_u
        D_u[D_u == float("inf")] = 0

        D_v = scatter_add(hyperedge_weight[hyperedge_index_v[1]],
                          hyperedge_index_v[0], dim=0, dim_size=num_nodes)
        D_v = 1.0 / D_v
        D_v[D_v == float("inf")] = 0


        B_u = scatter_add(x.new_ones(hyperedge_index_u.size(1)),
                        hyperedge_index_u[1], dim=0, dim_size=num_edges)
        B_u = 1.0 / B_u
        B_u[B_u == float("inf")] = 0

        B_v = scatter_add(x.new_ones(hyperedge_index_v.size(1)),
                          hyperedge_index_v[1], dim=0, dim_size=num_edges)
        B_v = 1.0 / B_v
        B_v[B_v == float("inf")] = 0

        out_u = self.propagate(hyperedge_index_u, x=x_u, norm=B_u, alpha=alpha,
                             size=(num_nodes, num_edges))
        out_v = self.propagate(hyperedge_index_v, x=x_v, norm=B_v, alpha=alpha,
                               size=(num_nodes, num_edges))
        out = self.proj_u(torch.tanh(  self.LayerNorm( torch.cat([out_u, out_v], dim=-1)) ))

        return out


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out


class HyperEdgeEmbed(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W} \in \mathbb{R}^M` is the diagonal hyperedge
    weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.

    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:

    .. code-block:: python

        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          hyperedge indices :math:`(|\mathcal{V}|, |\mathcal{E}|)`,
          hyperedge weights :math:`(|\mathcal{E}|)` *(optional)*
          hyperedge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

if __name__ == '__main__':
    hgcn = BiHypergraphConv(10, 10)
    X  = torch.rand((20, 10))
    hyperedge_index_u = torch.tensor([ [0, 1, 2, 1, 2, 3], [0, 0, 0, 1, 1, 1]])
    hyperedge_index_v = torch.tensor([ [5, 6, 7, 11, 12, 13], [0, 0, 0, 1, 1, 1]])
    hyperedge_index = (hyperedge_index_u, hyperedge_index_v)
    x = hgcn(X, hyperedge_index)