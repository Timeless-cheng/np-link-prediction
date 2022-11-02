import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict

from torch_geometric.nn import RGCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.conv import MessagePassing
from attention import Attention

class EntityEmbedding(nn.Module):

    def __init__(self, entity_embedding_dim, relation_embedding_dim, num_entities, num_relations, args, entity_embedding, relation_embedding):

        super(EntityEmbedding, self).__init__()
        self.args = args

        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.entity_embedding = nn.Embedding(self.num_entities, self.entity_embedding_dim)
        self.relation_embedding = nn.Parameter(torch.Tensor(self.num_relations, self.relation_embedding_dim))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        if self.args.pre_train:
            self.entity_embedding.weight.data.copy_(entity_embedding.clone().detach())
            self.relation_embedding.data.copy_(relation_embedding.clone().detach())

            if not self.args.fine_tune:
                self.entity_embedding.weight.requires_grad = False
                self.relation_embedding.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        self.gnn = GENConv(self.entity_embedding_dim + self.relation_embedding_dim, self.entity_embedding_dim, self.num_relations * 2, num_bases = self.args.bases, root_weight = False, bias = False)

        self.score_function = self.args.score_function

    def forward(self, unseen_entity, triplets, use_cuda, total_unseen_entity_embedding = None):
        
        # Pre-process
        src, rel, dst = triplets.transpose()
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))   # indices to reconstruct the original array from unique values.

        unseen_index = np.where(uniq_v == unseen_entity)[0][0]
        rel_index = np.concatenate((rel, rel))

        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + self.num_relations))       # why add self.num_relations ??

        # Torch
        # node_id for all entities
        node_id = torch.LongTensor(uniq_v)
        edge_index = torch.stack((
            torch.LongTensor(src),
            torch.LongTensor(dst)
        ))
        edge_type = torch.LongTensor(rel)

        if use_cuda:
            node_id = node_id.cuda()
            edge_index = edge_index.cuda()
            edge_type = edge_type.cuda()

        # entity embeddings and relation embeddings
        x = self.entity_embedding(node_id)
        rel_emb = self.relation_embedding[rel_index]

        embeddings = self.gnn(x, edge_index, edge_type, rel_emb, edge_norm = None)
        unseen_entity_embedding = embeddings[unseen_index]
        unseen_entity_embedding = self.dropout(self.relu(unseen_entity_embedding))

        return unseen_entity_embedding

class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim, z_dim):
        
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    # def aggregate(self, r):
        
    #     return torch.mean(r, dim=0)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, few, r_dim)
        """
        # r = self.aggregate(r)
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return torch.distributions.Normal(mu, sigma)

class LatentEncoder(nn.Module):

    def __init__(self, embed_size=100, num_hidden1=500, num_hidden2=200, r_dim=100, dropout_p=0.5):
        super(LatentEncoder, self).__init__()
        self.embed_size = embed_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(3 * embed_size + 1, num_hidden1)),
            # ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden1, num_hidden2)),
            # ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, r_dim)),
            # ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, x):

        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)

        return x  # (B, few, r_dim)

class GENConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(GENConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(int(in_channels / 2), out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)


    def forward(self, x, edge_index, edge_type, rel_emb, edge_norm=None, size=None):
        """"""

        self.rel_emb = rel_emb

        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)


    def message(self, x_j, edge_index_j, edge_index_i, edge_type, edge_norm):
        
        # Concat node and relation embedding
        x_j = torch.cat((
            x_j,
            self.rel_emb
        ), dim = 1)

        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):

        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
            
        if (self.root is None) and (self.bias is None):
            return aggr_out

        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)

# class Decoder(nn.Module):

#     def __init__(self, args, embed_dim):
#         super(Decoder, self).__init__()
#         self.args = args

#         self.header_encoder = nn.Linear(embed_dim, embed_dim)
#         self.tail_encoder = nn.Linear(embed_dim, embed_dim)
    
#     def forward(self, query, z, score_function):
#         h = query[:, :100]
#         r = query[:, 100:200]
#         t = query[:, 200:300]

#         h = h + self.header_encoder(z)
#         t = t + self.tail_encoder(z)

#         if score_function == 'DistMult':
#             score = h * r * t
#             score = torch.sum(score, dim = 1)

#         return score

class Decoder(nn.Module):

    def __init__(self, args, embed_dim):
        super(Decoder, self).__init__()
        self.args = args

        self.enc_z = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, embed, z):

        return embed + self.enc_z(z)


# For Attention
class Deterministic_Encoder(nn.Module):
    '''
        Deterministic Encoder
    '''

    def __init__(self, input_dim, layer_sizes=[500, 200], num_latents=100, attention = None):
        '''
            Np deterministic encoder
            
            Args:
                input_dim : x_dim + y_dim
                layer_sizes : the array of each lyaer size in encoding MLP
                num_latents : the latent dimensionality
        '''

        super().__init__()

        # Attention
        self._attention = attention
        
        device = torch.device('cuda')
        embedding_type = 'mlp'
        layer_sizes = [100, 100]
        dim = {'q_last_dim' : 300, 'k_last_dim' : 300, \
                'v_last_dim' : 100}
        att_type = 'multihead'
        self._attention_model = Attention(device, embedding_type, layer_sizes, dim, att_type)
            
        # Layer size
        self._layer_sizes = layer_sizes

        # Output dim
        self._num_latents = num_latents

        # MLP embedding architectures
        self._layers = nn.ModuleList([])
        for i in range(len(self._layer_sizes)):
            if i == 0:
                self._layers.append(nn.Linear(input_dim*3+1, self._layer_sizes[0]))
            else:
                self._layers.append(nn.Linear(self._layer_sizes[i-1], self._layer_sizes[i]))

        self._last_layer = nn.Linear(self._layer_sizes[-1], self._num_latents)

    def forward(self, encoder_input, target_x=None):
        '''
            Ecoding the input into representation using latent encoder

            Args:
                context_x: [batch_size, the number of observation, x_size(dimension)] 
                context_y : [batch_size, the number of observation, y_size(dimension)] 

            Returns:
                representation : [batch_size, the nmber of observation, num_lantents]
        '''

        # task_size, _, filter_size = tuple(encoder_input.shape)
        input_shape = tuple(encoder_input.shape)

        # Input
        # hidden = encoder_input.view((-1, filter_size))
        hidden = encoder_input.view((-1, input_shape[-1]))

        # MLP embedidng for NP
        for i, layer in enumerate(self._layers):
            hidden = F.relu(layer(hidden))

        # Last layer
        hidden = self._last_layer(hidden)

        # Reshaping
        # hidden = hidden.view((task_size, -1, self._num_latents)) # [batch_size, the number of point, the last element in the list]
        # [1200, 128] --> 
        hidden = hidden.view((*input_shape[:-2], -1, self._num_latents)) # [(n_sample, batch_size), the number of point, the last element in the list]

        if self._attention is not None:
            # Attentive neural process
            context_x = encoder_input[:, :, :-1]
            rep, weights = self._attention_model(target_x[:, :, :-1], context_x, hidden)
            
        else:
            # Conditional neural processes
            # Aggregation of representation
            rep = hidden.mean(dim=-2)
            weights = None

        return rep, weights 
