from bdb import Breakpoint
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.nn.pool import knn
from torch_geometric.utils import grid, add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.conv import MessagePassing
from torch.autograd import Variable
from torch_geometric.nn import knn_graph, GraphConv, radius_graph
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_sparse import coalesce
import math
from torch.nn import LazyLinear
from torch import FloatTensor, LongTensor
from torch_cluster import knn, radius
from torch.nn import LayerNorm, BatchNorm1d

from torchsummary import summary

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return emb


class PositionalEncoding1D(nn.Module):
    # from https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
    # https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_x):
        """
        :param tensor: A 3d tensor of size (batch_size, ch)
        :return: Positional Encoding Matrix of size (batch_size, ch)
        """
        if len(pos_x.shape) != 1:
            raise RuntimeError("The input tensor has to be 1d!")
        
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)

        return emb_x


# the whole network GNN(dataset:HeteroData, edge_idxs:LongTensor<L>)->FloatTensor<L>:
class SpatialTemporalGNN(torch.nn.Module):
    # https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_DeepGCNs_Can_GCNs_Go_As_Deep_As_CNNs_ICCV_2019_paper.pdf
    def __init__(self, embed_size=64, layer_num=3, temporal_encoding=True, agent_identifier=True, **kwargs):
        super(SpatialTemporalGNN, self).__init__()
        self.g2g_edge_embed = Seq(LazyLinear(embed_size))
        self.p2p_edge_embed = Seq(LazyLinear(embed_size))
        self.g_layers = torch.nn.ModuleList()
        for _ in range(layer_num):
            self.g_layers.append(SimpleMPNN(embed_size=embed_size))
        self.p_layers = torch.nn.ModuleList()
        for _ in range(layer_num):
            self.p_layers.append(SimpleMPNN(embed_size=embed_size))
        self.global_token = nn.Parameter(torch.zeros(2*embed_size))        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=2*embed_size, dim_feedforward=2*embed_size, nhead=1)
        self.AttentionLayer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.PirorityLayer = Seq(LazyLinear(embed_size), ReLU(), Lin(embed_size, 1))
        
        self.embed_size = embed_size
        self.temporal_encoding = temporal_encoding
        self.agent_identifier = agent_identifier

    def forward(self, data:HeteroData, **kwargs) -> FloatTensor:
        data = data.clone()

        for g_layer in self.g_layers:
            data['g_node'].x = g_layer(data['g_node'].x, data['g_to_g'].edge_index, data['g_to_g'].edge_attr)

        g2p_index = data['g_to_p'].edge_index
        if self.temporal_encoding:
            data['p_node'].x[g2p_index[1, :]] = data['p_node'].x[g2p_index[1, :]] + data['g_node'].x[g2p_index[0, :]]
        else:
            data['p_node'].x[g2p_index[1, :]] = data['g_node'].x[g2p_index[0, :]]

        if self.agent_identifier:
            for p_layer in self.p_layers:
                data['p_node'].x = p_layer(data['p_node'].x, data['p_to_p'].edge_index, data['p_to_p'].edge_attr)

            batch_agent_id = data['problem_setting'].ptr[data['p_node'].batch]+data['p_node'].path[:, 0]
            global_feature = scatter_max(data['p_node'].x, batch_agent_id, dim_size=len(data['problem_setting'].batch), dim=0)[0]        
            data['p_node'].x = torch.cat((data['p_node'].x, global_feature[batch_agent_id]), dim=-1)
        
        # attention
        # turn into batches:
        batch_num = data['p_node'].batch.max()+1
        p_node_num = len(data['p_node'].x)

        device = data['p_node'].x.device
        # fill value into attention_feature and src_pad_mask
        idxs = (data['p_node'].batch, torch.arange(p_node_num).to(device) - data['p_node'].ptr[data['p_node'].batch])
        attention_feature = torch.zeros(batch_num, idxs[1].max()+2, 2*self.embed_size).float().to(device)
        attention_feature[idxs[0], 1+idxs[1], :] = data['p_node'].x
        attention_feature[:, 0, :] = self.global_token.unsqueeze(dim=0)
        src_pad_mask = torch.ones_like(attention_feature[:,:,0])
        src_pad_mask[:, 0] = 0
        src_pad_mask[idxs[0], 1+idxs[1]] = 0
        
        attention_feature = self.AttentionLayer(attention_feature.permute(1, 0, 2), src_key_padding_mask=src_pad_mask.bool())
        attention_feature = attention_feature[0, :, :]
        return self.PirorityLayer(attention_feature).squeeze(dim=-1)
        
        # global_path_feat = scatter_max(data['p_node'].x, data['p_node'].batch, dim_size=(1+data['p_node'].batch.max()), dim=0)[0]
        # return self.PirorityLayer(global_path_feat).squeeze(dim=-1)


class SimpleMPNN(MessagePassing):
    def __init__(self, embed_size, aggr: str = 'max', **kwargs):
        super(SimpleMPNN, self).__init__(aggr=aggr, **kwargs)
        self.fx = Seq(LazyLinear(embed_size), ReLU(), Lin(embed_size, embed_size))

    def forward(self, x, edge_index, edge_attr):
        """"""
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
        try:
            return x + out
        except:
            return out

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_j - x_i, x_j, x_i, edge_attr], dim=-1)
        values = self.fx(z)
        return values