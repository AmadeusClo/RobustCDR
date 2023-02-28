import numpy as np
import dgl
import dgl.function as GF
import torch.nn as nn
import torch
import torch.nn.functional as F

class NodeGATLayer(nn.Module):
    def __init__(self, node_embed_size):
        super(NodeGATLayer, self).__init__()
        self.fc1 = nn.Linear(node_embed_size, node_embed_size, bias=False)
        self.fc2 = nn.Linear(node_embed_size, 1, bias=False)
    
    def message_func(self, edges):
        return {'e' : self.fc2(torch.tanh(self.fc1(edges.src['N']))),
                'N' : edges.src['N']}
        
    def reduce_func(self, nodes):
        a = F.softmax(nodes.mailbox['e'], dim=1)
        return {'Np' : torch.sum(a * nodes.mailbox['N'], dim=1)}
        
    def forward(self, g, N):
        with g.local_scope():
            g.ndata['N'] = N
            g.update_all(self.message_func, self.reduce_func)
            return g.ndata['Np']
        
class EdgeGATLayer(nn.Module):
    def __init__(self, edge_embed_size, node_embed_size):
        super(EdgeGATLayer, self).__init__()
        self.fc1 = nn.Linear(edge_embed_size, edge_embed_size, bias=False)
        self.fc2 = nn.Linear(edge_embed_size, 1, bias=False)
        self.fc3 = nn.Linear(edge_embed_size, node_embed_size, bias=False)
    
    def forward(self, g, E):
        with g.local_scope():
            g.edata['E'] = E
            g.multi_update_all({'interact' : (GF.copy_e('E', 'tmp_I'), GF.mean('tmp_I', 'YI')),
                                'similar' : (GF.copy_e('E', 'tmp_S'), GF.mean('tmp_S', 'YS'))}, 'sum')
            YI = g.nodes['node'].data['YI']
            YS = g.nodes['node'].data['YS']
            eI = self.fc2(torch.tanh(self.fc1(YI)))
            eS = self.fc2(torch.tanh(self.fc1(YS)))
            a = F.softmax(torch.cat((eI, eS), dim=1), dim=1).reshape(-1, 2, 1)
            OY = (a * torch.stack((YI, YS), dim=1)).sum(dim=1)
            return self.fc3(OY)
        
class ReCDR(nn.Module):
    def __init__(self, domains, train_mat, train_mat_local, exp_src, exp_dst, args, device):
        super(ReCDR, self).__init__()
        self.domains = domains

        self.num_users, self.num_items = train_mat.shape
        inter_user, inter_item = train_mat.nonzero()
        inter_user = torch.tensor(inter_user).long()
        inter_item = torch.tensor(inter_item).long() + self.num_users
        self.num_nodes = self.num_users + self.num_items

        # ********** Building Graph **********
        # global graph for node aggregation
        self.ng = dgl.graph(data = (np.concatenate((inter_user, exp_src)), np.concatenate((inter_item, exp_dst))),
                            idtype = torch.int32,
                            num_nodes = self.num_nodes,
                            device = 'cpu')
        self.ng = dgl.to_bidirected(self.ng).to(device)
        print("ng :", self.ng)

        # global heterograph for edge aggregation
        self.eg = dgl.heterograph(data_dict = { ('node','interact','node') : (inter_user, inter_item),
                                                ('node','similar','node') : (exp_src, exp_dst)},
                                  idtype = torch.int32,
                                  num_nodes_dict = { 'node' : self.num_nodes },
                                  device = 'cpu')
        self.eg = dgl.to_bidirected(self.eg).to(device)
        print("eg :", self.eg)

        # local graphs for each domain
        self.local_g = {}
        for domain in domains:
            local_user, local_item = train_mat_local[domain].nonzero()
            local_user = torch.tensor(local_user).long()
            local_item = torch.tensor(local_item).long() + self.num_users # item node ID = column ID + num_users
            self.local_g[domain] = dgl.graph(data = (local_user, local_item),
                                             idtype = torch.int32,
                                             num_nodes = self.num_nodes,
                                             device = 'cpu')
            self.local_g[domain] = dgl.to_bidirected(self.local_g[domain]).to(device)
        print("local_g :", self.local_g)

        # ********** Model Parameters **********
        self.num_edges = self.eg.num_edges('interact') + self.eg.num_edges('similar')
        
        self.local_node_embed = nn.ModuleDict()
        self.local_node_agg = nn.ModuleDict()
        for domain in domains:
            self.local_node_embed[domain] = nn.Embedding(self.num_nodes, args.node_embed)
            self.local_node_agg[domain] = NodeGATLayer(args.node_embed)
        
        self.global_node_embed = nn.Embedding(self.num_nodes, args.node_embed)
        self.global_node_agg = NodeGATLayer(args.node_embed)

        self.global_edge_embed = nn.Embedding(self.num_edges, args.edge_embed)
        self.global_edge_agg = EdgeGATLayer(args.edge_embed, args.node_embed)
    
        self.Ws = nn.Parameter(torch.FloatTensor(size=(self.num_nodes, args.node_embed)))

        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, domain):
        assert domain in self.domains, "Invalid domain ID"
        # local node embedding
        local_Np = self.local_node_agg[domain](self.local_g[domain], self.local_node_embed[domain].weight)
        
        # global embedding
        # node
        Np = self.global_node_agg(self.ng, self.global_node_embed.weight)
        # edge
        num_interact = self.eg.num_edges('interact')
        F = self.global_edge_agg(self.eg, E = {'interact' : self.global_edge_embed.weight[:num_interact],
                                               'similar'  : self.global_edge_embed.weight[num_interact:]})
        # fusion
        S = self.Ws * F + Np
#        S = Np # no edge features
        
        # global & local fusion
        H = torch.cat((local_Np, S), dim=1)
        return H[:self.num_users], H[self.num_users:]