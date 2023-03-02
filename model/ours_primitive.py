import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as gnn
import dgl.function as GF

from math import sqrt

class LocalNodeGATLayer(nn.Module):
    def __init__(self, node_embed_size):
        super(LocalNodeGATLayer, self).__init__()
        self.fc1 = nn.Linear(node_embed_size, node_embed_size, bias=False)
        self.fc2 = nn.Linear(node_embed_size, 1, bias=False)
    
    def message_func(self, edges):
        return {'e' : self.fc2(torch.tanh(self.fc1(edges.src['N']))),
                      #F.leaky_relu(self.fc2(torch.cat((self.fc1(edges.src['N']), self.fc1(edges.dst['N'])), dim=1))),
                'N' : edges.src['N']}
        
    def reduce_func(self, nodes):
        a = F.softmax(nodes.mailbox['e'], dim=1)
        return {'Np' : torch.sum(a * nodes.mailbox['N'], dim=1)}
        
    def forward(self, g, N):
        with g.local_scope():
            g.ndata['N'] = N
            g.update_all(self.message_func, self.reduce_func)
            return g.ndata['Np']

class NodeGATLayer(nn.Module):
    def __init__(self, node_embed_size, domains):
        super(NodeGATLayer, self).__init__()
        self.domains = domains
        self.num_domains = len(domains)
        self.node_embed_size = node_embed_size
        self.fc1 = nn.ModuleList()
        self.fc2 = nn.ModuleList()
        for i in range(self.num_domains): # LeakyReLU(a^T [Wh_i||Wh_j])
            self.fc1.append(nn.Linear(node_embed_size, node_embed_size, bias=False))
            self.fc2.append(nn.Linear(node_embed_size*2, 1, bias=False))
        self.Wq = nn.Linear(node_embed_size, node_embed_size, bias=False)
        self.Wk = nn.Linear(node_embed_size, node_embed_size, bias=False)
        self.Wv= nn.Linear(node_embed_size, node_embed_size, bias=False)
    
    def mes_func_0(self, edges):
        return {'e_CDs_and_Vinyl'      : F.leaky_relu(self.fc2[0](torch.cat((self.fc1[0](edges.src['N']), self.fc1[0](edges.dst['N'])), dim=1))),
                             #self.fc2[0](torch.tanh(self.fc1[0](edges.src['N']))),
                'N_CDs_and_Vinyl'      : edges.src['N']}
    def mes_func_1(self, edges):
        return {'e_Digital_Music'      : F.leaky_relu(self.fc2[1](torch.cat((self.fc1[1](edges.src['N']), self.fc1[1](edges.dst['N'])), dim=1))),
                             #self.fc2[1](torch.tanh(self.fc1[1](edges.src['N']))),
                'N_Digital_Music'      : edges.src['N']}
    def mes_func_2(self, edges):
        return {'e_Musical_Instruments': F.leaky_relu(self.fc2[2](torch.cat((self.fc1[2](edges.src['N']), self.fc1[2](edges.dst['N'])), dim=1))),
                             #self.fc2[1](torch.tanh(self.fc1[1](edges.src['N']))),
                'N_Musical_Instruments': edges.src['N']}
    
    def red_func_0(self, nodes):
        a = F.softmax(nodes.mailbox['e_CDs_and_Vinyl'      ], dim=1)
        return {'CDs_and_Vinyl'      : torch.sum(a * nodes.mailbox['N_CDs_and_Vinyl'      ], dim=1)}
    def red_func_1(self, nodes):
        a = F.softmax(nodes.mailbox['e_Digital_Music'      ], dim=1)
        return {'Digital_Music'      : torch.sum(a * nodes.mailbox['N_Digital_Music'      ], dim=1)}
    def red_func_2(self, nodes):
        a = F.softmax(nodes.mailbox['e_Musical_Instruments'], dim=1)
        return {'Musical_Instruments': torch.sum(a * nodes.mailbox['N_Musical_Instruments'], dim=1)}
    
    def forward(self, g, N, target_domain, device):#, local_Np):
        with g.local_scope():
            g.ndata['N'] = N
#            funcs = {}
#            for i, domain in enumerate(self.domains):
#                funcs[domain] = (lambda edges: {f'e_{domain}': self.fc2[i](torch.tanh(self.fc1[i](edges.src['N']))),
#                                                f'N_{domain}': edges.src['N']},
#                                 lambda nodes: {f'{domain}': torch.sum(F.softmax(nodes.mailbox[f'e_{domain}'], dim=1) * nodes.mailbox[f'N_{domain}'], dim=1)})
            funcs = {'CDs_and_Vinyl'      :(self.mes_func_0, self.red_func_0),
                     'Digital_Music'      :(self.mes_func_1, self.red_func_1),
                     'Musical_Instruments':(self.mes_func_2, self.red_func_2)}
            
            g.multi_update_all(funcs, 'sum')
            e = torch.empty((g.num_nodes('node'), len(self.domains))).to(device)
            
            for i, domain in enumerate(self.domains):
                query = self.Wq(g.nodes['node'].data[target_domain]) # 换成 local graph 的信息如何？
               # query = self.Wq(local_Np)
                key = self.Wk(g.nodes['node'].data[domain])
                value = self.Wv(g.nodes['node'].data[domain])
                e[:, i] =  torch.sum(query * key, dim=1) / sqrt(query.shape[1])
            a = F.softmax(e, dim=1)
            Np = torch.zeros((g.num_nodes('node'), self.node_embed_size)).to(device)
            for i, domain in enumerate(self.domains):
                Np += a[:, i].reshape(-1, 1) * g.ndata[domain]#value#
            return Np
        
class EdgeGATLayer(nn.Module):
    def __init__(self, edge_embed_size, node_embed_size):
        super(EdgeGATLayer, self).__init__()
        self.fc1 = nn.Linear(edge_embed_size, edge_embed_size, bias=False)
        self.fc2 = nn.Linear(edge_embed_size, 1, bias=False)
        self.fc3 = nn.Linear(edge_embed_size, node_embed_size, bias=False)
    
    def forward(self, g, E):
        with g.local_scope():
            g.edata['E'] = E
            g.multi_update_all({'CDs_and_Vinyl'      : (GF.copy_e('E', 'tmp_C'), GF.mean('tmp_C', 'YC')),
                                'Digital_Music'      : (GF.copy_e('E', 'tmp_D'), GF.mean('tmp_D', 'YD')),
                                'Musical_Instruments': (GF.copy_e('E', 'tmp_M'), GF.mean('tmp_M', 'YM'))}, 'sum')
            YC = g.nodes['node'].data['YC']
            YD = g.nodes['node'].data['YD']
            YM = g.nodes['node'].data['YM']
            eC = self.fc2(torch.tanh(self.fc1(YC)))
            eD = self.fc2(torch.tanh(self.fc1(YD)))
            eM = self.fc2(torch.tanh(self.fc1(YM)))
            
            a = F.softmax(torch.cat((eC, eD, eM), dim=1), dim=1).reshape(-1, 3, 1)
            OY = (a * torch.stack((YC, YD, YM), dim=1)).sum(dim=1)
            return self.fc3(OY)
        
class ReCDREdgeGATLayer(nn.Module):
    def __init__(self, edge_embed_size, node_embed_size):
        super(ReCDREdgeGATLayer, self).__init__()
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

class ours(nn.Module):
    def __init__(self, domains, train_mat, train_mat_local, exp_src, exp_dst, args, device):
        super(ours, self).__init__()
        self.domains = domains
        self.device = device

        self.num_users, self.num_items = train_mat.shape
        inter_user, inter_item = train_mat.nonzero()
        inter_user = torch.tensor(inter_user).long()
        inter_item = torch.tensor(inter_item).long() + self.num_users
        self.num_nodes = self.num_users + self.num_items

        # ********** Building Graph **********
        # global heterograph for node aggregation
        data_dict = {}
        for domain in domains:
            local_user, local_item = train_mat_local[domain].nonzero()
            local_user = np.array(local_user)
            local_item = np.array(local_item) + self.num_users
            data_dict[('node', domain, 'node')] = (local_user, local_item)
        self.ng = dgl.heterograph(data_dict = data_dict,
                                  idtype = torch.int32,
                                  num_nodes_dict = { 'node' : self.num_nodes },
                                  device = 'cpu')
        self.ng = dgl.to_bidirected(self.ng).to(device)

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
#        self.num_edges = 0
#        for domain in domains:
#            self.num_edges += self.eg.num_edges(domain)
        self.num_edges = self.eg.num_edges('interact') + self.eg.num_edges('similar')

        self.local_node_embed = nn.ModuleDict()
        self.local_node_agg1 = nn.ModuleDict()
#        self.local_node_agg2 = nn.ModuleDict()
        for domain in domains:
            self.local_node_embed[domain] = nn.Embedding(self.num_nodes, args.node_embed)
            self.local_node_agg1[domain] = LocalNodeGATLayer(args.node_embed)
#            self.local_node_agg2[domain] = LocalNodeGATLayer(args.node_embed)
        
        self.global_node_embed = nn.Embedding(self.num_nodes, args.node_embed)
        self.global_node_agg1 = NodeGATLayer(args.node_embed, domains)
#        self.global_node_agg2 = NodeGATLayer(args.node_embed, domains)
        
        self.global_edge_embed = nn.Embedding(self.num_edges, args.edge_embed)
        #self.global_edge_agg = EdgeGATLayer(edge_embed_size, node_embed_size)
        self.global_edge_agg = ReCDREdgeGATLayer(args.edge_embed, args.node_embed)
    
        self.Ws = nn.Parameter(torch.FloatTensor(size=(self.num_nodes, args.node_embed)))
        self.reset_parameters()
#        self.dropout = nn.Dropout(p=0.2)
    
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, target_domain):
        # local embedding
        local_emb_0 = self.local_node_embed[target_domain].weight
        local_Np = self.local_node_agg1[target_domain](self.local_g[target_domain], local_emb_0)
#        local_emb_1 = local_emb_0 + (self.local_node_agg1[dom_id](local_g, self.dropout(local_emb_0)))
#        local_emb_2 = local_emb_1 + (self.local_node_agg2[dom_id](local_g, self.dropout(local_emb_1)))
        
        # global embedding
        # node
        global_emb_0 = self.global_node_embed.weight
        Np = self.global_node_agg1(self.ng, global_emb_0, target_domain, self.device)
#        global_emb_1 = global_emb_0 + (self.global_node_agg1(ng, self.dropout(global_emb_0), dom_id))#, local_Np)
#        global_emb_2 = global_emb_1 + (self.global_node_agg2(ng, self.dropout(global_emb_1), dom_id))
        
        # edge
#        E = {}
#        tmp = 0
#        for i, domain in enumerate(abbr_domains):
#        num_interact = eg.num_edges(domain)
#            E[domain] = self.global_edge_embed.weight[tmp : tmp + num_interact]
#            tmp += num_interact

        num_interact = self.eg.num_edges('interact')
        E = {'interact': self.global_edge_embed.weight[:num_interact],
             'similar' : self.global_edge_embed.weight[num_interact:]}
        Ep = self.global_edge_agg(self.eg, E)
        S = self.Ws * Ep + Np# * (1-dens_norm[dom_id]) # density debias
        
        # fusion
        H = torch.cat((local_Np, S), dim=1)
#        H = torch.cat((local_emb_0, local_emb_1, local_emb_2, global_emb_0, global_emb_1, global_emb_2), dim=1)
#        H = torch.cat((local_emb_0 + local_emb_1 + local_emb_2, global_emb_0 + global_emb_1 + global_emb_2), dim=1)
        return H[:self.num_users], H[self.num_users:]
