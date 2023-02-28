import torch.nn as nn
import torch
import torch.nn.functional as F
import dgl

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc_att = nn.Linear(in_dim*2, 1, bias=False)
        self.fc_upd = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_V = nn.Linear(in_dim, out_dim)
    
    def message_func(self, edges):
        qq = torch.cat([edges.src['q'], edges.dst['q']], dim=1)
        return {'K' : edges.src['q'], 'e' : torch.sigmoid(self.fc_att(qq)), 'V' : self.fc_V(edges.src['q'])}
        
    def reduce_func(self, nodes):
        a = F.softmax(nodes.mailbox['e'], dim=1)
        OK = torch.sum(a * nodes.mailbox['K'], dim=1)
        OV = torch.sum(a * nodes.mailbox['V'], dim=1)
        return {'q_new' : self.fc_upd(nodes.data['q'] + OK),
                'OV' : OV}
        
    def forward(self, g, q):
        with g.local_scope():
            g.ndata['q'] = q
            g.multi_update_all({'review':(self.message_func, self.reduce_func),
                                'review_by':(self.message_func, self.reduce_func)}, 'sum')
            return g.ndata['q_new'], g.ndata['OV']
            
class HeroGRAPH(nn.Module):
    def __init__(self, domains, train_mat, embed_size, device):
        super(HeroGRAPH, self).__init__()
        self.domains = domains
        num_users, num_items = train_mat.shape
        self.local_user_embed = nn.ModuleDict()
        self.local_item_embed = nn.ModuleDict()
        for domain in domains:
            self.local_user_embed[domain] = nn.Embedding(num_users, embed_size)
            self.local_item_embed[domain] = nn.Embedding(num_items, embed_size)
        self.global_user_embed = nn.Embedding(num_users, embed_size)
        self.global_item_embed = nn.Embedding(num_items, embed_size)
        self.att1 = GATLayer(embed_size, 64)
        self.att2 = GATLayer(64, 16)
        self.fc = nn.Linear(16*2, embed_size)
        self.reset_parameters()
        
        # building graph
        train_edge_src, train_edge_dst = train_mat.nonzero()
        edges = {
            ('user','review','item') : (train_edge_src, train_edge_dst),
            ('item','review_by','user') : (train_edge_dst, train_edge_src)
        }
        nodes = {
            'user' : num_users,
            'item' : num_items
        }
        self.g = dgl.heterograph(data_dict = edges, idtype=torch.int32, num_nodes_dict=nodes, device=device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, domain):
        assert domain in self.domains, f"Invalid domain {domain}"
        # local embedding
        E_u = self.local_user_embed[domain].weight
        E_i = self.local_item_embed[domain].weight
        
        # global embedding
        q = {'user' : self.global_user_embed.weight,
             'item' : self.global_item_embed.weight}
        q, _ = self.att1(self.g, q)
        q, OV = self.att2(self.g, q)
        G_u = F.relu(self.fc(torch.cat((q['user'], OV['user']), dim=1)))
        G_i = F.relu(self.fc(torch.cat((q['item'], OV['item']), dim=1)))
        
        return torch.cat((E_u, G_u), dim=1), torch.cat((E_i, G_i), dim=1)