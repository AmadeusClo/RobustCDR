import torch.nn as nn

class BPRMF(nn.Module):
    def __init__(self, domains, train_mat, embed_size):
        super(BPRMF, self).__init__()
        self.domains = domains
        num_users, num_items = train_mat.shape
        self.user_embed = nn.ModuleDict()
        self.item_embed = nn.ModuleDict()
        for domain in domains:
            self.user_embed[domain] = nn.Embedding(num_users, embed_size)
            self.item_embed[domain] = nn.Embedding(num_items, embed_size)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, domain):
        assert domain in self.domains, f"Invalid domain [{domain}]"
        return self.user_embed[domain].weight, self.item_embed[domain].weight