from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix

class BPRDataset(Dataset): # dataset for specific domain
    def __init__(self, data, train_mat=None, item_list=None, num_ng=1, is_training=None):
        '''
        @param data [num_samples][1 + 1 + num_ng]
        @param train_mat : coo_matrix for the domain
        @param item_list : item ID list of the domain
        @param num_ng : number of negative samples for each positive sample
        @param is_training : boolean
        '''
        super(BPRDataset, self).__init__()
        self.data = np.array(data)
        self.train_mat = train_mat
        self.num_items = len(item_list)
        self.item_list = item_list
        self.num_ng = num_ng
        self.is_training = is_training
    
    def ng_sample(self):
        assert self.is_training, "no need for negative sample when testing"
        train_mat = self.train_mat.todok()
        length = self.data.shape[0]

        # pick negative samples randomly
        self.neg_data = np.random.randint(low=0, high=self.num_items, size=(length, self.num_ng))
        for i in range(length):
            for j in range(self.num_ng):
                uid = self.data[i, 0]
                jidid = self.neg_data[i, j]
                while (uid, self.item_list[jidid]) in train_mat:
                    jidid = np.random.randint(low=0, high=self.num_items)
                self.neg_data[i, j] = self.item_list[jidid]
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        uid = self.data[idx][0]
        iid = self.data[idx][1]
        if self.is_training:
            jids = self.neg_data[idx]
            return uid, iid, jids
        else:
            return uid, iid