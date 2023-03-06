import numpy as np
import torch
import os
from utils.logger import log

def hit(gt_item, top_item):
    if gt_item in top_item:
        return 1
    return 0

def ndcg(gt_item, top_item):
    if gt_item in top_item:
        index = top_item.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0

def eval_domain(model, data, domain, device, num_ng_eval=99, topk=5):
    HR, NDCG = [], []
    user = torch.tensor(data[:,0]).squeeze().to(device).long()
    item = torch.tensor(data[:,1]).squeeze().to(device).long()
    
    embedding_user, embeeding_item = model(domain)
    pred = (embedding_user[user] * embeeding_item[item]).sum(dim=1)
    
    batch_size = int(1 + num_ng_eval)
    assert user.shape[0] % batch_size == 0
    batch_num = int(user.shape[0] / batch_size)
    for i in range(batch_num):
        batch_pred = pred[i*batch_size : (i+1)*batch_size].squeeze()
        _, indices = torch.topk(batch_pred, topk)
        batch_item = item[i*batch_size : (i+1)*batch_size].squeeze()
        top_item = torch.take(batch_item, indices).cpu().numpy().tolist()
        gt_item = batch_item[0].item()
        HR.append(hit(gt_item, top_item))
        NDCG.append(ndcg(gt_item, top_item))
    return np.mean(HR), np.mean(NDCG)

def eval(model, device, num_ng_eval, topk, train_loss, domains, eval_data, log_path):
    HR, NDCG = {}, {}
    for domain in domains:
        HR_dom, NDCG_dom = eval_domain(model, eval_data[domain], domain, device, num_ng_eval, topk)
#        print(f'[domain {domain}] HR@{topk} = {HR_dom}, NDCG@{topk} = {NDCG_dom}, loss = {train_loss[domain][-1]}')
        if train_loss == None:
            print(f'[domain {domain}] HR@{topk} = {HR_dom}, NDCG@{topk} = {NDCG_dom}')
        else:
            log(f'[domain {domain}] HR@{topk} = {HR_dom}, NDCG@{topk} = {NDCG_dom}, loss = {train_loss[domain][-1]}', log_path)
        HR[domain] = HR_dom
        NDCG[domain] = NDCG_dom
    return HR, NDCG