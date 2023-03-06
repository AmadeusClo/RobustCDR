import torch
from torch.utils.data import dataloader
from torch_geometric.nn import Node2Vec

import scipy.sparse as sp

import numpy as np
from tqdm import tqdm
import random
import os
import sys
from math import sqrt, ceil
import argparse
import matplotlib.pyplot as plt
import time
import pickle

from dataset.BPRDataset import BPRDataset
from utils.evaluation import eval
from utils.logger import log

from model.BPRMF import BPRMF
from model.HeroGRAPH import HeroGRAPH
from model.ReCDR import ReCDR
from model.ours import ours

def rel_exp(train_mat, args, device):
    num_users, num_items = train_mat.shape
    train_edge_src, train_edge_dst = train_mat.nonzero()
    train_edge_src = torch.tensor(train_edge_src).long()
    train_edge_dst = torch.tensor(train_edge_dst).long() + num_users

    edge_index = torch.stack([torch.cat((train_edge_src, train_edge_dst)), 
                              torch.cat((train_edge_dst, train_edge_src))])
    n2v_model = Node2Vec(edge_index, embedding_dim=args.n2v_embed, walk_length=20,
                         context_size=10, walks_per_node=10,
                         num_negative_samples=1, p=1, q=2, sparse=True).to(device)
    num_workers = 0 if sys.platform.startswith('win') else 4
    loader = n2v_model.loader(batch_size=128, shuffle=True,
                              num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(n2v_model.parameters()), lr=0.01)
    
    print("Training Node2Vec model...")
    for epoch in range(1, args.n2v_epoch + 1):
        n2v_model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = n2v_model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss / len(loader)
        print(f'Epoch: {epoch:02d}, Loss: {total_loss:.4f}')
    
    # calculate similarity and generate expansion edges
    with torch.no_grad():
        S = n2v_model()
        block_size = int(sqrt(S.shape[0]))
        block_num = ceil(S.shape[0]*1./block_size)
        exp_src, exp_dst = [], []
        norm_S = (S / torch.sqrt((S * S).sum(dim=1)).reshape(-1, 1)).to(device)
        print("Generating expansion edges according to similarity...")
        for i in tqdm(range(block_num)):
            similarity = norm_S[block_size*i:block_size*(i+1), :] @ norm_S.T
            for j in range(block_size):
                if j + block_size*i >= S.shape[0]:break
                similarity[j, j + block_size*i] = 0
            rel_exp = sp.coo_matrix(similarity.cpu() > args.n2v_alpha)
            local_exp_src, local_exp_dst = rel_exp.nonzero()
            exp_src.extend(local_exp_src + block_size*i)
            exp_dst.extend(local_exp_dst)
    return exp_src, exp_dst

def get_model(train_mat, train_mat_local, domains, args, device):
    if args.model == 'BPRMF':
        return BPRMF(domains, train_mat, args.node_embed)
    elif args.model == 'HeroGRAPH':
        return HeroGRAPH(domains, train_mat, args.node_embed, device)
    elif args.model == 'ReCDR':
        exp_src, exp_dst = rel_exp(train_mat, args, device)
        return ReCDR(domains, train_mat, train_mat_local, exp_src, exp_dst, args, device)
    elif args.model == 'ours':
        exp_src, exp_dst = rel_exp(train_mat, args, device)
        return ours(domains, train_mat, train_mat_local, exp_src, exp_dst, args, device)
    else:
        raise NotImplementedError
    
#def adjust_learning_rate(optimizer, lr):
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr

def adjust_learning_rate(optimizer, args):
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(param_group['lr'] * args.decay, args.min_lr)

def save_history(train_loss, HR, NDCG, HR_ave_best, epoch_best, args, exp_name, exp_path):
    his = {
        'loss' : train_loss,
        'HR': HR,
        'NDCG': NDCG,
        'HR_ave_best' : HR_ave_best,
        'epoch_best' : epoch_best,
        'args': args
    }
    with open(os.path.join(exp_path, exp_name + '.his'), 'wb') as outfile:
        pickle.dump(his, outfile)
    return his

def save_model(model, opt, epoch, his, ckpt_name, exp_path):
    params = {
        'epoch' : epoch,
        'model' : model,
        'opt' : opt,
        'args' : args,
        'his' : his
    }
    save_path = os.path.join(exp_path, ckpt_name + '.ckpt')
    torch.save(params, save_path)

def train(args, device):
    local_time = time.localtime()
    timestr = f"{local_time.tm_year:0>4d},{local_time.tm_mon:0>2d},{local_time.tm_mday:0>2d},{local_time.tm_hour:0>2d}.{local_time.tm_min:0>2d}'{local_time.tm_sec:0>2d}''"
    exp_name = f"{args.model}_{args.dataset}_{args.experiment_name}_{timestr}"
    exp_path = os.path.join('./', exp_name)
    os.makedirs(exp_path)
    log_path = os.path.join(exp_path, ".log")

    # datset domains & directory
    if args.dataset == 0:
        domains = ['CDs_and_Vinyl',
                   'Digital_Music',
                   'Musical_Instruments']
    elif args.dataset == 1:
        domains = ['book', 'movie', 'music']
#        assert args.data_dir == './data/douban'
    else:
        raise NotImplementedError
    abbr_domains = [domain.split('_')[0] for domain in domains]
    data_dir = os.path.join(args.data_dir, '+'.join(abbr_domains))

    # load global and domain-specific training data
    train_mat = sp.load_npz(os.path.join(data_dir, '+'.join(abbr_domains) + '_train.npz'))
    
    train_mat_local, item_list = {}, {}
    for dom_id, domain in enumerate(domains):
        mat = sp.load_npz(os.path.join(data_dir, abbr_domains[dom_id] + '_train.npz'))
        train_mat_local[domain] = mat
        
        num_interaction_dom = len(mat.nonzero()[0])
        num_user_dom = (mat.sum(axis=1)!=0).sum()

        item_list[domain] = np.loadtxt(os.path.join(data_dir, abbr_domains[dom_id] + '_itemlist.txt'), dtype=int)
        num_item_dom = len(item_list[domain])
        
        density = 1. * num_interaction_dom / num_user_dom / num_item_dom
        log(f'[{domain}]: user {num_user_dom}, item {num_item_dom}, density {density * 100}%', log_path)

    # load evaluation data
    eval_data = {}
    for dom_id, domain in enumerate(domains):
        eval_data[domain] = np.loadtxt(os.path.join(data_dir, abbr_domains[dom_id] + '_eval.txt'), dtype=int)

    # build training datasets
    train_loader = {}
    for domain in domains:
        user_dom, item_dom = train_mat_local[domain].nonzero()
        train_data = np.hstack((user_dom.reshape(-1, 1), item_dom.reshape(-1, 1))).tolist()
        train_dataset = BPRDataset(train_data, train_mat_local[domain], item_list[domain], args.num_ng, True)
        train_loader[domain] = dataloader.DataLoader(train_dataset,
                                                     batch_size = ceil(len(user_dom)*1./args.batch_num),
                                                     shuffle=True,
                                                     num_workers=0,
                                                     drop_last=False)
    assert all(len(train_loader[domain])==args.batch_num for domain in domains), "Batch amount alignment error"
    # validate the positive and negative training samples
#    for domain in domains:
#        train_loader[domain].dataset.ng_sample()
#        mat = train_mat_local[domain].todok()
#        for uids, iids, jids in train_loader[domain]:
#            for j in range(len(uids)):
#                assert ((uids[j].item(), iids[j].item()) in mat) and ((uids[j].item(), jids[j].item()) not in mat)

    # build graph & model
    model = get_model(train_mat, train_mat_local, domains, args, device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    HR, NDCG, train_loss = {}, {}, {}
    for domain in domains:
        HR[domain], NDCG[domain], train_loss[domain] = [], [], []
    curEpoch = 1
    HR_ave_best = 0
    epoch_best = 0

    if args.load_model: # load model ckpt from disk
        ckpt_path = os.path.join('./', args.ckpt_name, args.ckpt_name + f'_{args.ckpt_epoch}.ckpt')
        ckpt = torch.load(ckpt_path)
        model = ckpt['model']
        curEpoch = ckpt['epoch'] + 1
        opt = ckpt['opt']
        opt._warned_capturable_if_run_uncaptured = True # pytorch 1.12.0 bug fix
        train_loss = ckpt['his']['loss']
        HR = ckpt['his']['HR']
        NDCG = ckpt['his']['NDCG']
        HR_ave_best = ckpt['his']['HR_ave_best']
        epoch_best = ckpt['his']['epoch_best']
        log('Load successfully.', log_path)
    model.to(device)

    # train & eval
    log(f"model training on device {next(model.parameters()).device}", log_path)
    log(args, log_path)
    for epoch in range(curEpoch, args.epoch + 1):
        curEpoch = epoch
        model.train()
        # warmup learning rate
#       if epoch == 5:
#           adjust_learning_rate(opt, warmup_lr)
#       if epoch == 20:
#           adjust_learning_rate(opt, lr)

        # dataset iterator for each domain
        train_loader_iter = {}
        for domain in domains:
            train_loader[domain].dataset.ng_sample()
            train_loader_iter[domain] = iter(train_loader[domain])
        # record loss
        for domain in domains:
            train_loss[domain].append(0)
        
        rg = tqdm(range(args.batch_num))
        for _ in rg:
            for dom_id, domain in enumerate(domains):
                user, item_pos, item_negs = next(train_loader_iter[domain]) # <u,i,j> 用户 正样本 负样本
                user = user.long().to(device)
                item_pos = item_pos.long().to(device)
                item_negs = item_negs.long().to(device)

                embedding_user, embedding_item = model(domain)
                embed_u = embedding_user[user]
                embed_pos = embedding_item[item_pos]
                embed_neg = embedding_item[item_negs]
                pred_pos = (embed_u * embed_pos).sum(dim=1).reshape(-1, 1)
                pred_negs = (embed_u.reshape(-1, 1, embedding_user.shape[-1])
                           * embed_neg.reshape(-1, args.num_ng, embedding_item.shape[-1])).sum(dim=2)

                bpr_loss = - (pred_pos - pred_negs).sigmoid().log().sum() / args.num_ng
                train_loss[domain][-1] += bpr_loss.item() / len(user)

                reg_loss = ((embed_u**2 + embed_pos**2).sum() + (embed_neg**2).sum()) / (args.num_ng + 2)
        
                if len(args.reg_domain) == len(domains):
                    loss = (bpr_loss + args.reg_domain[dom_id] * reg_loss) / len(user) # domain-specific regularization
                else:
                    loss = (bpr_loss + args.reg * reg_loss) / len(user) # unique regularization

                #adjust_learning_rate(opt, lr_domain[dom_id]) # domain-specific learning rate
                opt.zero_grad()
                loss.backward()
                opt.step()

        for domain in domains:
            train_loss[domain][-1] /= args.batch_num
        
        model.eval()
#        print(f"[Epoch {epoch}]:\n")
        log(f"[Epoch {epoch}]: average loss {np.array([loss[-1] for loss in train_loss.values()]).mean()}", log_path, timeStamp=True)
        with torch.no_grad():
            epoch_HR, epoch_NDCG = eval(model, device, args.num_ng_eval, args.topk, train_loss, domains, eval_data, log_path)
            for domain in domains:
                HR[domain].append(epoch_HR[domain])
                NDCG[domain].append(epoch_NDCG[domain])
            HR_ave = np.array([hr for hr in epoch_HR.values()]).mean()

        # save experiment results
        his = save_history(train_loss, HR, NDCG, HR_ave_best, epoch_best, args, exp_name, exp_path)
        if epoch % args.save_freq == 0:
            save_model(model, opt, epoch, his, exp_name + f'_{epoch}', exp_path)
        if HR_ave > HR_ave_best:
            epoch_best = epoch
            HR_ave_best = HR_ave
            save_model(model, opt, epoch, his, exp_name + f'_average_best', exp_path)
#            print(f'Average best model saved. {epoch}, {epoch_HR}')
            log(f'Average best model saved. {epoch}, {epoch_HR}', log_path)
        if epoch - epoch_best >= args.patience:
#            print(f'Early stopping in epoch {epoch}.')
            log(f'Early stopping in epoch {epoch}.', log_path)
            break

    for domain in domains:
        plt.plot(train_loss[domain], label=f'[{domain}] loss')
    plt.legend()
    plt.savefig(os.path.join(exp_path, f"loss_{args.epoch}.jpg"))
    plt.close()

    for domain in domains:
        plt.plot(HR[domain], label=f"[{domain}] HR@{args.topk}")
        plt.plot(NDCG[domain], label=f"[{domain}] NDCG@{args.topk}")
        plt.legend()
        plt.savefig(os.path.join(exp_path, f"evaluation_{domain}_{args.epoch}.jpg"))
        plt.close()

def run(args):
    if not args.cpu:
        assert torch.cuda.is_available(), "cuda not available"
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        device = 'cpu'
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.mode=='train':
        train(args, device)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train.py')
    # model parameters
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model', type=str, default='BPRMF', help='BPRMF, HeroGRAPH, ReCDR, ours')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--ckpt_name', type=str, help='Directory of the model to load.')
    parser.add_argument('--ckpt_epoch', type=int, help='Epoch of the ckpt to load.')
    parser.add_argument('--experiment_name', type=str, default='noname')
    parser.add_argument('--patience', type=int, default=30, help='How many epoches without improvement is tolerant before early stopping.')
    # dataset parameters
    parser.add_argument('--dataset', type=int, default=0, help="0 for ['CDs_and_Vinyl', 'Digital_Music', 'Musical_Instruments'], 1 for douban")
    parser.add_argument('--data_dir', type=str, default='./data/Amazon_2018')
    parser.add_argument('--batch_num', type=int, default=20)
    parser.add_argument('--num_ng', type=int, default=1)
    # training hyper parameters
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--node_embed', type=int, default=64)
    parser.add_argument('--edge_embed', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=1.0, help='Learning rate decay speed.')
    parser.add_argument('--min_lr', type=float, default=0.0001)
    parser.add_argument('--reg', type=float, default=0.01)
    parser.add_argument('--reg_domain', nargs = '*', type=float, default=[])
    # evaluation parameters
    parser.add_argument('--num_ng_eval', type=int, default=99)
    parser.add_argument('--topk', type=int, default=5)
    # others
    parser.add_argument('--cpu', action='store_true', default=False, help='To use cpu for training & evaluating.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n2v_epoch', type=int, default=30)
    parser.add_argument('--n2v_embed', type=int, default=64)
    parser.add_argument('--n2v_alpha', type=float, default=0.9, help='Threshold for relation expansion.')
 
    args = parser.parse_args()
    print(args)

    run(args)
