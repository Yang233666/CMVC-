#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import os
import numpy as np
from tqdm import tqdm
import math


def hinge_loss(positive_score, negative_score, gamma, show=False):
    err = positive_score - negative_score + gamma
    if show:
        print(err)
    max_err = err.clamp(0)
    return max_err

def contrastive_loss(y_pred: 'tensor', tuple_num=3, tau=0.05, device=0) -> 'tensor':
    y_true = torch.arange(y_pred.shape[0]).cuda(device)
    use_row = torch.where(y_true % tuple_num == 0)[0].unsqueeze(1)
    use_row = torch.cat([use_row, use_row + 1], dim=1).reshape(-1)
    y_true = (use_row - use_row % tuple_num * 2) + 1

    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    sim = sim - torch.eye(y_pred.shape[0]).cuda(device) * 1e12
    sim = torch.index_select(sim, 0, use_row)
    sim = sim / tau
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


class KGEModel(nn.Module):
    def __init__(self, model_name, dict_local, init, E_init, R_init, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False, mlp_dim=300):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.embed_loc = dict_local
        self.E_init = E_init
        self.R_init = R_init
        self.init = init
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        modulus_weight = 1.0
        phase_weight = 0.5
        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))

        ''' Intialize embeddings '''
        if self.init == 'crawl':
            self.entity_embedding = nn.Parameter(torch.from_numpy(self.E_init))
            self.relation_embedding = nn.Parameter(torch.from_numpy(self.R_init))
        else:
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(tensor=self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(tensor=self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        if model_name == 'pRotatE' or model_name == 'HAKE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'HAKE']:
            raise ValueError('model %s not supported' % model_name)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, mlp_dim, bias=True),
        )
        self.head[0].weight = nn.Parameter(torch.eye(self.head[0].weight.shape[0], self.head[0].weight.shape[1]))
        self.head[0].bimas = nn.Parameter(torch.zeros(self.head[0].bias.shape))
        self.head[3].weight = nn.Parameter(torch.eye(self.head[3].weight.shape[0], self.head[3].weight.shape[1]))
        self.head[3].bias = nn.Parameter(torch.zeros(self.head[3].bias.shape))

    def __del__(self):
        torch.cuda.empty_cache()

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == 'cl_np':
            return self.head(torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample.view(-1)
            ))
        elif mode == 'cl_rp':
            return self.head(torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample.view(-1)
            ))
        else:
            raise ValueError('mode %s not supported' % mode)
            
        # model_func = {
        #     'TransE': self.TransE
        # }

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'HAKE': self.HAKE
        }
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    def HAKE(self, head, relation, tail, mode):
        # print(head.shape, relation.shape, tail.shape)
        pi = 3.14159262358979323846
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(relation, 3, dim=2)
        # phase_relation, mod_relation, bias_relation = torch.chunk(relation, 2, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / pi)
        phase_relation = phase_relation / (self.embedding_range.item() / pi)
        phase_tail = phase_tail / (self.embedding_range.item() / pi)
        # print(phase_head.shape, phase_relation.shape, phase_tail.shape)
        if mode == 'head-batch':
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)
        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight
        score = self.gamma.item() - (phase_score + r_score)
        return score
    
    @staticmethod
    def train_step(args, model, optimizer, train_iterator, np_seed_neg_dict=None, np_seed_pair=None,
                   rp_seed_neg_dict=None, rp_seed_pair=None):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        negative_sample_size = int(args.single_negative_sample_size)
        gamma = torch.full((1, negative_sample_size), float(args.single_gamma))

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        # (h, r, t) -> (h1,h2,h3)
        # seed pair: (h1,h2)
        # pair_+(h1, h2), pair_-(h1, h3) -> loss
        # negative_sample_transe, negative_sample_c = negative_sample[0:10], negative_sample[11:]
        # mlp

        if args.cuda:
            positive_sample = positive_sample.cuda(args.fact_cuda)
            negative_sample = negative_sample.cuda(args.fact_cuda)
            subsampling_weight = subsampling_weight.cuda(args.fact_cuda)
            gamma = gamma.cuda(args.fact_cuda)

        negative_score = model((positive_sample, negative_sample), mode=mode)
        positive_score = model(positive_sample)
        positive_score = positive_score.repeat(1, negative_sample_size)

        loss = hinge_loss(positive_score, negative_score, gamma)
        if args.uni_weight:
            loss = loss.sum()
        else:
            loss = (subsampling_weight * loss).sum()/subsampling_weight.sum()

        if args.regularization != 0.0:
            regularization = args.regularization * (
                model.entity_embedding.norm(p=3)**3 +
                model.relation_embedding.norm(p=3).norm(p=3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        if args.conbine_loss:
            loss.backward()
            optimizer.step()

        loss_cl = 0
        np_train = []
        np_sample = torch.index_select(positive_sample, dim=1, index=torch.tensor([0,2]).cuda(args.fact_cuda) if args.cuda else torch.tensor([0,2]))
        for a, b in np_seed_pair:
            if a in np_sample and b in np_sample:
                np_train.append([a, b] + np_seed_neg_dict[a])
        np_embedding = model(torch.tensor(np_train, dtype=torch.int).cuda(args.fact_cuda) if args.cuda else torch.tensor(np_train, dtype=torch.int), 'cl_np')
        step_size = args.fact_step_size*(args.fact_neg_num+2)

        for i in range(math.ceil(np_embedding.shape[0]/step_size)):
            loss_cl += contrastive_loss(np_embedding[step_size*i:step_size*(i+1), :], tuple_num=(args.fact_neg_num+2), device=args.fact_cuda)

        rp_train = []
        rp_sample = positive_sample[:, 1]
        for a, b in rp_seed_pair:
            if a in rp_sample and b in rp_sample:
                rp_train.append([a, b] + rp_seed_neg_dict[a])
        rp_embedding = model(torch.tensor(rp_train, dtype=torch.int).cuda(args.fact_cuda) if args.cuda else torch.tensor(rp_train, dtype=torch.int),  'cl_rp')
        for i in range(math.ceil(rp_embedding.shape[0]/(step_size))):
            loss_cl += contrastive_loss(rp_embedding[step_size*i:step_size*(i+1), :], tuple_num=(args.fact_neg_num+2), device=args.fact_cuda)

        if args.conbine_loss:
            loss_cl.backward()
            optimizer.step()

        beta = 1
        loss += beta*loss_cl

        if not args.conbine_loss:
            loss.backward()
            optimizer.step()

        log = {
            **regularization_log,
            'loss': loss.item()
        }
        return log

    @staticmethod
    def cross_train_step(args, model, optimizer, seed_iterator, np_seed_neg_dict=None, np_seed_pair=None,
                   rp_seed_neg_dict=None, rp_seed_pair=None):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        negative_sample_size = int(args.cross_negative_sample_size)
        gamma = torch.full((1, negative_sample_size), float(args.cross_gamma))  # 返回大小为sizes,单位值为fill_value的矩阵

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, seed_sim, mode = next(seed_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda(args.fact_cuda)
            negative_sample = negative_sample.cuda(args.fact_cuda)
            subsampling_weight = subsampling_weight.cuda(args.fact_cuda)
            gamma = gamma.cuda(args.fact_cuda)
            seed_sim = seed_sim.cuda(args.fact_cuda)

        negative_score = model((positive_sample, negative_sample), mode=mode)
        positive_score = model(positive_sample)
        seed_sim = torch.from_numpy(np.diag(seed_sim.t().cpu().numpy()[0])).cuda(args.fact_cuda)

        positive_score = positive_score.repeat(1, negative_sample_size)

        loss = hinge_loss(positive_score, negative_score, gamma)
        loss = loss.sum(dim=1) * seed_sim

        if args.uni_weight:
            loss = loss.sum()
        else:
            loss = (subsampling_weight * loss).sum() / subsampling_weight.sum()

        if args.regularization != 0.0:
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        if args.conbine_loss:
            loss.backward()
            optimizer.step()

        loss_cl = 0
        np_train = []
        np_sample = torch.index_select(positive_sample, dim=1, index=torch.tensor([0,2]).cuda(args.fact_cuda) if args.cuda else torch.tensor([0,2]) )
        for a, b in np_seed_pair:
            if a in np_sample and b in np_sample:
                np_train.append([a, b] + np_seed_neg_dict[a])
        np_embedding = model(torch.tensor(np_train, dtype=torch.int).cuda(args.fact_cuda) if args.cuda else torch.tensor(np_train, dtype=torch.int), 'cl_np')
        step_size = args.fact_step_size*(args.fact_neg_num+2)

        for i in range(math.ceil(np_embedding.shape[0]/step_size)):
            loss_cl += contrastive_loss(np_embedding[step_size*i:step_size*(i+1), :], tuple_num=(args.fact_neg_num+2), device=args.fact_cuda)

        rp_train = []
        rp_sample = positive_sample[:, 1]
        for a, b in rp_seed_pair:
            if a in rp_sample and b in rp_sample:
                rp_train.append([a, b] + rp_seed_neg_dict[a])
        rp_embedding = model(torch.tensor(rp_train, dtype=torch.int).cuda(args.fact_cuda) if args.cuda else torch.tensor(rp_train, dtype=torch.int), 'cl_rp')
        for i in range(math.ceil(rp_embedding.shape[0]/(step_size))):
            loss_cl += contrastive_loss(rp_embedding[step_size*i:step_size*(i+1), :], tuple_num=(args.fact_neg_num+2), device=args.fact_cuda)

        if args.conbine_loss:
            loss_cl.backward()
            optimizer.step()

        beta = 1
        loss = loss + loss_cl * beta
        if not args.conbine_loss:
            loss.backward()
            optimizer.step()

        log = {
            **regularization_log,
            'loss': loss.item()
        }

        return log

    def get_seeds(self, p, side_info, logging):
        self.p = p
        self.side_info = side_info
        self.logging = logging
        self.id2ent, self.id2rel = self.side_info.id2ent, self.side_info.id2rel
        self.ent2id, self.rel2id = self.side_info.ent2id, self.side_info.rel2id
        self.ent2triple_id_list, self.rel2triple_id_list = self.side_info.ent2triple_id_list, self.side_info.rel2triple_id_list
        self.trpIds = self.side_info.trpIds
        entity_embedding, relation_embedding = self.entity_embedding.data, self.relation_embedding.data
        self.seed_trpIds, self.seed_sim = [], []
        for i in tqdm(range(len(entity_embedding))):
            for j in range(i + 1, len(entity_embedding)):
                e1_embed, e2_embed = entity_embedding[i], entity_embedding[j]
                sim = torch.cosine_similarity(e1_embed, e2_embed, dim=0)
                if sim > self.p.entity_threshold:
                    ent1, ent2 = self.id2ent[i], self.id2ent[j]
                    for ent in [ent1, ent2]:
                        triple_list = self.ent2triple_id_list[ent]
                        for triple_id in triple_list:
                            triple = self.trpIds[triple_id]
                            if str(self.id2ent[triple[0]]) == str(ent1):
                                trp = (self.ent2id[str(ent2)], triple[1], triple[2])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
                            if str(self.id2ent[triple[0]]) == str(ent2):
                                trp = (self.ent2id[str(ent1)], triple[1], triple[2])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
                            if str(self.id2ent[triple[2]]) == str(ent1):
                                trp = (triple[0], triple[1], self.ent2id[str(ent2)])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
                            if str(self.id2ent[triple[2]]) == str(ent2):
                                trp = (triple[0], triple[1], self.ent2id[str(ent1)])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)

        for i in tqdm(range(len(relation_embedding))):
            for j in range(i + 1, len(relation_embedding)):
                r1_embed, r2_embed = relation_embedding[i], relation_embedding[j]
                sim = torch.cosine_similarity(r1_embed, r2_embed, dim=0)
                if sim > self.p.relation_threshold:
                    rel1, rel2 = self.id2rel[i], self.id2rel[j]
                    for rel in [rel1, rel2]:
                        triple_list = self.rel2triple_id_list[rel]
                        for triple_id in triple_list:
                            triple = self.trpIds[triple_id]
                            if str(self.id2rel[triple[1]]) == str(rel1):
                                trp = (triple[0], self.rel2id[str(rel2)], triple[2])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
                            if str(self.id2rel[triple[1]]) == str(rel2):
                                trp = (triple[0], self.rel2id[str(rel1)], triple[2])
                                self.seed_trpIds.append(trp)
                                self.seed_sim.append(sim)
        return self.seed_trpIds, self.seed_sim

    def set_logger(self, p):
        '''
        Write logs to checkpoint and console
        '''

        if p.do_train:
            log_file = os.path.join(p.out_path or p.init_checkpoint, 'train.log')
        else:
            log_file = os.path.join(p.out_path or p.init_checkpoint, 'test.log')

        logging.getLogger().handlers = []
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def log_metrics(self, mode, step, metrics):
        '''
        Print the evaluation logs
        '''
        for metric in metrics:
            logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))