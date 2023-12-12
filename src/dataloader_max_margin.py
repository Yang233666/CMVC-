#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch.utils.data import Dataset
import numpy as np


def seed_pair2cluster(seed_pair_list):
    cluster_list = []

    def idx_cluster(cluster_list, x):
        for idx, l in enumerate(cluster_list):
            if x in l:
                return idx
        return -1

    for a, b in seed_pair_list:
        if a != b:
            cluster_a = idx_cluster(cluster_list, a)
            cluster_b = idx_cluster(cluster_list, b)
            if cluster_a < 0 and cluster_b < 0:
                cluster_list.append([a, b])
            elif cluster_a < 0 <= cluster_b:
                cluster_list[cluster_b].append(a)
            elif cluster_a >= 0 > cluster_b:
                cluster_list[cluster_a].append(b)
            else:
                if cluster_a != cluster_b:
                    cluster_list[min(cluster_a, cluster_b)] += cluster_list[max(cluster_a, cluster_b)]
                    cluster_list.pop(max(cluster_a, cluster_b))
                else:
                    continue
        else:
            cluster = idx_cluster(cluster_list, a)
            if cluster < 0:
                cluster_list.append([a])
    cluster = dict()
    for idx, l in enumerate(cluster_list):
        cluster[idx] = list(set(cluster_list[idx]))

    return cluster


class TrainDataset(Dataset):
    def __init__(self, triples, seed_pair, nentity, nrelation, negative_sample_size, mode, seed_only=True):
        self.cluster_list = seed_pair2cluster(seed_pair)
        self.seed_list = []
        for a, b in seed_pair:
            if a not in self.seed_list:
                self.seed_list.append(a)
            if b not in self.seed_list:
                self.seed_list.append(b)

        self.triples = triples
        self.len = len(self.triples)
        self.triple_set = set(self.triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            negative_sample = torch.from_numpy(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class SeedDataset(Dataset):
    def __init__(self, triples, seed_pair, nentity, nrelation, negative_sample_size, mode, seed_sim, seed_only=True):
        self.cluster_list = seed_pair2cluster(seed_pair)
        self.seed2cluster = dict()
        for key, l in self.cluster_list.items():
            for seed in l:
                self.seed2cluster[seed] = key
        self.seed_list = []
        for a, b in seed_pair:
            if a not in self.seed_list:
                self.seed_list.append(a)
            if b not in self.seed_list:
                self.seed_list.append(b)
        self.notinseed_list = list(set(range(nentity))-set(self.seed_list))
        self.triples = triples

        self.triple_set = set(self.triples)
        self.len = len(self.triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(triples)
        self.seed_sim = seed_sim
        self.mode = mode
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        seed_sim = self.seed_sim[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
        seed_sim = torch.Tensor([seed_sim])
        return positive_sample, negative_sample, subsampling_weight, seed_sim, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.stack([_[2] for _ in data], dim=0)
        seed_sim = torch.stack([_[3] for _ in data], dim=0)
        mode = data[0][4]
        return positive_sample, negative_sample, subsample_weight, seed_sim, mode

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail, iteralble=False):
        if iteralble:
            self.iterator_head = dataloader_head
            self.iterator_tail = dataloader_tail
        else:
            self.iterator_head = self.one_shot_iterator(dataloader_head)
            self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
