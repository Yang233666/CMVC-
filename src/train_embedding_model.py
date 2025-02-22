import pickle
import copy
import faiss
import logging
import numpy as np
import os
import torch
import tqdm
from torch.utils.data import DataLoader
from multiprocessing import Process

from src.dataloader_max_margin import TrainDataset, SeedDataset, BidirectionalOneShotIterator, seed_pair2cluster
from src.model_max_margin import KGEModel


def pair2triples(seed_pair_list, ent_list, ent2id, id2ent, ent2triple_id_list, trpIds, entity_embedding, cos_sim,
                 is_cuda=False, high_confidence=False):
    seed_trpIds, seed_sim = [], []
    if is_cuda:
        entity_embed = entity_embedding.data
    else:
        entity_embed = entity_embedding

    for seed_pair in seed_pair_list:
        i, j = seed_pair[0], seed_pair[1]
        if i < len(ent_list) and j < len(ent_list):
            ent1, ent2 = ent_list[i], ent_list[j]
            e1_embed, e2_embed = entity_embed[i], entity_embed[j]
            if is_cuda:
                sim = torch.cosine_similarity(e1_embed, e2_embed, dim=0)
            else:
                if not np.dot(e1_embed, e2_embed) == 0:
                    sim = cos_sim(e1_embed, e2_embed)
                else:
                    sim = 0
            if high_confidence:
                if sim > 0.9:
                    Append = True
                else:
                    Append = False
            else:
                Append = True
            if Append:
                for ent in [ent1, ent2]:
                    triple_list = ent2triple_id_list[ent]
                    for triple_id in triple_list:
                        triple = trpIds[triple_id]
                        if str(id2ent[triple[0]]) == str(ent1):
                            trp = (ent2id[str(ent2)], triple[1], triple[2])
                            seed_trpIds.append(trp)
                            seed_sim.append(sim)
                        if str(id2ent[triple[0]]) == str(ent2):
                            trp = (ent2id[str(ent1)], triple[1], triple[2])
                            seed_trpIds.append(trp)
                            seed_sim.append(sim)
                        if str(id2ent[triple[2]]) == str(ent1):
                            trp = (triple[0], triple[1], ent2id[str(ent2)])
                            seed_trpIds.append(trp)
                            seed_sim.append(sim)
                        if str(id2ent[triple[2]]) == str(ent2):
                            trp = (triple[0], triple[1], ent2id[str(ent1)])
                            seed_trpIds.append(trp)
                            seed_sim.append(sim)
    return seed_trpIds, seed_sim


def seed_process(seed_pair_list, entity_embedding, neg_num=5):
    cluster_list = seed_pair2cluster(seed_pair_list)
    entity_num = entity_embedding.shape[0]
    for key in cluster_list.keys():
        cluster_list[key] = list(filter(lambda x: x < entity_num, cluster_list[key]))

    seed_list = []
    for a, b in seed_pair_list:
        if a not in seed_list:
            seed_list.append(a)
        if b not in seed_list:
            seed_list.append(b)


    seed_neg_list = dict()
    for k, np_list in cluster_list.items():
        cluster_tmp = copy.copy(cluster_list)
        cluster_tmp.pop(k)
        train_inputs_tmp = list(cluster_tmp.values())
        train_inputs_tmp = [i for l in train_inputs_tmp for i in l]
        embedding_tmp = entity_embedding[train_inputs_tmp, :]

        index = faiss.IndexFlatL2(embedding_tmp.shape[1])
        index.add(embedding_tmp)
        for np in np_list:
            if neg_num > 0:
                D, I = index.search(entity_embedding[np].reshape(1, -1), neg_num)
                if neg_num > 1:
                    neg_list = []
                    for i in I.squeeze().tolist():
                        neg_list.append(train_inputs_tmp[i])
                else:
                    neg_list = [I.item()]
                seed_neg_list[np] = neg_list
            else:
                seed_neg_list[np] = []

    return seed_neg_list


class Train_Embedding_Model(Process):
    """
    Learns embeddings for NPs and relation phrases
    """

    def __init__(self, params, side_info, E_init, R_init, np_seed_pair, rp_seed_pair, new_seed_triples, new_seed_sim,
                 model_training_time):
        Process.__init__(self)
        self.p = params
        self.side_info = side_info
        self.E_init = E_init
        self.R_init = R_init
        self.web_seed_pair_list = np_seed_pair
        self.rp_seed_pair = rp_seed_pair
        self.new_seed_trpIds = new_seed_triples
        self.new_seed_sim = new_seed_sim
        self.model_training_time = model_training_time

    def __del__(self):
        torch.cuda.empty_cache()
        print("Train_Embedding_Model del ... ")

    def run(self):
        self.train()

    def train(self):
        np.random.seed(0)
        nentity, nrelation = len(self.side_info.ent_list), len(self.side_info.rel_list)
        train_triples = self.side_info.trpIds
        logging.info('#train: %d' % len(train_triples))

        self.nentity = nentity
        self.nrelation = nrelation

        logging.info('Model: %s' % self.p.model)
        logging.info('#entity: %d' % nentity)
        logging.info('#relation: %d' % nrelation)
        logging.info('#train: %d' % len(train_triples))

        # combine the train triples and seed triples
        use_soft_learning = self.p.use_soft_learning
        # --------------------------------------------------

        kge_model = KGEModel(
            model_name=self.p.model,
            dict_local=self.p.embed_loc,
            init=self.p.embed_init,
            E_init=self.E_init,
            R_init=self.R_init,
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=self.p.hidden_dim,
            gamma=self.p.single_gamma,
            double_entity_embedding=self.p.double_entity_embedding,
            double_relation_embedding=self.p.double_relation_embedding
        )
        kge_model.set_logger(self.p)

        logging.info('Model Parameter Configuration:')
        for name, param in kge_model.named_parameters():
            logging.info(
                'Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

        if self.p.cuda:
            kge_model = kge_model.cuda(self.p.fact_cuda)

        if self.p.do_train:
            np_seed_neg_dict = seed_process(self.web_seed_pair_list, self.E_init, neg_num=self.p.fact_neg_num)
            rp_seed_neg_dict = seed_process(self.rp_seed_pair, self.R_init, neg_num=self.p.fact_neg_num)
            train_dataloader_head = DataLoader(
                TrainDataset(train_triples, self.web_seed_pair_list, nentity, nrelation,
                             self.p.single_negative_sample_size, 'head-batch'),
                batch_size=self.p.single_batch_size,
                shuffle=True,
                num_workers=max(1, self.p.cpu_num // 6),
                collate_fn=TrainDataset.collate_fn
            )

            train_dataloader_tail = DataLoader(
                TrainDataset(train_triples, self.web_seed_pair_list, nentity, nrelation,
                             self.p.single_negative_sample_size, 'tail-batch'),
                batch_size=self.p.single_batch_size,
                shuffle=True,
                num_workers=max(1, self.p.cpu_num // 6),
                collate_fn=TrainDataset.collate_fn
            )
            self.train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail, False)
            # Set training configuration
            current_learning_rate = self.p.learning_rate
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
            if self.p.warm_up_steps:
                warm_up_steps = self.p.warm_up_steps
            else:
                warm_up_steps = self.p.max_steps // 2

        if self.p.init_checkpoint:
            # Restore model from checkpoint directory
            logging.info('Loading checkpoint %s...' % self.p.init_checkpoint)
            checkpoint = torch.load(os.path.join(self.p.init_checkpoint, 'checkpoint'))
            init_step = checkpoint['step']
            kge_model.load_state_dict(checkpoint['model_state_dict'])
            if self.p.do_train:
                current_learning_rate = checkpoint['current_learning_rate']
                warm_up_steps = checkpoint['warm_up_steps']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # logging.info('Ramdomly Initializing %s Model...' % self.p.model)
            init_step = 0

        step = init_step

        logging.info('Start Training...')
        logging.info('init_step = %d' % init_step)
        logging.info('single_batch_size = %d' % self.p.single_batch_size)
        logging.info('single_negative_adversarial_sampling = %d' % self.p.single_negative_sample_size)
        logging.info('hidden_dim = %d' % self.p.hidden_dim)
        logging.info('single_gamma = %f' % self.p.single_gamma)
        
        if self.p.use_cross_seed:
            logging.info('self.p.use_cross_seed = %f' % self.p.use_cross_seed)
            logging.info('self.p.max_steps = %f' % self.p.max_steps)
            logging.info('self.p.turn_to_seed = %f' % self.p.turn_to_seed)
            logging.info('self.p.seed_max_steps = %f' % self.p.seed_max_steps)
        else:
            logging.info('Do not use seeds ...')

        # Set valid dataloader as it would be evaluated during training

        if self.p.do_train:
            logging.info('learning_rate = %f' % current_learning_rate)

            training_logs = []
            if self.p.use_cross_seed:
                if len(self.new_seed_trpIds) > 0:
                    logging.info('#ALL seed: %d' % len(self.new_seed_trpIds))
                    seed_triples = self.new_seed_trpIds
                    seed_sim = self.new_seed_sim

                else:
                    seed_triples = self.side_info.seed_trpIds
                    seed_sim = self.side_info.seed_sim
                    logging.info('#EL seed: %d' % len(seed_triples))

                if use_soft_learning:
                    print('use soft seed loss !')
                else:
                    for i in range(len(seed_sim)):
                        seed_sim[i] = 1
                    print('seed_sim:', type(seed_sim), len(seed_sim), seed_sim[0:10])
                    print('do not use soft seed loss !')
                seed_dataloader_head = DataLoader(
                    SeedDataset(seed_triples, self.web_seed_pair_list, nentity, nrelation,
                                self.p.cross_negative_sample_size, 'head-batch',
                                seed_sim),
                    batch_size=self.p.cross_batch_size,
                    shuffle=True,
                    num_workers=max(1, self.p.cpu_num // 6),
                    collate_fn=SeedDataset.collate_fn
                )

                seed_dataloader_tail = DataLoader(
                    SeedDataset(seed_triples, self.web_seed_pair_list, nentity, nrelation,
                                self.p.cross_negative_sample_size, 'tail-batch',
                                seed_sim),
                    batch_size=self.p.cross_batch_size,
                    shuffle=True,
                    num_workers=max(1, self.p.cpu_num // 6),
                    collate_fn=SeedDataset.collate_fn
                )
                self.seed_iterator = BidirectionalOneShotIterator(seed_dataloader_head, seed_dataloader_tail)

            # Training Loop
            loss_list = []
            step_list = []
            f1_list = []
            l_list = []

            folder = '../file/' + self.p.dataset + '_' + self.p.split + '/' + 'multi_view/relation_view/'
            for step in range(init_step, self.p.max_steps):
                log = kge_model.train_step(self.p, kge_model, optimizer, self.train_iterator, np_seed_neg_dict,
                                           self.web_seed_pair_list,
                                           rp_seed_neg_dict, self.rp_seed_pair)
                loss = log['loss']
                loss_list.append(loss)
                step_list.append(step)
                training_logs.append(log)
                if self.p.use_cross_seed:
                    if step > 0 and step % self.p.turn_to_seed == 0:
                        for i in tqdm.tqdm(range(0, self.p.seed_max_steps)):
                            log = kge_model.cross_train_step(self.p, kge_model, optimizer, self.seed_iterator,
                                                             np_seed_neg_dict, self.web_seed_pair_list,
                                                             rp_seed_neg_dict, self.rp_seed_pair)
                            training_logs.append(log)
                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, kge_model.parameters()),
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * 3

                if step % self.p.log_steps == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                    KGEModel.log_metrics(self.p, 'Training average', step, metrics)
                    l_list.append(metrics['loss'])
                    training_logs = []

        self.entity_embedding = kge_model.entity_embedding.detach().cpu().numpy()
        self.relation_embedding = kge_model.relation_embedding.detach().cpu().numpy()
        with torch.no_grad():
            entity_mlp = kge_model(torch.arange(0, kge_model.nentity, dtype=torch.int).cuda(self.p.fact_cuda),
                                   'cl_np').detach().cpu().numpy()
            relation_mlp = kge_model(torch.arange(0, kge_model.nrelation, dtype=torch.int).cuda(self.p.fact_cuda),
                                     'cl_rp').detach().cpu().numpy()
        del kge_model
        pickle.dump(self.entity_embedding, open(folder + 'entity_embedding_%d' % self.model_training_time, 'wb'))
        pickle.dump(self.relation_embedding, open(folder + 'relation_embedding_%d' % self.model_training_time, 'wb'))
        pickle.dump(entity_mlp, open(folder + 'entity_mlp_%d' % self.model_training_time, 'wb'))
        pickle.dump(relation_mlp, open(folder + 'relation_mlp_%d' % self.model_training_time, 'wb'))
        # pickle.dump(l_list, open(fname + 'loss_list.pkl', 'wb'))
        return entity_mlp, relation_mlp



