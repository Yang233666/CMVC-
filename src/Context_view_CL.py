import pandas as pd
import math, time, pickle
from src.helper import checkFile
import torch
from torch import nn
from torch import optim
from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn.functional as F
import random
import itertools
import numpy as np
import faiss
from multiprocessing import Process


def simcse_sup_loss(y_pred, tuple_num=3, tau=0.05, device=1):
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


class SimCSE(nn.Module):
    def __init__(self, max_length, bert_folder='../data/bert-base-uncased', DROPOUT=0.1):
        super(SimCSE, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('../data/bert-base-uncased')
        config = BertConfig.from_pretrained(bert_folder)
        config.attention_probs_dropout_prob = DROPOUT  # 修改config的dropout系数
        config.hidden_dropout_prob = DROPOUT
        self.bert = BertModel.from_pretrained(bert_folder, config=config)

        self.tokenizer = BertTokenizer.from_pretrained('../data/bert-base-uncased')
        self.bert = BertModel.from_pretrained('../data/bert-base-uncased')

        bert_dim = config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(bert_dim, bert_dim, bias=True),
            nn.BatchNorm1d(bert_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bert_dim, 256, bias=True),
        )
        self.head[0].weight = nn.Parameter(torch.eye(self.head[0].weight.shape[0], self.head[0].weight.shape[1]))
        self.head[0].bias = nn.Parameter(torch.zeros(self.head[0].bias.shape))
        self.head[3].weight = nn.Parameter(torch.eye(self.head[3].weight.shape[0], self.head[3].weight.shape[1]))
        self.head[3].bias = nn.Parameter(torch.zeros(self.head[3].bias.shape))

        self.max_length = max_length
        print('self.max_length:', self.max_length)

    def __del__(self):
        torch.cuda.empty_cache()
        print("SimCSE del ... ")

    def forward(self, batch_sentences, device=1):
        input_ids, attention_mask = self.to_tokenizer(batch_sentences, device)
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        feature = self.head(bert_cls_hidden_state)
        return bert_cls_hidden_state, feature

    def to_tokenizer(self, batch_sentences, device=1):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           truncation=True,
                                                           max_length=self.max_length,
                                                           pad_to_max_length=True)
        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda(device)
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda(device)
        return input_ids, attention_mask


class BERT_Model(Process):

    def __init__(self, params, side_info, input_list, cluster_predict_list, true_ent2clust, true_clust2ent,
                 model_training_time, sub_uni2triple_dict=None, rel_id2sentence_list=None,
                 K=0):
        Process.__init__(self)
        self.p = params
        self.side_info = side_info
        self.input_list = input_list
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent
        self.model_training_time = model_training_time
        self.sub_uni2triple_dict = sub_uni2triple_dict
        self.rel_id2sentence_list = rel_id2sentence_list
        if 'reverb45k' in self.p.dataset:
            self.epochs = 81
            self.lr = 0.0005
        else:
            self.epochs = 16
            self.lr = 0.001

        self.K = K
        self.cluster_predict_list = cluster_predict_list
        print('self.epochs:', self.epochs)
        self.coefficient_1, self.coefficient_2 = 0.95, 0.99
        self.max_length = 256

        self.use_subset = False
        self.tuple_num = 22
        self.batch_size = 88

    def run(self):
        self.fine_tune()

    def fine_tune(self):
        print('Fine-tune BERT ', 'self.model_training_time:', self.model_training_time)
        target_list = []
        cluster2target_dict = dict()
        num = 0
        for i in range(len(self.cluster_predict_list)):
            label = self.cluster_predict_list[i]
            if label not in cluster2target_dict:
                cluster2target_dict.update({label: num})
                num += 1
            target_list.append(cluster2target_dict[label])
        self.target_num = max(target_list) + 1
        self.sentences_list, self.targets_list = [], []
        self.sub2sentence_id_dict = dict()

        print('self.p.input:', self.p.input)
        print('self.max_length:', self.max_length)
        all_length = 0
        num = 0

        for i in range(len(self.input_list)):
            if self.p.input == 'entity':
                ent_id = self.side_info.ent2id[self.input_list[i]]
            else:
                ent_id = self.side_info.rel2id[self.input_list[i]]
            if not (self.p.input == 'entity' and ent_id not in self.side_info.isSub):
                if self.p.input == 'entity':
                    sentence_id_list = self.side_info.ent_id2sentence_list[ent_id]
                else:
                    sentence_id_list = self.rel_id2sentence_list[ent_id]
                longest_index, longest_length = 0, 0
                for j in range(len(sentence_id_list)):
                    id = sentence_id_list[j]
                    sentence = self.side_info.sentence_List[id]
                    if len(sentence) > longest_length and len(sentence) < self.max_length + 50:
                        longest_index, longest_length = j, len(sentence)
                sentence_id_list = [sentence_id_list[longest_index]]
                all_length += longest_length
                sentences_num_list = []
                for sentence_id in sentence_id_list:
                    sentence = self.side_info.sentence_List[sentence_id]
                    self.sentences_list.append(sentence)
                    target = target_list[i]
                    self.targets_list.append(target)
                    sentences_num_list.append(num)
                    num += 1
                self.sub2sentence_id_dict.update({i: sentences_num_list})
        ave = all_length / len(self.input_list)
        print('all_length:', all_length, 'ave:', ave)
        print()
        print('self.sentences_list:', type(self.sentences_list), len(self.sentences_list))
        print('self.targets_list:', type(self.targets_list), len(self.targets_list))
        different_labels = list(set(self.targets_list))
        print('different_labels:', type(different_labels), len(different_labels), different_labels)

        sentence_data = {'sentences': self.sentences_list, 'targets': self.targets_list}
        frame = pd.DataFrame(sentence_data)
        self.sentences = frame['sentences'].values
        self.targets = frame['targets'].values
        self.train_inputs, self.train_targets = self.sentences, self.targets
        # train the model
        sim_sce_model = SimCSE(self.max_length).cuda(self.p.context_cuda)
        optimizer = optim.SGD(sim_sce_model.parameters(), lr=self.lr)

        fname = '../file/' + self.p.dataset + '_' + self.p.split + '/multi_view/train_batch_' + str(
            self.model_training_time) + '.pkl'
        if not checkFile(fname):
            train_inputs_embedding, train_mlp_embedding = get_embeddings(sim_sce_model, self.train_inputs,
                                                                         self.p.context_cuda)
            self.sorted_train_inputs, self.sorted_train_targets = conv2tuple(self.train_inputs, self.train_targets,
                                                                             train_inputs_embedding, self.tuple_num,
                                                                             shuffle=True)
            pickle.dump((self.sorted_train_inputs, self.sorted_train_targets), open(fname, 'wb'))
            print('train batch save!')
            del train_inputs_embedding, train_mlp_embedding
        else:
            self.sorted_train_inputs, self.sorted_train_targets = pickle.load(open(fname, 'rb'))
            print('train batch load!')
        del fname

        batch_train_inputs, batch_train_targets = [], []
        batch_count = math.ceil(len(self.sorted_train_inputs) / self.batch_size)
        print('batch_count:', batch_count)

        for i in range(batch_count):
            batch_train_inputs.append(
                np.array(self.sorted_train_inputs[i * self.batch_size: (i + 1) * self.batch_size]))
            batch_train_targets.append(
                np.array(self.sorted_train_targets[i * self.batch_size: (i + 1) * self.batch_size]))
        fname = '../file/' + self.p.dataset + '_' + self.p.split + '/multi_view/checkpoint_' + str(
            self.model_training_time)
        if checkFile(fname):
            checkpoint = torch.load(fname)
            start_epoc = checkpoint['epoc'] + 1
            sim_sce_model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Load checkpoint at %d" % start_epoc)
        else:
            start_epoc = 0
        del fname

        for epoch in range(start_epoc, self.epochs):
            avg_epoch_loss = 0
            for i in range(batch_count):
                optimizer.zero_grad()
                cls_output, mlp_output = sim_sce_model(batch_train_inputs[i], self.p.context_cuda)
                loss = simcse_sup_loss(mlp_output, self.tuple_num, 0.05, device=self.p.context_cuda)
                loss.backward()
                optimizer.step()
                avg_epoch_loss += loss.item()
            real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
            print(real_time,
                  "Epoch: %d, Lr:%.4f, Loss: %.4f" % (epoch, optimizer.param_groups[0]['lr'], avg_epoch_loss))
            if epoch in []:
                cls, mlp = get_embeddings(sim_sce_model, self.train_inputs, self.p.context_cuda)
                checkpoint = {
                    'epoc': epoch,
                    'net': sim_sce_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                fname = '../file/' + self.p.dataset + '_' + self.p.split + '/multi_view/'
                torch.save(checkpoint, fname + 'checkpoint_' + str(self.model_training_time))
                pickle.dump(cls,
                            open(fname + '/context_view_' + self.p.input + '/bert_cls_el_'
                                 + str(self.model_training_time) + '_' + str(epoch), 'wb'))
                pickle.dump(mlp,
                            open(fname + '/context_view_' + self.p.input + '/mlp_output_el_'
                                 + str(self.model_training_time) + '_' + str(epoch), 'wb'))

        self.BERT_CLS, self.MLP_OUTPUT = get_embeddings(sim_sce_model, self.train_inputs, self.p.context_cuda)
        del sim_sce_model

        print('self.BERT_CLS:', type(self.BERT_CLS), self.BERT_CLS.shape)
        print('self.MLP_OUTPUT:', type(self.MLP_OUTPUT), self.MLP_OUTPUT.shape)

        folder_to_make = '../file/' + self.p.dataset + '_' + self.p.split + '/multi_view/context_view_' + str(
            self.p.input) + '/'
        pickle.dump(self.BERT_CLS, open(folder_to_make + 'bert_cls_el_' + str(self.model_training_time), 'wb'))
        pickle.dump(self.MLP_OUTPUT, open(folder_to_make + 'mlp_output_el_' + str(self.model_training_time), 'wb'))
        return self.BERT_CLS, self.MLP_OUTPUT


def get_embeddings(model, inputs, device):
    with torch.no_grad():
        for i in range(math.ceil(len(inputs) / 80)):
            cls_list, output = model(np.array(inputs[i * 80: (i + 1) * 80]), device)
            if i == 0:
                bert_cls_hidden_state = cls_list
                outputs = output
            else:
                bert_cls_hidden_state = torch.cat((bert_cls_hidden_state, cls_list), 0)
                outputs = torch.cat([outputs, output], 0)
        train_input_embedding = bert_cls_hidden_state.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
    return train_input_embedding, outputs


def input_seed(train_inputs, train_targets, train_inputs_embedding):
    seed_dict = {}
    inputs_out = []
    targets_out = []
    embedding_out = []

    for idx, key in enumerate(train_targets):
        if key in seed_dict:
            seed_dict[key].append(idx)
        else:
            seed_dict[key] = [idx]
    seed_dict = list(seed_dict.items())
    i = 0
    seed_center = []
    for (key, idx_list) in seed_dict:
        if len(idx_list) > 1:
            inputs_out.extend(train_inputs[[idx_list]].squeeze().tolist())
            embedding_out.extend(train_inputs_embedding[[idx_list]].squeeze().tolist())
            targets_out.extend([i] * len(idx_list))

            seed_center.append(np.sum(train_inputs_embedding[[idx_list]], axis=0) / len(idx_list))
            i = i + 1

    seed_dict = {}
    for idx, key in enumerate(targets_out):
        if key in seed_dict:
            seed_dict[key].append(idx)
            # seed_embeddings[key] =
        else:
            seed_dict[key] = [idx]
    return np.array(inputs_out), np.array(targets_out), np.array(embedding_out).astype('float32'), seed_dict, np.array(
        seed_center).astype('float32')


def conv2tuple(train_inputs, train_targets, train_input_embedding, tuple_num, sorted=False, unsup=False, shuffle=False,
               rand=False, top=True, center=False):
    seed_dict = {}
    neg_num = tuple_num - 2
    if rand:
        for idx, key in enumerate(train_targets):
            if key in seed_dict:
                seed_dict[key].append(idx)
            else:
                seed_dict[key] = [idx]
    else:
        # train_input ---BERT--> train_input_embedding
        train_inputs, train_targets, train_input_embedding, seed_dict, seed_center = input_seed(train_inputs,
                                                                                                train_targets,
                                                                                                train_input_embedding)

    if sorted:
        seed_dict = sorted(seed_dict.items(), key=lambda e: len(e[1]), reverse=True)
    else:
        seed_dict = list(seed_dict.items())

    sorted_train_inputs, sorted_train_targets = [], []
    from src.utils import cos_sim
    seed_sim = []
    for (key, idx_list) in seed_dict:
        if len(idx_list) > 1:
            # for idx in idx_list:
            #     self.sorted_train_inputs.append(self.train_inputs[idx])
            #     self.sorted_train_targets.append(key)
            if not rand:
                if center:
                    seed_dict_tmp = seed_dict[:key] + seed_dict[key + 1:]
                    seed_center_tmp = np.delete(seed_center, key, axis=0)
                    index = faiss.IndexFlatL2(seed_center_tmp.shape[1])
                    index.add(seed_center_tmp)
                else:
                    train_inputs_tmp = np.delete(train_inputs, idx_list)
                    train_input_embedding_tmp = np.delete(train_input_embedding, idx_list, axis=0)
                    index = faiss.IndexFlatL2(train_input_embedding_tmp.shape[1])
                    index.add(train_input_embedding_tmp)

            for i, j in itertools.combinations(idx_list, 2):
                sorted_train_inputs.append(train_inputs[i])
                sorted_train_inputs.append(train_inputs[j])
                seed_sim.append(cos_sim(train_input_embedding[i], train_input_embedding[j]))
                if neg_num == 0:
                    continue
                if not rand:
                    x_q = np.expand_dims(train_input_embedding[i], 0)
                    if not top:
                        x_q = -x_q
                    D, I = index.search(x_q, neg_num)
                    if center:
                        num = 0
                        neg_list = []
                        while len(neg_list) < neg_num:
                            idx = seed_dict_tmp[I[0, num]][1]
                            if len(neg_list) + len(idx) > neg_num:
                                idx = random.sample(idx, neg_num - len(neg_list))
                            neg_list = neg_list + idx
                            num = num + 1
                        if neg_num > 1:
                            sorted_train_inputs.append(train_inputs[[neg_list]].squeeze().tolist())
                        else:
                            sorted_train_inputs.extend(train_inputs[[neg_list]].squeeze().tolist())
                    else:
                        if neg_num == 1:
                            sorted_train_inputs.append(train_inputs_tmp[[I]].squeeze().tolist())
                        else:
                            sorted_train_inputs.extend(train_inputs_tmp[[I]].squeeze().tolist())
                else:
                    for n in range(neg_num):
                        (k, l) = random.choice(seed_dict)
                        while k == key or len(l) <= 1:
                            # while k == key:
                            (k, l) = random.choice(seed_dict)
                        sorted_train_inputs.append(train_inputs[random.choice(l)])
        elif unsup:
            for n in range(neg_num):
                sorted_train_inputs.append(train_inputs[idx_list[0]])
                sorted_train_inputs.append(train_inputs[idx_list[0]])

                if neg_num == 1:
                    (k, l) = random.choice(seed_dict)
                    while k == key:
                        (k, l) = random.choice(seed_dict)
                    sorted_train_inputs.append(train_inputs[random.choice(l)])
    if shuffle:
        l = np.array(sorted_train_inputs).reshape(-1, tuple_num)
        np.random.seed(0)
        np.random.shuffle(l)
        sorted_train_inputs = list(l.reshape(-1))
    print("Training set generate successful!")
    return sorted_train_inputs, sorted_train_targets
