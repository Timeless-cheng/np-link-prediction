import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence

import numpy as np
import random
import os
import copy

from models import Decoder, LatentEncoder, MuSigmaEncoder, EntityEmbedding
from utils import load_data, load_processed_data, cal_mrr

class NueralProcess(nn.Module):
   
    def __init__(self, args, use_cuda):
    
        super(NueralProcess, self).__init__()

        self.args = args
        self.use_cuda = use_cuda

        self.entity2id, self.relation2id, self.train_triplets, self.valid_triplets, self.test_triplets = load_data('./Dataset/raw_data/{}'.format(args.dataset))

        # entities - 2500/1000/1500 for train/validate/test
        # associated triplets - 72065/6246/9867
        self.meta_train_task_entity_to_triplets, self.meta_valid_task_entity_to_triplets, self.meta_test_task_entity_to_triplets \
            = load_processed_data('./Dataset/processed_data/{}'.format(self.args.dataset))

        # all triplets: 310116
        # all entities: 14541
        # 模型评估阶段才会使用
        self.all_triplets = torch.LongTensor(np.concatenate((self.train_triplets, self.valid_triplets, self.test_triplets)))
        
        # entities for valid and test
        self.meta_task_entity = np.concatenate((list(self.meta_valid_task_entity_to_triplets.keys()), list(self.meta_test_task_entity_to_triplets.keys())))
        # not hold an entity during training. 
        self.entities_list = np.delete(np.arange(len(self.entity2id)), self.meta_task_entity)

        self.load_pretrain_embedding()
        self.embed = EntityEmbedding(self.args.embed_size, self.args.embed_size, len(self.entity2id), len(self.relation2id),
                            args = self.args, entity_embedding = self.pretrain_entity_embedding, relation_embedding = self.pretrain_relation_embedding)
        # MLP
        self.latent_encoder = LatentEncoder(embed_size=args.embed_size, num_hidden1=500, num_hidden2=200,
                                r_dim=args.embed_size, dropout_p=0.5)
        # MuSigma
        self.dist = MuSigmaEncoder(args.embed_size, args.embed_size)

        # Decoder
        self.decoder = Decoder(self.args, self.args.embed_size)
        
        # We set the embedding of unseen entities as the zero vector.
        meta_task_entity = torch.LongTensor(self.meta_task_entity)
        self.embed.entity_embedding.weight.data[meta_task_entity] = torch.zeros(len(meta_task_entity), self.embedding_size)
    
        if self.use_cuda:
            self.all_triplets = self.all_triplets.cuda()

    def load_pretrain_embedding(self):

        # Set the embedding dimension for both entity and relation as 100
        self.embedding_size = int(self.args.pre_train_emb_size)
        if self.args.pre_train:

            pretrain_model_path = './Pretraining/{}'.format(self.args.dataset)

            entity_file_name = os.path.join(pretrain_model_path, '{}_entity_{}.npy'.format(self.args.pre_train_model, self.embedding_size))
            relation_file_name = os.path.join(pretrain_model_path, '{}_relation_{}.npy'.format(self.args.pre_train_model, self.embedding_size))

            # pretrain_entity_embedding: [num_entities, 100]
            # pretrain_relation_embedding: [num_relations, 100]
            self.pretrain_entity_embedding = torch.Tensor(np.load(entity_file_name))
            self.pretrain_relation_embedding = torch.Tensor(np.load(relation_file_name))
        else:
            self.pretrain_entity_embedding = None
            self.pretrain_relation_embedding = None

    # # Meta-Learning for Long-Tail Tasks
    # def cal_train_few(self, epoch):

    #     if self.args.model_tail == 'log':

    #         for i in range(self.args.max_few):

    #             if epoch < (self.args.n_epochs / (2 ** i)):

    #                 continue

    #             return max(min(self.args.max_few, self.args.few + i - 1), self.args.few)

    #         return self.args.max_few

    #     else:

    #         return self.args.few

    def embed_concat(self, unseen_entities, total_unseen_entity_embedding, triplets, flag):
        
        h, r, t = torch.t(triplets)

        if self.use_cuda:
            h = h.cuda()
            t = t.cuda()
        
        h_embed = self.embed.entity_embedding(h)
        t_embed = self.embed.entity_embedding(t)
        r_embed = self.embed.relation_embedding[r]

        h = h.cpu().numpy()
        t = t.cpu().numpy()
        for idx in range(len(unseen_entities)):
            h_idx = np.where(h == unseen_entities[idx])[0]
            t_idx = np.where(t == unseen_entities[idx])[0]
            if h_idx.shape[0] != 0:
                h_embed[h_idx] = total_unseen_entity_embedding[idx]
            if t_idx.shape[0] != 0:
                t_embed[t_idx] = total_unseen_entity_embedding[idx]

        if flag == 1:       # positive triplets
            label = torch.ones(triplets.shape[0], 1).to(h_embed)
            embeddings = torch.cat([h_embed, r_embed, t_embed, label], dim=-1)
        elif flag == 0:     # negative triplets
            label = torch.zeros(triplets.shape[0], 1).to(h_embed)
            embeddings = torch.cat([h_embed, r_embed, t_embed, label], dim=-1)
        else:
            embeddings = torch.cat([h_embed, r_embed, t_embed], dim=-1)

        return embeddings
    
    def concat_one_unseen_embed(self, unseen, unseen_embedding, triplets, flag):
        
        h, r, t = torch.t(triplets)

        if self.use_cuda:
            h = h.cuda()
            t = t.cuda()
        
        h_embed = self.embed.entity_embedding(h)
        t_embed = self.embed.entity_embedding(t)
        r_embed = self.embed.relation_embedding[r]

        h = h.cpu().numpy()
        t = t.cpu().numpy()
        h_idx = np.where(h == unseen)[0]
        t_idx = np.where(t == unseen)[0]
        if h_idx.shape[0] != 0:
            h_embed[h_idx] = unseen_embedding
        if t_idx.shape[0] != 0:
            t_embed[t_idx] = unseen_embedding
        
        if flag == 1:       # positive triplets
            label = torch.ones(triplets.shape[0], 1).to(h_embed)
            embeddings = torch.cat([h_embed, r_embed, t_embed, label], dim=-1)
        elif flag == 0:     # negative triplets
            label = torch.zeros(triplets.shape[0], 1).to(h_embed)
            embeddings = torch.cat([h_embed, r_embed, t_embed, label], dim=-1)
        else:
            embeddings = torch.cat([h_embed, r_embed, t_embed], dim=-1)
        
        return embeddings


    def cal_score(self, query_embeds, z):
        h = query_embeds[:, :100]
        r = query_embeds[:, 100:200]
        t = query_embeds[:, 200:300]

        h = self.decoder(h, z)
        t = self.decoder(t, z)

        if self.args.score_function == 'DistMult':
            score = h * r * t
            score = torch.sum(score, dim = 1)
        
        return score


    def forward(self, epoch):
        # Training entities - 2500
        train_task_pool = list(self.meta_train_task_entity_to_triplets.keys())
        random.shuffle(train_task_pool)       

        total_unseen_entity = []
        total_unseen_entity_embedding = []

        support_pos_triplets = []
        support_neg_triplets = []
        query_pos_triplets = []
        query_neg_triplets = []

        context_dists = []
        target_dists = []

        # train_few = self.cal_train_few(epoch)
        train_few = 5

        # We randomly sample 500 unseen entities in the meta-training set.
        for unseen_entity in train_task_pool[:self.args.num_train_entity]:
            
            # randomly sample a unseen entities and its corresponding triplets.
            triplets = self.meta_train_task_entity_to_triplets[unseen_entity]
            random.shuffle(triplets)

            triplets = np.array(triplets)
            heads, relations, tails = triplets.transpose()
            
            train_triplets = triplets[:train_few]   # 10 default
            test_triplets = triplets[train_few:]
            if (len(triplets)) - train_few < 1:
                    continue

            entities_list = self.entities_list  # including entites except the meta-testing.
            false_candidates = np.array(list(set(entities_list) - set(np.concatenate((heads, tails)))))
            
            # Support set
            s_false_entities = np.random.choice(false_candidates, size=train_few * self.args.negative_sample)
            s_neg_samples = np.tile(train_triplets, (self.args.negative_sample, 1))
            s_neg_samples[s_neg_samples[:, 0] == unseen_entity, 2] = s_false_entities[s_neg_samples[:, 0] == unseen_entity]
            s_neg_samples[s_neg_samples[:, 2] == unseen_entity, 0] = s_false_entities[s_neg_samples[:, 2] == unseen_entity]  
            support_pos_triplets.extend(train_triplets)
            support_neg_triplets.extend(s_neg_samples)

            # Query set
            q_false_entities = np.random.choice(false_candidates, size=(len(triplets) - train_few) * self.args.negative_sample)
            q_neg_samples = np.tile(test_triplets, (self.args.negative_sample, 1))
            q_neg_samples[q_neg_samples[:, 0] == unseen_entity, 2] = q_false_entities[q_neg_samples[:, 0] == unseen_entity]
            q_neg_samples[q_neg_samples[:, 2] == unseen_entity, 0] = q_false_entities[q_neg_samples[:, 2] == unseen_entity]  
            query_pos_triplets.extend(test_triplets)
            query_neg_triplets.extend(q_neg_samples)

            unseen_entity_embedding = self.embed(unseen_entity, train_triplets, self.use_cuda)
            total_unseen_entity.append(unseen_entity)
            total_unseen_entity_embedding.append(unseen_entity_embedding)

            # Support set
            support_pos_triplet = torch.LongTensor(train_triplets)
            support_neg_triplet = torch.LongTensor(s_neg_samples)
            query_pos_triplet = torch.LongTensor(test_triplets)
            query_neg_triplet = torch.LongTensor(q_neg_samples)
            s_emb_p = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, support_pos_triplet, 1)
            s_emb_n = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, support_neg_triplet, 0)
            q_emb_p = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, query_pos_triplet, 1)
            q_emb_n = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, query_neg_triplet, 0)

            # Encoder
            s_pos_r = self.latent_encoder(s_emb_p)
            s_neg_r = self.latent_encoder(s_emb_n)
            q_pos_r = self.latent_encoder(q_emb_p)
            q_neg_r = self.latent_encoder(q_emb_n)
            # for each unseen entity.
            c_r = torch.cat([s_pos_r, s_neg_r], dim=0)    # prior
            t_r = torch.cat([s_pos_r, s_neg_r, q_pos_r, q_neg_r], dim=0)   # posteior
            c_r = torch.mean(c_r, dim=0)
            t_r = torch.mean(t_r, dim=0)
            context_dists.append(c_r)
            target_dists.append(t_r)

        # cal KL-divergence
        context_dists = torch.cat(context_dists, dim=0).view(-1, 100).sum(dim=0)
        target_dists = torch.cat(target_dists, dim=0).view(-1, 100).sum(dim=0)
        context_dist = self.dist(context_dists)
        target_dist = self.dist(target_dists)
        z = target_dist.rsample()
        kld = kl_divergence(target_dist, context_dist)

        total_unseen_entity = np.array(total_unseen_entity)
        total_unseen_entity_embedding = torch.cat(total_unseen_entity_embedding).view(-1, self.embedding_size)

        query_pos_triplets = torch.LongTensor(np.array(query_pos_triplets))
        query_neg_triplets = torch.LongTensor(np.array(query_neg_triplets))
        
        query_emb_p = self.embed_concat(total_unseen_entity, total_unseen_entity_embedding, query_pos_triplets, 1)
        query_emb_n = self.embed_concat(total_unseen_entity, total_unseen_entity_embedding, query_neg_triplets, 0)
        total_query_embs = torch.cat([query_emb_p, query_emb_n], dim=0)

        # Decoder
        score = self.cal_score(total_query_embs, z)
        pos_score = score[:len(query_emb_p)]
        neg_score = score[len(query_emb_p):]
        y = torch.ones(len(query_emb_p) * self.args.negative_sample)
        if self.use_cuda:
            y = y.cuda()
        pos_score = pos_score.repeat(self.args.negative_sample)
        hinge_loss = F.margin_ranking_loss(pos_score, neg_score, y, margin=self.args.margin)

        kl_loss = kld.mean(0)

        loss = kl_loss + hinge_loss
        if (epoch + 1) % 100 == 0:
            print("Epoch: {} \t loss: {:.4f} \t kl: {:.4f} \t hingeloss: {:.4f}".format(epoch, loss, kl_loss, hinge_loss))
        # print("Epoch: {} \t loss: {} \t kl: {} \t hingeloss: {}".format(epoch, loss, kl_loss, hinge_loss))

        return loss
    
    def eval_one_time(self, eval_type):
        
        if eval_type == 'valid':
            test_task_dict = self.meta_valid_task_entity_to_triplets
            test_task_pool = list(self.meta_valid_task_entity_to_triplets.keys())
        elif eval_type == 'test':
            test_task_dict = self.meta_test_task_entity_to_triplets
            test_task_pool = list(self.meta_test_task_entity_to_triplets.keys())
        else:
            raise ValueError("Eval Type <{}> is Wrong".format(eval_type))

        total_ranks = []
        total_subject_ranks = []
        total_object_ranks = []

        total_unseen_entity = []
        total_unseen_entity_embedding = []
        
        total_test_triplets_dict = {}

        support_pos_triplets = []
        support_neg_triplets = []
        query_triplets = []

        context_dists = []
        target_dists = []
        
        for unseen_entity in test_task_pool:
    
            triplets = test_task_dict[unseen_entity]
            triplets = np.array(triplets)
            heads, relations, tails = triplets.transpose()

            train_triplets = triplets[:self.args.few]
            test_triplets = triplets[self.args.few:]

            entities_list = self.entities_list  # including entites except the meta-testing.
            false_candidates = np.array(list(set(entities_list) - set(np.concatenate((heads, tails)))))

            if (len(triplets)) - self.args.few < 1:
                continue
            # Support set
            s_false_entities = np.random.choice(false_candidates, size=self.args.few * self.args.negative_sample)
            s_neg_samples = np.tile(train_triplets, (self.args.negative_sample, 1))
            s_neg_samples[s_neg_samples[:, 0] == unseen_entity, 2] = s_false_entities[s_neg_samples[:, 0] == unseen_entity]
            s_neg_samples[s_neg_samples[:, 2] == unseen_entity, 0] = s_false_entities[s_neg_samples[:, 2] == unseen_entity]  
            support_pos_triplets.extend(train_triplets)
            support_neg_triplets.extend(s_neg_samples)

            query_triplets.extend(test_triplets)

            # Train (Inductive)
            unseen_entity_embedding = self.embed(unseen_entity, train_triplets, use_cuda = self.use_cuda)
            total_unseen_entity.append(unseen_entity)
            total_unseen_entity_embedding.append(unseen_entity_embedding)
            total_test_triplets_dict[unseen_entity] = torch.LongTensor(test_triplets)
            # Support set
            support_pos_triplet = torch.LongTensor(train_triplets)
            support_neg_triplet = torch.LongTensor(s_neg_samples)
            s_emb_p = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, support_pos_triplet, 1)
            s_emb_n = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, support_neg_triplet, 0)

            # Encoder
            s_pos_r = self.latent_encoder(s_emb_p)
            s_neg_r = self.latent_encoder(s_emb_n)
            # for each unseen entity.
            c_r = torch.cat([s_pos_r, s_neg_r], dim=0)
            c_r = torch.mean(c_r, dim=0)
            context_dists.append(c_r)
        
        context_dists = torch.cat(context_dists, dim=0).view(-1, 100).sum(dim=0)
        context_dist = self.dist(context_dists)
        z = context_dist.rsample()

        total_unseen_entity = np.array(total_unseen_entity)
        total_unseen_entity_embedding = torch.cat(total_unseen_entity_embedding).view(-1, self.embedding_size)

        # Decoder
        all_entity_embeddings = copy.deepcopy(self.embed.entity_embedding.weight).detach()
        all_relation_embeddings = copy.deepcopy(self.embed.relation_embedding).detach()

        ranks, ranks_s, ranks_o= cal_mrr(self.decoder, z, total_unseen_entity, total_unseen_entity_embedding, all_entity_embeddings, all_relation_embeddings, total_test_triplets_dict,
        self.all_triplets, use_cuda = self.use_cuda, score_function = self.args.score_function)

        if len(ranks_s) != 0:
            total_subject_ranks.append(ranks_s)

        if len(ranks_o) != 0:
            total_object_ranks.append(ranks_o)

        total_ranks.append(ranks)

        results = {}
                        
        # Subject
        total_subject_ranks = torch.cat(total_subject_ranks)
        total_subject_ranks += 1

        results['subject_ranks'] = total_subject_ranks
        results['subject_mrr'] = torch.mean(1.0 / total_subject_ranks.float()).item()

        for hit in [1, 3, 10]:
            avg_count = torch.mean((total_subject_ranks <= hit).float())
            results['subject_hits@{}'.format(hit)] = avg_count.item()

        # Object
        total_object_ranks = torch.cat(total_object_ranks)
        total_object_ranks += 1

        results['object_ranks'] = total_object_ranks
        results['object_mrr'] = torch.mean(1.0 / total_object_ranks.float()).item()

        for hit in [1, 3, 10]:
            avg_count = torch.mean((total_object_ranks <= hit).float())
            results['object_hits@{}'.format(hit)] = avg_count.item()

        # Total
        total_ranks = torch.cat(total_ranks)
        total_ranks += 1

        results['total_ranks'] = total_ranks
        results['total_mrr'] = torch.mean(1.0 / total_ranks.float()).item()

        for hit in [1, 3, 10]:
            avg_count = torch.mean((total_ranks <= hit).float())
            results['total_hits@{}'.format(hit)] = avg_count.item()

        return results


