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
   
    def __init__(self, args, use_cuda, all_triplets, meta_task_entity, num_entity, num_relation):
    
        super(NueralProcess, self).__init__()

        self.args = args
        self.use_cuda = use_cuda

        self.load_pretrain_embedding()
        self.embed = EntityEmbedding(self.args.embed_size, self.args.embed_size, num_entity, num_relation,
                            args = self.args, entity_embedding = self.pretrain_entity_embedding, relation_embedding = self.pretrain_relation_embedding)
        # MLP
        self.latent_encoder = LatentEncoder(embed_size=args.embed_size, num_hidden1=500, num_hidden2=200,
                                r_dim=args.embed_size, dropout_p=0.5)
        # MuSigma
        self.dist = MuSigmaEncoder(args.embed_size, args.embed_size)
        # Decoder
        self.decoder = Decoder(self.args, self.args.embed_size)
        
        # We set the embedding of unseen entities as the zero vector.
        self.embed.entity_embedding.weight.data[meta_task_entity] = torch.zeros(len(meta_task_entity), self.embedding_size)
    
        if self.use_cuda:
            self.all_triplets = all_triplets.cuda()

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


    def embed_concat(self, unseen_entities, total_unseen_entity_embedding, triplets):

        h, r, t = torch.t(triplets.view(-1, 3))

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


    def get_embeddings(self, data, mode='train'):
        total_unseen_entity, support_pos_triplets, support_neg_triplets, query_pos_triplets, query_neg_triplets = data

        total_unseen_entity_embedding = []

        support_pos_embeddings = []
        support_neg_embeddings = []
        query_pos_embeddings = []
        query_neg_embeddings = []

        total_test_triplets_dict = {}

        for idx, unseen_entity in enumerate(total_unseen_entity):

            # A RGCN
            unseen_entity_embedding = self.embed(unseen_entity, support_pos_triplets[idx], self.use_cuda)
            total_unseen_entity_embedding.append(unseen_entity_embedding)
            
            # Support set
            support_pos_triplet = torch.LongTensor(support_pos_triplets[idx])
            support_neg_triplet = torch.LongTensor(support_neg_triplets[idx])
            s_emb_p = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, support_pos_triplet, 1)
            s_emb_n = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, support_neg_triplet, 0)
            support_pos_embeddings.append(s_emb_p)
            support_neg_embeddings.append(s_emb_n)
            
            query_pos_triplet = torch.LongTensor(query_pos_triplets[idx])
            q_emb_p = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, query_pos_triplet, 1)
            query_pos_embeddings.append(q_emb_p)
            if mode == 'train':
                query_neg_triplet = torch.LongTensor(query_neg_triplets[idx])                
                q_emb_n = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, query_neg_triplet, 0)
                query_neg_embeddings.append(q_emb_n)
            else:
                total_test_triplets_dict[unseen_entity] = torch.LongTensor(query_pos_triplets[idx])
        
        support_pos_embeddings = torch.stack(support_pos_embeddings, 0)
        support_neg_embeddings = torch.stack(support_neg_embeddings, 0)
        if mode == 'train':
            query_pos_embeddings = torch.stack(query_pos_embeddings, 0)
            query_neg_embeddings = torch.stack(query_neg_embeddings, 0)
            return total_unseen_entity_embedding, support_pos_embeddings, support_neg_embeddings, query_pos_embeddings, query_neg_embeddings

        return total_test_triplets_dict, total_unseen_entity_embedding, support_pos_embeddings, support_neg_embeddings

    def forward(self, epoch, data_loaders):
        data = data_loaders.get_query_support(epoch)
        total_unseen_entity = data[0]

        # Encoder
        total_unseen_entity_embedding, s_p_emb, s_n_emb, q_p_emb, q_n_emb = self.get_embeddings(data)
        s_pos_r = self.latent_encoder(s_p_emb)
        s_neg_r = self.latent_encoder(s_n_emb)
        q_pos_r = self.latent_encoder(q_p_emb)
        q_neg_r = self.latent_encoder(q_n_emb)

        c_r = torch.cat([s_pos_r, s_neg_r], dim=1)    # prior
        t_r = torch.cat([s_pos_r, s_neg_r, q_pos_r, q_neg_r], dim=1)   # posteior
        context_dists = torch.mean(c_r, dim=1)    # [batch_size, 100]
        target_dists = torch.mean(t_r, dim=1)
        context_dist = self.dist(context_dists)
        target_dist = self.dist(target_dists)
        z = target_dist.rsample()
        kld = kl_divergence(target_dist, context_dist).sum(-1)

        # Decoder
        total_unseen_entity = np.array(total_unseen_entity)
        total_unseen_entity_embedding = torch.cat(total_unseen_entity_embedding).view(-1, self.embedding_size)

        query_pos_triplets = torch.LongTensor(np.array(data[-2]))
        query_neg_triplets = torch.LongTensor(np.array(data[-1]))

        query_emb_p = self.embed_concat(total_unseen_entity, total_unseen_entity_embedding, query_pos_triplets)
        query_emb_n = self.embed_concat(total_unseen_entity, total_unseen_entity_embedding, query_neg_triplets)
        total_query_embs = torch.cat([query_emb_p, query_emb_n], dim=0)
        
        # z: positive + negative
        z_pos = z.repeat(1, query_pos_triplets.shape[1]).view(-1, 100)
        z_neg = z.repeat(1, query_pos_triplets.shape[1] * self.args.negative_sample).view(-1, 100)
        z_all = torch.cat([z_pos, z_neg], dim=0)
        score = self.cal_score(total_query_embs, z_all)
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
        print("Epoch: {} \t loss: {} \t kl: {} \t hingeloss: {}".format(epoch, loss, kl_loss, hinge_loss))

        return loss
    
    def eval_one_time(self, data_loaders):
        
        data = data_loaders.get_query_support(0)
        total_unseen_entity = data[0]
        
        # Encoder
        total_test_triplets_dict, total_unseen_entity_embedding, s_p_emb, s_n_emb= self.get_embeddings(data, 'valid')
        s_pos_r = self.latent_encoder(s_p_emb)
        s_neg_r = self.latent_encoder(s_n_emb)
        c_r = torch.cat([s_pos_r, s_neg_r], dim=1)
        context_dists = torch.mean(c_r, dim=1)
        context_dist = self.dist(context_dists)
        z = context_dist.rsample()

        total_ranks = []
        total_subject_ranks = []
        total_object_ranks = []

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


