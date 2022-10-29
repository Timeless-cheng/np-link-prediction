import torch
import random
import numpy as np

import os
import copy

from utils import load_data, load_processed_data
from models import EntityEmbedding

class DataLoader:
    '''
        Parallel the encoder of support set and query set.
        return: support/query set
    '''
    def __init__(self, args, task, all_entities, mode='train', meta_task_entity=None):
        
        self.args = args
        self.entities_list = all_entities
        self.triplets = task
        self.task = list(task.keys())
        self.mode = mode

    # Meta-Learning for Long-Tail Tasks
    def cal_train_few(self, epoch):

        if self.args.model_tail == 'log':
            for i in range(self.args.max_few):
                if epoch < (self.args.n_epochs / (2 ** i)):
                    continue
                return max(min(self.args.max_few, self.args.few + i - 1), self.args.few)
            return self.args.max_few
        else:
            return self.args.few
    
    def get_query_support(self, epoch):
    
        if self.mode == 'train':
            train_few = self.cal_train_few(epoch)
            random.shuffle(self.task)
            task_pool = self.task[:self.args.num_train_entity]
        else:
            task_pool = self.task
        
        total_unseen_entity = []

        support_pos_triplets = []
        support_neg_triplets = []
        query_pos_triplets = []
        query_neg_triplets = []

        for unseen_entity in task_pool:
            triplets = self.triplets[unseen_entity]
            triplets = np.array(triplets)
            heads, relations, tails = triplets.transpose()

            if self.mode == 'train':
                random.shuffle(triplets)
                if (len(triplets)) - train_few < 5:
                    continue
                train_triplets = triplets[:train_few]   # 10 default
                test_triplets = triplets[train_few:train_few+5]
            else:
                if (len(triplets)) - self.args.few < 1:
                    continue
                train_few = self.args.few
                train_triplets = triplets[:self.args.few]
                test_triplets = triplets[self.args.few:]

            entities_list = self.entities_list  # including entites except the meta-testing.
            false_candidates = np.array(list(set(entities_list) - set(np.concatenate((heads, tails)))))

            # Support set
            s_false_entities = np.random.choice(false_candidates, size=train_few * self.args.negative_sample)
            s_neg_samples = np.tile(train_triplets, (self.args.negative_sample, 1))
            s_neg_samples[s_neg_samples[:, 0] == unseen_entity, 2] = s_false_entities[s_neg_samples[:, 0] == unseen_entity]
            s_neg_samples[s_neg_samples[:, 2] == unseen_entity, 0] = s_false_entities[s_neg_samples[:, 2] == unseen_entity]
            support_pos_triplets.append(train_triplets)
            support_neg_triplets.append(s_neg_samples)

            # Query set
            if self.mode == 'train':
                q_false_entities = np.random.choice(false_candidates, size=5 * self.args.negative_sample)
                q_neg_samples = np.tile(test_triplets, (self.args.negative_sample, 1))
                q_neg_samples[q_neg_samples[:, 0] == unseen_entity, 2] = q_false_entities[q_neg_samples[:, 0] == unseen_entity]
                q_neg_samples[q_neg_samples[:, 2] == unseen_entity, 0] = q_false_entities[q_neg_samples[:, 2] == unseen_entity]  
                query_neg_triplets.append(q_neg_samples)
            query_pos_triplets.append(test_triplets)

            total_unseen_entity.append(unseen_entity)

        return [total_unseen_entity, support_pos_triplets, support_neg_triplets, query_pos_triplets, query_neg_triplets]


            
            

        


