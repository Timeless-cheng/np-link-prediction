import random
import torch
import os
import tqdm
import pickle
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Data load and Preprocess
def load_data(file_path):
    '''
        argument:
            file_path: ./Dataset/raw_data/FB15k-237
        
        return:
            entity2id, relation2id, train_triplets, valid_triplets, test_triplets
    '''

    print("load data from {}".format(file_path))

    with open(os.path.join(file_path, 'entity2id.txt')) as f:
        entity2id = dict()

        for line in f:
            entity, eid = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relation2id.txt')) as f:
        relation2id = dict()

        for line in f:
            relation, rid = line.strip().split('\t')
            relation2id[relation] = int(rid)

    train_triplets = read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
    valid_triplets = read_triplets(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
    test_triplets = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets

def read_triplets(file_path, entity2id, relation2id):
    
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return np.array(triplets)

def load_processed_data(file_path):

    # with open(os.path.join(file_path, 'filtered_triplets.pickle'), 'rb') as f:
    #     filtered_triplets = pickle.load(f)

    # with open(os.path.join(file_path, 'meta_train_task_triplets.pickle'), 'rb') as f:
    #     meta_train_task_triplets = pickle.load(f)

    # with open(os.path.join(file_path, 'meta_valid_task_triplets.pickle'), 'rb') as f:
    #     meta_valid_task_triplets = pickle.load(f)

    # with open(os.path.join(file_path, 'meta_test_task_triplets.pickle'), 'rb') as f:
    #     meta_test_task_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_train_task_entity_to_triplets.pickle'), 'rb') as f:
        meta_train_task_entity_to_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_valid_task_entity_to_triplets.pickle'), 'rb') as f:
        meta_valid_task_entity_to_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_test_task_entity_to_triplets.pickle'), 'rb') as f:
        meta_test_task_entity_to_triplets = pickle.load(f)

    # return filtered_triplets, meta_train_task_triplets, meta_valid_task_triplets, meta_test_task_triplets, \
    #         meta_train_task_entity_to_triplets, meta_valid_task_entity_to_triplets, meta_test_task_entity_to_triplets

    return meta_train_task_entity_to_triplets, meta_valid_task_entity_to_triplets, meta_test_task_entity_to_triplets


def sort_and_rank(score, target):
    x, indices = torch.sort(score, dim=1, descending=True)  # 按行降序排列
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def cal_mrr(decoder, z, total_unseen_entity, total_unseen_entity_embedding, all_entity_embeddings, all_relation_embeddings, total_test_triplets_dict, all_triplets, use_cuda, score_function):
    
    num_entity = len(all_entity_embeddings)
    subject_count = 0
    object_count = 0
    all_entity_embeddings[total_unseen_entity] = total_unseen_entity_embedding

    ranks = []
    ranks_s = []
    ranks_o = []

    head_relation_triplets = all_triplets[:, :2]
    tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

    for idx, (entity, test_triplets) in enumerate(total_test_triplets_dict.items()):
    # 对于每个测试的元组都计算一个可能成为其头节点或者尾节点的entity

        if use_cuda:
            device = torch.device('cuda')
            test_triplets = test_triplets.cuda()

        for test_triplet in test_triplets:

            if test_triplet[0] == entity:

                subject_count += 1

                subject = test_triplet[0]
                relation = test_triplet[1]
                object_ = test_triplet[2]

                subject_relation = torch.LongTensor([subject, relation])
                if use_cuda:
                    subject_relation = subject_relation.cuda()

                # 原头关系元组中含有关系则删掉
                delete_index = torch.sum(head_relation_triplets == subject_relation, dim = 1)
                delete_index = torch.nonzero(delete_index == 2).squeeze()
                
                if use_cuda:
                    device = torch.device('cuda')
                    # 得到应该有的tail node，然后从整体中删除tail entity。
                    delete_entity_index = all_triplets[delete_index, 2].view(-1).cpu().numpy()
                    perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                    perturb_entity_index = torch.from_numpy(perturb_entity_index).to(device)
                    perturb_entity_index = torch.cat((object_.view(-1), perturb_entity_index))
                else:
                    delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
                    perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                    perturb_entity_index = torch.from_numpy(perturb_entity_index)
                    perturb_entity_index = torch.cat((object_.view(-1), perturb_entity_index))

                # Score
                if score_function == 'DistMult':

                    emb_ar = decoder(all_entity_embeddings[subject], z[idx]) * all_relation_embeddings[relation]
                    emb_ar = emb_ar.view(-1, 1, 1)

                    emb_c = decoder(all_entity_embeddings[perturb_entity_index], z[idx])
                    emb_c = emb_c.transpose(0, 1).unsqueeze(1)

                    out_prod = torch.bmm(emb_ar, emb_c)
                    score = torch.sum(out_prod, dim = 0)
                    
                elif score_function == 'TransE':

                    # head_embedding = all_entity_embeddings[subject]
                    # relation_embedding = all_relation_embeddings[relation]
                    # tail_embeddings = all_entity_embeddings[perturb_entity_index]
                    head_embedding = decoder(all_entity_embeddings[subject], z[idx])
                    relation_embedding = all_relation_embeddings[relation]
                    tail_embeddings = decoder(all_entity_embeddings[perturb_entity_index], z[idx])
                    
                    score = - torch.norm((head_embedding + relation_embedding - tail_embeddings), p = 2, dim = 1)
                    score = score.view(1, -1)

                else:

                    raise TypeError

                # Cal Rank
                # 第一个值【0】是真实标签，看真实的entity在上述序列中能排多少位？
                if use_cuda:
                    target = torch.tensor(0).to(device)
                    ranks_s.append(sort_and_rank(score, target).cpu())

                else:
                    target = torch.tensor(0)
                    score = score.cpu()
                    ranks_s.append(sort_and_rank(score, target))


            elif test_triplet[2] == entity:

                object_count += 1

                subject = test_triplet[0]
                relation = test_triplet[1]
                object_ = test_triplet[2]

                object_relation = torch.LongTensor([object_, relation])
                if use_cuda:
                    object_relation = object_relation.cuda()

                delete_index = torch.sum(tail_relation_triplets == object_relation, dim = 1)
                delete_index = torch.nonzero(delete_index == 2).squeeze()

                if use_cuda:
                    device = torch.device('cuda')
                    delete_entity_index = all_triplets[delete_index, 0].view(-1).cpu().numpy()
                    perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                    perturb_entity_index = torch.from_numpy(perturb_entity_index).to(device)
                    perturb_entity_index = torch.cat((subject.view(-1), perturb_entity_index))
                else:
                    delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
                    perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                    perturb_entity_index = torch.from_numpy(perturb_entity_index)
                    perturb_entity_index = torch.cat((subject.view(-1), perturb_entity_index))

                # Score
                if score_function == 'DistMult':

                    emb_ar = all_entity_embeddings[object_] * all_relation_embeddings[relation]
                    emb_ar = emb_ar.view(-1, 1, 1)

                    emb_c = all_entity_embeddings[perturb_entity_index]
                    emb_c = emb_c.transpose(0, 1).unsqueeze(1)

                    out_prod = torch.bmm(emb_ar, emb_c)
                    score = torch.sum(out_prod, dim = 0)
                    
                elif score_function == 'TransE':

                    head_embeddings = all_entity_embeddings[perturb_entity_index]
                    relation_embedding = all_relation_embeddings[relation]
                    tail_embedding = all_entity_embeddings[object_]

                    score = head_embeddings + relation_embedding - tail_embedding
                    score = - torch.norm(score, p = 2, dim = 1)
                    score = score.view(1, -1)

                else:

                    raise TypeError

                # Cal Rank
                if use_cuda:
                    target = torch.tensor(0).to(device)
                    ranks_o.append(sort_and_rank(score, target).cpu())

                else:
                    target = torch.tensor(0)
                    score = score.cpu()
                    ranks_o.append(sort_and_rank(score, target))

            else:
                
                raise TypeError

    if subject_count == 0:
        ranks_o = torch.cat(ranks_o)
        ranks = ranks_o

    elif object_count == 0:
        ranks_s = torch.cat(ranks_s)
        ranks = ranks_s

    else:
        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)
        ranks = torch.cat((ranks_s, ranks_o))

    return ranks, ranks_s, ranks_o


