import torch
import torch.optim as optim
from data_loader import DataLoader
from nueralprocess import NueralProcess

import numpy as np
import logging
from tqdm import tqdm

from params import get_params
from utils import set_seed, load_data, load_processed_data

logging.basicConfig(level=logging.INFO, filename='log.log', filemode='a', format='%(asctime)s - %(levelname)s: %(message)s')

class Trainer(object):

    def __init__(self, args, data_loaders, all_triplets, meta_task_entity, num_entity, num_relation):

        super(Trainer, self).__init__()

        self.args = args
        self.all_triplets = all_triplets
        self.meta_task_entity = meta_task_entity
        self.train_loader = data_loaders[0]
        self.val_loader = data_loaders[1]
        self.test_loader = data_loaders[2]

        # use cuda or not
        self.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(args.gpu)
            print('use cuda')

        self.model = NueralProcess(args, self.use_cuda, self.all_triplets, self.meta_task_entity, num_entity, num_relation)
            
        if self.use_cuda:
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
                
        self.best_mrr = 0

    def train(self):

        for epoch in tqdm(range(self.args.n_epochs)):
            self.model.train()
            loss = self.model.forward(epoch, self.train_loader)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Meta-Valid
            # if epoch % self.args.evaluate_every == 0:
            if 1:
                print("-------------------------------------Valid---------------------------------------")
                with torch.no_grad():
                    self.model.eval()
                    results = self.model.eval_one_time(self.val_loader)
                    mrr = results['total_mrr']
                    logging.info("Epoch: {} - Validation".format(epoch))
                    logging.info("Total MRR (filtered): {:.6f}".format(results['total_mrr']))
                    logging.info("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1']))
                    logging.info("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3']))
                    logging.info("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10']))
                    print("Total MRR (filtered): {:.6f}".format(results['total_mrr']))
                    print("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1']))
                    print("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3']))
                    print("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10']))
                    if mrr > self.best_mrr:
                        self.best_mrr = mrr
                        torch.save({'state_dict': self.model.state_dict(), 'epoch': epoch}, './Checkpoints/{}/best_mrr_model.pth'.format(self.args.dataset))
        
        # For test
        checkpoint = torch.load('./Checkpoints/{}/best_mrr_model.pth'.format(self.args.dataset))
        self.model.load_state_dict(checkpoint['state_dict'])
        print("Using best epoch: {}, {}".format(checkpoint['epoch'], self.exp_name))
        # Test
        with torch.no_grad():
            self.model.eval()
            results = self.model.eval_one_time(self.test_loader)
            mrr = results['total_mrr']
            logging.info("Total MRR (filtered): {:.6f}".format(results['total_mrr']))
            logging.info("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1']))
            logging.info("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3']))
            logging.info("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10']))
            print("Total MRR (filtered): {:.6f}".format(results['total_mrr']))
            print("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1']))
            print("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3']))
            print("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10']))


if __name__ == '__main__':

    args = get_params()
    logging.info(args)
    print(args)
    
    set_seed(args.seed)
    entity2id, relation2id, train_triplets, valid_triplets, test_triplets = load_data('./Dataset/raw_data/{}'.format(args.dataset))

    meta_train_task_entity_to_triplets, meta_valid_task_entity_to_triplets, meta_test_task_entity_to_triplets \
            = load_processed_data('./Dataset/processed_data/{}'.format(args.dataset))

    meta_task_entity = np.concatenate((list(meta_valid_task_entity_to_triplets.keys()), list(meta_test_task_entity_to_triplets.keys())))
    entities_list = np.delete(np.arange(len(entity2id)), meta_task_entity)

    all_triplets = torch.LongTensor(np.concatenate((train_triplets, valid_triplets, test_triplets)))

    # return: [unseen_entities, support_pos, support_neg, query_pos, query_neg]
    train_data_loader = DataLoader(args, meta_train_task_entity_to_triplets, entities_list, 'train', meta_task_entity)
    val_data_loader = DataLoader(args, meta_valid_task_entity_to_triplets, entities_list, 'valid')
    test_data_loader = DataLoader(args, meta_test_task_entity_to_triplets, entities_list, 'test')
    data_loaders = [train_data_loader, val_data_loader, test_data_loader]
    
    trainer = Trainer(args, data_loaders, all_triplets, meta_task_entity, len(entity2id), len(relation2id))
    trainer.train()