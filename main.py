import torch
import torch.optim as optim
from nueralprocess import NueralProcess

import logging
from tqdm import tqdm

from params import get_params
from utils import set_seed

logging.basicConfig(level=logging.INFO, filename='nell.log', filemode='a', format='%(asctime)s - %(levelname)s: %(message)s')

class Trainer(object):

    def __init__(self, args):

        super(Trainer, self).__init__()

        self.args = args
        # use cuda or not
        self.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(args.gpu)
            print('use cuda')

        self.model = NueralProcess(args, self.use_cuda)
            
        if self.use_cuda:
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
                
        self.best_mrr = 0

    def train(self):

        for epoch in tqdm(range(self.args.n_epochs)):
            self.model.train()
            loss = self.model.forward(epoch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Meta-Valid
            if epoch % self.args.evaluate_every == 0:
            # if 1:
                print("-------------------------------------Valid---------------------------------------")
                with torch.no_grad():
                    self.model.eval()
                    results = self.model.eval_one_time(eval_type='valid')
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
                        # patient = 0
                        self.best_mrr = mrr
                        torch.save({'state_dict': self.model.state_dict(), 'epoch': epoch}, './Checkpoints/{}/best_mrr_model.pth'.format(self.args.dataset))
                    
        # # For test
        checkpoint = torch.load('./Checkpoints/{}/best_mrr_model.pth'.format(self.args.dataset))
        self.model.load_state_dict(checkpoint['state_dict'])
        # print("Using best epoch: {}, {}".format(checkpoint['epoch'], self.exp_name))
        # Test
        with torch.no_grad():
            self.model.eval()
            results = self.model.eval_one_time(eval_type='test')
            mrr = results['total_mrr']
            logging.info("Test phase: ")
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
    
    trainer = Trainer(args)
    trainer.train()
