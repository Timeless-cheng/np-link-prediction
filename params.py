import argparse

def get_params():
    parser = argparse.ArgumentParser(description='NP for unseen entity prediction')
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--n-epochs", type=int, default=25000)
    parser.add_argument("--dataset", type=str, default='FB15k-237')
    parser.add_argument("--negative-sample", type=int, default=16)   # 32

    # parser.add_argument("--pre-train", action='store_true')
    # parser.add_argument("--fine-tune", action='store_true')
    parser.add_argument("--pre-train", default=True)
    parser.add_argument("--fine-tune", default=True)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--pre-train-model", type=str, default='DistMult')
    parser.add_argument("--pre-train-emb-size", type=str, default='100')
    parser.add_argument("--embed-size", type=int, default=100, help="100 for FB15k-237")
    parser.add_argument("--num-train-entity", type=int, default=500)
    parser.add_argument("--model", type=str, default='TransGEN')
    parser.add_argument("--score-function", type=str, default='DistMult')

    parser.add_argument("--bases", type=int, default=100)
    parser.add_argument("--model-tail", type=str, default='log')
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--few", type=int, default=3)
    parser.add_argument("--max-few", type=int, default=10)

    parser.add_argument("--evaluate_every", type=int, default=1000)

    args = parser.parse_args()
    
    return args


