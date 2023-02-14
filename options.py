import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E, 2 for SVHN, 3 for CIFAR-100")
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size: B, 128 for SVHN and CIFAR-100")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')

    # other arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--dataset', type=str, default='cifar-10', help="name \
                        of dataset: cifar-10, svhn, cifar-100")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default='0', help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=6, help='random seed')
    parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step K')
    parser.add_argument('--out-dir',type=str,default='../SFAT_result',help='dir of output')
    parser.add_argument('--agg-opt',type=str,default='FedAvg',help='option of on-device learning: FedAvg, FedProx, Scaffold')
    parser.add_argument('--agg-center',type=str,default='FedAvg',help='option of aggregation: FedAvg, SFAT')
    parser.add_argument('--mu', type=float, default=0.01, help='mu for FedProx')
    parser.add_argument('--modeltype',type=str,default='NIN',help='different model structure')
    parser.add_argument('--pri', type=float, default=1.4, help='weight for (1+alpha)/(1-alpha): 1.2, 1.4, 1.6 ...')
    parser.add_argument('--topk',type=int, default=1, help='top client to be upweight')
    parser.add_argument('--train-method',type=str,default='AT',help='different training method: AT, TRADES, MART')
  
    args = parser.parse_args()
    return args
