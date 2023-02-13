import copy
import torch
from torchvision import datasets, transforms
from sampling import cifar_iid, svhn_iid, cifar_noniid_skew, svhn_noniid_skew, cifar100_noniid_skew, svhn_noniid_unequal, cifar10_noniid_unequal


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'cifar-10':
        data_dir = '../data/cifar-10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = cifar10_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid_skew(train_dataset, args.num_users)

    elif args.dataset == 'svhn':
        data_dir = '../data/svhn/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                       transform=apply_transform)

        test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = svhn_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = svhn_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                #print(train_dataset.labels)
                user_groups = svhn_noniid_skew(train_dataset, args.num_users)

    elif args.dataset == 'cifar-100':
        data_dir = '../data/cifar-100/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar100_noniid_skew(train_dataset, args.num_users)


    return train_dataset, test_dataset, user_groups

# FedAvg
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        print(key)
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

# # FedAvg unequal
def average_weights_unequal(w, idx_num):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        print(key)
        w_avg[key] = w_avg[key] * float(idx_num[0]*len(w)/sum(idx_num))
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * float(idx_num[i]*len(w)/sum(idx_num))
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
    
# SFAT
def average_weights_alpha(w, lw, idx, p):
    """
    Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        cou = 0
        if (lw[0] >= idx):
            w_avg[key] = w_avg[key] * p
        for i in range(1, len(w)):
            if (lw[i] >= idx) and (('bn' not in key)):
                w_avg[key] = w_avg[key] + w[i][key] * p
            else:
                cou += 1 
                w_avg[key] = w_avg[key] + w[i][key]
        w_avg[key] = torch.div(w_avg[key], cou+(len(w)-cou)*p)
    return w_avg


# # SFAT unequal
def average_weights_alpha_unequal(w, lw, idx, p, idx_num):
    """
    Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        cou = 0
        if (lw[0] >= idx):
            w_avg[key] = w_avg[key] * p * float(idx_num[0]*len(w)/sum(idx_num))
        else:
            w_avg[key] = w_avg[key] * float(idx_num[0]*len(w)/sum(idx_num))
        for i in range(1, len(w)):
            if (lw[i] >= idx) and (('bn' not in key)):
                w_avg[key] = w_avg[key] + w[i][key] * p * float(idx_num[i]*len(w)/sum(idx_num))
            else:
                cou += 1 
                w_avg[key] = w_avg[key] + w[i][key] * float(idx_num[i]*len(w)/sum(idx_num))
        w_avg[key] = torch.div(w_avg[key], cou+(len(w)-cou)*p)
    return w_avg  

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
