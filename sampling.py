import numpy as np
from torchvision import datasets, transforms

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid_skew(dataset, num_users, skew=2):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    number_each_class = 5000
    num_shards, num_imgs = int(num_users*2), int(25000/num_users)
    idx_shard = [i for i in range(num_shards)]

    # divide the data by the class labels equally among K client
    new_datas = [[] for _ in range(10)]
    M_k = [[] for _ in range(num_users)]

    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    for i in range(10):
        new_datas[i] = idxs[i*number_each_class:(i+1)*number_each_class]
        print(len(new_datas[i]))
    
    for i in range(num_users):
        M_k[i] = idxs[i*int(len(idxs)/num_users):(i+1)*int(len(idxs)/num_users)]
    for i in range(num_users):
        print(len(M_k[i]))
    kk = len(M_k[0])
    for i in range(num_users):
        for j in range(num_users):
            if i == j:
                rand_set = set(np.random.choice(M_k[j], int(kk*(100-(num_users-1)*skew)/100),replace=False))
                M_k[j] = list(set(M_k[j])-rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], [rand]), axis=0)
            else:
                rand_set = set(np.random.choice(M_k[j], int(kk*(skew)/100),replace=False))
                M_k[j] = list(set(M_k[j])-rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], [rand]), axis=0)                
    for i in range(num_users):
        print(len(dict_users[i]))
    return dict_users
    
    
def svhn_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users 
    
def svhn_noniid_skew(dataset, num_users, skew=2):
    """
    Sample non-I.I.D client data from SVHN dataset
    :param dataset:
    :param num_users:
    :return:
    """
    number_each_class = int(73257/5)
    num_shards, num_imgs = int(num_users*2), int(73257/(num_users*2))
    idx_shard = [i for i in range(num_shards)]

    # divide the data by the class labels equally among K client
    new_datas = [[] for _ in range(10)]
    M_k = [[] for _ in range(num_users)]

    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(73257)
    # labels = dataset.train_labels.numpy()
    labels = dataset.labels

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    for i in range(10):
        new_datas[i] = idxs[i*number_each_class:(i+1)*number_each_class]
        print(len(new_datas[i]))
    
    for i in range(num_users):
        M_k[i] = idxs[i*int(len(idxs)/num_users):(i+1)*int(len(idxs)/num_users)]
    for i in range(num_users):
        print(len(M_k[i]))
    kk = len(M_k[0])
    for i in range(num_users):
        for j in range(num_users):
            if i == j:
                rand_set = set(np.random.choice(M_k[j], int(kk*(100-(num_users-1)*skew)/100),replace=False))
                M_k[j] = list(set(M_k[j])-rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], [rand]), axis=0)
            else:
                rand_set = set(np.random.choice(M_k[j], int(kk*(skew)/100),replace=False))
                M_k[j] = list(set(M_k[j])-rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], [rand]), axis=0)                
    for i in range(num_users):
        print(len(dict_users[i]))
    return dict_users

def cifar100_noniid_skew(dataset, num_users, skew=2):
    """
    Sample non-I.I.D client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    number_each_class = 500
    num_shards, num_imgs = int(num_users*2), int(25000/num_users)
    idx_shard = [i for i in range(num_shards)]

    # divide the data by the class labels equally among K client
    new_datas = [[] for _ in range(100)]
    M_k = [[] for _ in range(num_users)]

    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    for i in range(100):
        new_datas[i] = idxs[i*number_each_class:(i+1)*number_each_class]
        print(len(new_datas[i]))
    
    for i in range(num_users):
        M_k[i] = idxs[i*int(len(idxs)/num_users):(i+1)*int(len(idxs)/num_users)]
    for i in range(num_users):
        print(len(M_k[i]))
    kk = len(M_k[0])
    for i in range(num_users):
        for j in range(num_users):
            if i == j:
                rand_set = set(np.random.choice(M_k[j], int(kk*(100-(num_users-1)*skew)/100),replace=False))
                M_k[j] = list(set(M_k[j])-rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], [rand]), axis=0)
            else:
                rand_set = set(np.random.choice(M_k[j], int(kk*(skew)/100),replace=False))
                M_k[j] = list(set(M_k[j])-rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], [rand]), axis=0)                
    for i in range(num_users):
        print(len(dict_users[i]))
    return dict_users
