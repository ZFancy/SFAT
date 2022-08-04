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

def svhn_noniid_unequal(dataset, num_users):

    print(dataset)
    num_shards, num_imgs = 50, 1465 # 50 1465
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.labels[0:73250]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1 # 1
    max_shard = 30 # 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users



def cifar10_noniid_unequal(dataset, num_users):

    print(dataset)
    num_shards, num_imgs = 50, 1000 
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets[0:50000]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    print(random_shard_size)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users