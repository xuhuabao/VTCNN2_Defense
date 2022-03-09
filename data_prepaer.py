import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽一般输出和警告
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import OneHotEncoder


# 读取数据
def prepare_data(filename):
    if (os.path.exists(filename) == False):
        f = open("F:/dataset/RML2016.10a_dict.pkl",'rb')
        # Xd 共220个对象  {('QPSK', 2): array(1000) ...}
        Xd = pickle.load(f,encoding='iso-8859-1')

        # 获取 信噪比 和 11种数据类型
        snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
        # X存储全部的训练数据1000   lbl存储全部的样本标签[('8PSK', -20)...] 1000个相同
        X = []
        lbl = []
        for mod in mods:
            for snr in snrs:
                X.append(Xd[(mod,snr)])
                for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
        X = np.vstack(X)

        print('dataset.shape: ',X.shape)
        # print(np.unique(lbl))

        #  数据分割
        #  into training and test sets of the form we can train/test on
        #  while keeping SNR and Mod labels handy for each
        np.random.seed(2016)
        n_examples = X.shape[0]
        n_train = n_examples * 0.8
        train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
        test_idx = list(set(range(0,n_examples))-set(train_idx))
        X_train = X[train_idx]
        X_test = X[test_idx]

        # one-hot encoder labels
        train_list = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
        test_list = list(map(lambda x: mods.index(lbl[x][0]), test_idx))
        Y_train = train_list
        Y_test = test_list
        # # 为了onehot 将每个标签单独划分
        # train_list = np.array(train_list).reshape(len(train_list),-1)
        # test_list = np.array(test_list).reshape(len(test_list),-1)
        # enc = OneHotEncoder()
        # enc.fit(train_list)
        # Y_train = enc.transform(train_list).toarray()
        # enc.fit(test_list)
        # Y_test = enc.transform(test_list).toarray()

        # 将数据按照信噪比提取   数据未打乱
        data_split_snr=[]
        ids = [i for i in range(220000)]
        ids_MODS = list(map(lambda x: lbl[x][0], ids))
        ids_SNRs = list(map(lambda x: lbl[x][1], ids))
        ids_MODS = np.array(ids_MODS)
        for snr in snrs:
            data_SNRs = X[np.where(np.array(ids_SNRs) == snr)]
            data_MODS = ids_MODS[np.where(np.array(ids_SNRs) == snr)]
            lbl_SNRS = [ mods.index(x) for x in data_MODS ]
            data_split_snr.append((np.array(data_SNRs),np.array(lbl_SNRS)))

        read_dictionary = {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test,
            'lbl': lbl,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'mods': mods,
            'snrs': snrs,
            'data_split_snr':data_split_snr
        }
        np.save(filename,read_dictionary)

def load_data(filename):
    read_dictionary = np.load(filename, allow_pickle=True).item()
    data_split_snr = read_dictionary['data_split_snr']
    snrs = read_dictionary['snrs']
    mods = read_dictionary['mods']
    print(snrs)
    print(mods)
    for (x,y) in data_split_snr:
        print(len(y), end=' ')



if __name__ == '__main__':
    filename = 'data/splits_data.npy'
    prepare_data(filename)
    load_data(filename)
