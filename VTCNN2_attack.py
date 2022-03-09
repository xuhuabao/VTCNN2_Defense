import os
import sys

import ops
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, Convolution2D, ZeroPadding2D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


class History:
    def __init__(self):
        self.epoch = []
        self.history={
            "loss":[],
            "val_loss":[],
            "accuracy":[],
            "val_accuracy":[]
        }


def train_loop(model, X_train, Y_train, X_test, Y_test, batch_size, nb_epoch, patience, modelpath):
    # perform training ...
    #   - call the main training loop in keras for our network+dataset
    history = model.fit(X_train,
                        Y_train,
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        verbose=2,
                        validation_data=(X_test, Y_test),
                        callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(modelpath, monitor='accuracy', verbose=0,
                                                               save_best_only=True, mode='auto'),
                            tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=patience, verbose=0,
                                                             mode='auto', restore_best_weights=True)
                            #  monitor='val_loss',min_delta=0,patience=0,verbose=0,mode='auto',
                            #                baseline=None, restore_best_weights=False):
                        ])
    # we re-load the best weights once training is finished
    print('the best weights was saved to:', modelpath)

    return history


def distillation_train_loop(model, x_train, y_train, x_test, y_test, batch_size, nb_epoch, patience, modelpath, T):
    best = 0
    best_epoch = 0
    n = 0

    y_train = tf.one_hot(y_train, 11)
    y_test = tf.one_hot(y_test, 11)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10 * batch_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    progress_bar_train = tf.keras.utils.Progbar(nb_epoch)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)  # +++++++++++++++++++++ False
    # history返回
    history = History()

    for epoch in range(nb_epoch):

        progress_bar_train.add(1)
        train_loss = tf.keras.metrics.Mean()
        test_loss = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()  # one hot labels
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()  # one hot labels

        for idx_train, (x, y) in enumerate(train_dataset):  #

            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                y_pred = tf.nn.softmax(y_pred / T)  # ++++++++++++++++++++++++++++++++++++++++++++
                loss = loss_fn(y, y_pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_accuracy(y, y_pred)

        for idx_test, (x, y) in enumerate(test_dataset):
            y_pred = model(x, training=False)
            y_pred = tf.nn.softmax(y_pred / T)  # ++++++++++++++++++++++++++++++++++++++++++++++++
            t_loss = loss_fn(y, y_pred)
            test_loss(t_loss)
            test_accuracy(y, y_pred)

        template = '\nEpoch {}, Train Loss: {:.2f}, Train Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}'
        print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100,
                              test_loss.result(), test_accuracy.result() * 100))

        history.epoch.append(epoch)
        history.history["loss"].append(train_loss.result())
        history.history["val_loss"].append(test_loss.result())
        history.history["accuracy"].append(train_accuracy.result())
        history.history["val_accuracy"].append(test_accuracy.result())

        if test_accuracy.result() * 100 > best:
            n = 0
            best = test_accuracy.result() * 100
            model.save_weights(modelpath)
            best_epoch = epoch
        else:
            n += 1
        if n > patience:
            break

    print("best model was saved at {}: test epoch={} best acc={}".format(modelpath, best_epoch, best))

    return history


def adversial_attack(eps_arr, attacksnr, model, x_test, y_test, isdefense):
    if not isdefense:
        figpath = 'output/Attack_Acc_normal_{}.png'.format(attacksnr)
        npypath = 'output/Attack_Acc_normal_{}.npy'.format(attacksnr)
    else:
        figpath = 'output/Attack_Acc_defense_{}.png'.format(attacksnr)
        npypath = 'output/Attack_Acc_defense_{}.npy'.format(attacksnr)

    clean_accs, fgm_accs, pgd_accs, mim_accs = [], [], [], []
    for eps in eps_arr:
        # show_adversial_img(model, e, x_test[0])
        clean_acc, fgm_acc, pgd_acc, mim_acc = ops.attack_step(model, x_test, y_test, eps)
        clean_accs.append(clean_acc)
        fgm_accs.append(fgm_acc)
        pgd_accs.append(pgd_acc)
        mim_accs.append(mim_acc)

    xlen = [i for i in range(len(eps_arr))]
    plt.plot(xlen, fgm_accs, marker='o', color='#FF5722', label='FGSM')
    plt.plot(xlen, pgd_accs, marker='+', color='#2196F3', label='PGD')
    plt.plot(xlen, mim_accs, marker='*', color='#9E9E9E', label='MIM')
    plt.plot(xlen, clean_accs, marker='^', color='black', label='No attack')
    plt.legend()
    if attacksnr == 10:
        plt.ylim(0, 75)
    else:
        plt.ylim(0, 40)
    plt.title(figpath)
    plt.xticks(xlen, eps_arr)
    plt.tight_layout()
    plt.savefig(figpath)
    plt.show()
    np.save(npypath, {
        'eps_arr': eps_arr,
        'clean': clean_accs,
        'fgm': fgm_accs,
        'pgd': pgd_accs,
        'mim': mim_accs})


# 10db/-10db下, 不同eps, ***************************  fig8  fig9  fig12  *******************************************
def attack_under_diff_eps_and_confusion(attacksnr, eps_arr, modelpath, isdefense):

    read_dictionary = ops.load_data(datafilepath, snr=attacksnr, all=False)
    x_test = read_dictionary['X_test']
    y_test = read_dictionary['Y_test']
    in_shp = list(x_test.shape[1:])
    mods = read_dictionary['mods']

    # init VTCNN2 model
    model = ops.getNNModel(in_shp, mods)
    model.load_weights(modelpath)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
    print('SNR={}, evaluate score:{},'.format(attacksnr, score))
    model.summary()
    # sys.exit(0)
    # do adversial attack , 10db/-10db , fig8
    adversial_attack(eps_arr, attacksnr, model, x_test, y_test, isdefense)
    # confusion matrix, fig9
    if attacksnr == 10:
        ops.get_confusion_matrix(model, attacksnr, mods, batch_size, x_test, y_test, eps=0.0015, isdefense=isdefense)


# ************************************************  fig10  fig11   *************************************************
# 不同snrs下的攻击后准确率  扰动为0.001  0.0015   fig10  fig11
def attack_under_diff_snrs(snrs, eps_arr, isdefense):
    # Set up some params
    clean_e1_arr, fgm_e1_arr, pgd_e1_arr, mim_e1_arr = [], [], [], []
    clean_e2_arr, fgm_e2_arr, pgd_e2_arr, mim_e2_arr = [], [], [], []

    for attacksnr in snrs:
        print('attack_under_diff_snrs: ', attacksnr)

        read_dictionary = ops.load_data(datafilepath, snr=attacksnr, all=False)
        x_train = read_dictionary['X_train']
        x_test = read_dictionary['X_test']
        y_train = read_dictionary['Y_train']
        y_test = read_dictionary['Y_test']
        mods = read_dictionary['mods']

        # init VTCNN2 model
        in_shp = list(x_train.shape[1:])
        # model = ops.getNNModel(in_shp, mods)

        if not isdefense:
            modelpath = 'models/VTCNN2_normal_{}.wts.h5'.format(attacksnr)

            # train normal models -- train_loop *******************************************
            # model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            #               metrics='accuracy')
            # history = train_loop(model, x_train, y_train, x_test, y_test, batch_size, nb_epoch, patience, modelpath)
            # ops.show_loss(history, figpth='output/Training_performance_normal_{}.png'.format(attacksnr))
        else:
            modelpath = 'models/VTCNN2_defense_{}.wts.h5'.format(attacksnr)
            # train defense models -- distillation_train_loop ****************************************
            # history = distillation_train_loop(model, x_train, y_train, x_test, y_test, batch_size,
            #                                   nb_epoch, patience, modelpath, T)
            # ops.show_loss(history, figpth='output/Training_performance_defense_{}.png'.format(attacksnr))

        model = ops.getNNModel(in_shp, mods)
        model.load_weights(modelpath)
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                      , metrics=['accuracy'])
        score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
        print('isdefense: {}, model evaluate snr={}, score: {}'.format(isdefense, attacksnr, score))

        # sys.exit(0)
        # 实施攻击获得acc  e1 for 0.001 ; e2 for 0.0015 ,  fig10  fig11
        clean_e1, fgm_e1, pgd_e1, mim_e1 = ops.attack_step(model, x_test, y_test, eps_arr[0])
        clean_e1_arr.append(clean_e1)
        fgm_e1_arr.append(fgm_e1)
        pgd_e1_arr.append(pgd_e1)
        mim_e1_arr.append(mim_e1)
        clean_e2, fgm_e2, pgd_e2, mim_e2 = ops.attack_step(model, x_test, y_test, eps_arr[1])
        clean_e2_arr.append(clean_e2)
        fgm_e2_arr.append(fgm_e2)
        pgd_e2_arr.append(pgd_e2)
        mim_e2_arr.append(mim_e2)

    # 绘制折线图  fig10
    if not isdefense:
        figpth = 'output/Attack_Under_Diff_SNRs_normal_{}_{}.png'
        npypth = 'output/Attack_Under_Diff_SNRs_normal_{}_{}.npy'
    else:
        figpth = 'output/Attack_Under_Diff_SNRs_defense_{}_{}.png'
        npypth = 'output/Attack_Under_Diff_SNRs_defense_{}_{}.npy'

    ops.plot_under_diff_snr_line(snrs, clean_e1_arr, fgm_e1_arr, pgd_e1_arr, mim_e1_arr,
                             figpth=figpth.format('line', eps_arr[0]), npypth=npypth.format('line', eps_arr[0]))
    ops.plot_under_diff_snr_line(snrs, clean_e2_arr, fgm_e2_arr, pgd_e2_arr, mim_e2_arr,
                             figpth=figpth.format('line', eps_arr[1]), npypth=npypth.format('line', eps_arr[1]))

    # 绘制柱状图  fig11
    ops.plot_under_diff_snr_bar(fgm_e1_arr, pgd_e1_arr, mim_e1_arr, titleFormat=eps_arr[0],
                            figpth=figpth.format('bar', eps_arr[0]))
    ops.plot_under_diff_snr_bar(fgm_e2_arr, pgd_e2_arr, mim_e2_arr, titleFormat=eps_arr[1],
                            figpth=figpth.format('bar', eps_arr[1]))


if __name__ == '__main__':
    # 8800*20(SNR Number)=176000, 11(Signal class)*800=8800,

    # Under Different SNRs  a) ε=0.001  b)ε=0.0007
    nb_epoch = 400  # number of epochs to train on
    batch_size = 1024  # training batch size
    patience = 30
    T = 100
    isdefense = False
    datafilepath = 'data/splits_data.npy'

    # ************************************************* 01 *********************************************************
    # eps_arr1 = [0.001, 0.0015]  # fig10  fig11
    # snrs = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    # # snrs = [-10, -8, 8, 10]
    # attack_under_diff_snrs(snrs=snrs, eps_arr=eps_arr1, isdefense=isdefense)  # fig10  fig11
    # sys.exit(0)

    # ************************************************** 02 ********************************************************
    # 不同eps的模型攻击折线图混淆矩阵, fig8  fig9  fig12
    # Todo 将模型训练从  attack_under_diff_eps_and_confusion里面取出单独处理
    attacksnr = 10  # SNR=10dB, SNR=-10dB
    eps_arr2 = [0.0000, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030]
    # eps_arr = [0.0000, 0.0005, 0.0025, 0.0030]
    if not isdefense:
        modelpath = 'models/VTCNN2_normal_{}.wts.h5'.format(attacksnr)
    else:
        modelpath = 'models/VTCNN2_defense_{}.wts.h5'.format(attacksnr)
    attack_under_diff_eps_and_confusion(attacksnr=attacksnr, eps_arr=eps_arr2, modelpath=modelpath, isdefense=isdefense)

