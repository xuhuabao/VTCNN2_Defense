import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽一般输出和警告
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten,Convolution2D, ZeroPadding2D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method


# 读取数据
def load_data(filename, snr=10, all=True):
    # filename 数据集路径  snr信噪比   all是否加载全部数据
    if (os.path.exists(filename) == False):
        print('please prepare data first')
    else:
        if all:
            read_dictionary = np.load(filename, allow_pickle=True).item()
            # TODO 若不分snr 全部数据使用时要进行打乱  目前未用到此处
        else:
            read_dictionary = np.load(filename, allow_pickle=True).item()
            snrs = read_dictionary['snrs']
            data_split_snr = read_dictionary['data_split_snr']
            (x, y) = data_split_snr[snrs.index(snr)]
            np.random.seed(2016)
            state = np.random.get_state()
            np.random.shuffle(x)
            np.random.set_state(state)
            np.random.shuffle(y)
            index = int(len(x) * 0.8)
            x_train = x[:index]
            x_test = x[index:]
            y_train = y[:index]
            y_test = y[index:]
            read_dictionary['X_train'] = x_train
            read_dictionary['X_test'] = x_test
            read_dictionary['Y_train'] = y_train
            read_dictionary['Y_test'] = y_test
    return read_dictionary


def getNNModel(in_shp, classes):
    dr = 0.5  # dropout rate (%)
    model = tf.keras.Sequential()

    model.add(Reshape([1] + in_shp, input_shape=in_shp))
    model.add(ZeroPadding2D(padding=(0, 2), data_format='channels_first'))
    model.add(Convolution2D(256, (1, 3), padding='valid', data_format='channels_first',
                            activation="relu", name="conv1", kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2), data_format='channels_first'))
    model.add(Convolution2D(80, (2, 3), activation="relu", name="conv2", data_format='channels_first'))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense(len(classes), kernel_initializer='he_normal', name="dense2"))
    # model.add(Activation('softmax'))
    model.add(Reshape([len(classes)]))
    # 本例中样本标签不是one-hot 所以损失使用sparse_categorical_crossentropy，而非categorical_crossentropy
    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               optimizer='adam', metrics=['accuracy'])
    return model


# 绘制VTCNN模型训练的表现
def show_loss(history, figpth):
    # Show loss curves
    plt.figure()
    plt.title(figpth)
    # plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    # plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.plot(history.epoch, history.history['accuracy'], label='accuracy')
    plt.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figpth)
    plt.show()


# 返回在扰动率eps下的模型准确度
def attack_step(model, x_test, y_test, eps):
    # model 被攻击的模型  x_test原始样本   y_test原始样本标签   eps攻击的扰动率
    # Evaluate on clean and adversarial data
    test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
    test_acc_fgm = tf.metrics.SparseCategoricalAccuracy()
    test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()
    test_acc_mim = tf.metrics.SparseCategoricalAccuracy()

    progress_bar_test = tf.keras.utils.Progbar(y_test.shape[0])
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(512)
    for x, y in test_ds:
        y_pred = model.predict(x)
        test_acc_clean(y, y_pred)

        # fgm 攻击
        x_fgm = fast_gradient_method(model, x, eps, np.inf)
        y_pred_fgm = model.predict(x_fgm)
        test_acc_fgm(y, y_pred_fgm)

        # PGD 攻击 接受数据类型为float32 在  projected_gradient_descent 120行  eta = adv_x - x
        x_pgd = projected_gradient_descent(model, x, eps, eps / 10., 40, np.inf, rand_init=True)
        y_pred_pgd = model.predict(x_pgd)
        test_acc_pgd(y, y_pred_pgd)

        # MIM 攻击
        x_mim = momentum_iterative_method(model, x, eps, eps / 10., 40, np.inf)
        y_pred_mim = model.predict(x_mim)
        test_acc_mim(y, y_pred_mim)

        progress_bar_test.add(x.shape[0])

    clean_acc = test_acc_clean.result() * 100
    fgm_acc = test_acc_fgm.result() * 100
    pgd_acc = test_acc_pgd.result() * 100
    mim_acc = test_acc_mim.result() * 100
    print("test acc with eps={}\n \t clean examples {:.3f}(%) FGM examples {:.3f}(%) PGD examples {:.3f}(%) "
          " MIM examples {:.3f}(%) "
          .format(eps, clean_acc, fgm_acc, pgd_acc, mim_acc))

    return clean_acc, fgm_acc, pgd_acc, mim_acc


# 计算攻击前后的混淆矩阵
def get_confusion_matrix(model, attacksnr, classes, batch_size, x_test, y_test, eps, isdefense):
    # 攻击前后混淆矩阵
    conf_before = np.zeros([len(classes), len(classes)])
    conf_after = np.zeros([len(classes), len(classes)])

    # 计算攻击前的混淆矩阵
    predict_y_before = model.predict(x_test, batch_size=batch_size)

    for i in range(0, x_test.shape[0]):
        j = y_test[i]
        k = int(np.argmax(predict_y_before[i, :]))
        conf_before[j, k] = conf_before[j, k] + 1

    # 计算攻击后的混淆矩阵
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    for data, target in test_ds:
        # MIM 攻击
        x_mim = momentum_iterative_method(model, data, eps, eps / 10., 40, np.inf)
        predict_y_mim = model.predict(x_mim)

        for i in range(0, data.shape[0]):
            j = target[i]
            k = int(np.argmax(predict_y_mim[i, :]))
            conf_after[j, k] = conf_after[j, k] + 1

    if not isdefense:
        figpth1 = 'output/Confusion_matrix_normal_{}_snr_{}.pdf'.format('before attack', attacksnr)
        title1 = '{}, Before Attack'.format('Normal VT-CNN2')

        figpth2 = 'output/Confusion_matrix_normal_{}_snr_{}.pdf'.format('after attack', attacksnr)
        title2 = '{}, Generated with Mim'.format('Normal VT-CNN2')
    else:
        figpth1 = 'output/Confusion_matrix_defense_{}_snr_{}.pdf'.format('before attack', attacksnr)
        title1 = '{}, Before Attack'.format('Defense VT-CNN2')

        figpth2 = 'output/Confusion_matrix_defense_{}_snr_{}.pdf'.format('after attack', attacksnr)
        title2 = '{}, Generated with Mim'.format('Defense VT-CNN2')

    plot_confusion_matrix(conf_before, title=title1, figpth=figpth1, labels=classes)
    plot_confusion_matrix(conf_after, title=title2, figpth=figpth2, labels=classes)


# 将混淆矩阵绘制出来
def plot_confusion_matrix(cm, title, figpth, cmap=plt.cm.Blues, labels=[],):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label', {'size':12})
    plt.xlabel('Predicted label', {'size':12})

    # 显示文字
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < 100 else "white")

    plt.tight_layout()
    plt.savefig(figpth)
    plt.show()


# **********************************************************************************************************
# 绘制折线图 fig10
def plot_under_diff_snr_line(snrs, clean_accs, fgm_accs, pgd_accs, mim_accs, figpth, npypth):
    xlen = [i for i in range(len(snrs))]
    plt.plot(xlen, fgm_accs, marker='o', color='#FF5722', label='FGSM')
    plt.plot(xlen, pgd_accs, marker='+', color='#2196F3', label='PGD')
    plt.plot(xlen, mim_accs, marker='*', color='#9E9E9E', label='MIM')
    plt.plot(xlen, clean_accs, marker='^', color='#FFC107', label='No attack')
    plt.legend()
    plt.title(figpth)
    plt.xticks(xlen, snrs)
    plt.ylim(0, 80)
    plt.tight_layout()
    plt.savefig(figpth)
    plt.show()
    np.save(npypth, {
        'snrs': snrs,
        'clean': clean_accs,
        'fgm': fgm_accs,
        'pgd': pgd_accs,
        'mim': mim_accs})


# 绘制柱状图 fig11
def plot_under_diff_snr_bar(fgm_acc, pgd_acc, mim_acc, titleFormat,
                            figpth):
    num_list = [np.mean(fgm_acc), np.mean(pgd_acc), np.mean(mim_acc)]

    x_tick = ['FGSM', 'PGD', 'MIM']
    color = ['#FFEB3B', '#8BC34A', '#F44336']
    title = "Average Accuracy under All SNRs with  ε = {}".format(titleFormat)
    # title = figpth + ' ε = {}'.format(titleFormat)

    plt.title(title)
    plt.xlabel("Attack Methods")
    plt.ylabel("Accuracy")
    if titleFormat == 0.001:
        plt.ylim(25, 60)
    elif titleFormat == 0.0015:
        plt.ylim(15, 60)
    # 自定义legend()
    fgm_patch = mpatches.Patch(color=color[0], label=x_tick[0])
    pgd_patch = mpatches.Patch(color=color[1], label=x_tick[1])
    mim_patch = mpatches.Patch(color=color[2], label=x_tick[2])
    plt.legend(handles=[fgm_patch, pgd_patch, mim_patch])
    # 柱状图绘制
    bar = plt.bar(range(len(num_list)), num_list, width=0.5, color=color, tick_label=x_tick)
    plt.bar_label(bar, label_type='edge')
    plt.tight_layout()
    plt.savefig(figpth)
    plt.show()
