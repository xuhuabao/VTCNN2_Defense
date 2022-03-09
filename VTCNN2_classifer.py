import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import OneHotEncoder
from model import get_model

config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# 读取数据
def prepare_data(filename):
    if( os.path.exists(filename) == False):
        f = open("F:/dataset/RML2016.10a_dict.pkl",'rb')
        Xd = pickle.load(f,encoding='iso-8859-1')
        snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
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
        train_list = np.array(train_list).reshape(len(train_list),-1)
        test_list = np.array(test_list).reshape(len(test_list),-1)
        enc = OneHotEncoder()
        enc.fit(train_list)
        Y_train = enc.transform(train_list).toarray()
        enc.fit(test_list)
        Y_test = enc.transform(test_list).toarray()

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
        }
        np.save(filename,read_dictionary)
    else:
        read_dictionary = np.load(filename, allow_pickle=True).item()

    return read_dictionary


def train_loop(model, X_train,Y_train,X_test,Y_test,batch_size,nb_epoch,filepath):
    # perform training ...
    #   - call the main training loop in keras for our network+dataset
    history = model.fit(X_train,
                        Y_train,
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        verbose=2,
                        validation_data=(X_test, Y_test),
                        callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                                               save_best_only=True, mode='auto'),
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                        ])
    # we re-load the best weights once training is finished
    print('the best weights was saved to:',filepath)

    return history


def train_loop2(model, train_dataset, test_dataset, nb_epoch, filepath):
    best = 0
    n = 0
    progress_bar_train = tf.keras.utils.Progbar(nb_epoch)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)  # ++++++++++++++++++++++++++++++++

    # callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
    #                                        save_best_only=True, mode='auto'),
    #     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    # ]

    for epoch in range(nb_epoch):

        progress_bar_train.add(1)
        train_loss = tf.keras.metrics.Mean()
        test_loss = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()  # one-hot labels
        # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()  # integer labels

        for idx_train, (x, y) in enumerate(train_dataset):  #

            with tf.GradientTape() as tape:
                y_pred = model(x)
                y_pred = tf.nn.softmax(y_pred/1)  # ++++++++++++++++++++++++++++++++++++++++++++
                loss = loss_object(y, y_pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_accuracy(y, y_pred)

        for idx_test, (x, y) in enumerate(test_dataset):
            y_pred = model(x, training=False)
            y_pred = tf.nn.softmax(y_pred/1)  # ++++++++++++++++++++++++++++++++++++++++++++++++
            t_loss = loss_object(y, y_pred)
            test_loss(t_loss)
            test_accuracy(y, y_pred)

        if test_accuracy.result()*100 > best:
            n = 0
            best = test_accuracy.result()*100
            model.save_weights(filepath)
        else:
            n += 1
        if n > 4:
            break

        template = 'Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100,
                              test_loss.result(), test_accuracy.result()*100))


def show_loss(history):
    # Show loss curves
    plt.figure()
    plt.title('Training performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    plt.savefig('output/tvcnn2_classifer/Training performance.png')
    plt.show()


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('output/tvcnn2_classifer/'+title+'.png')
    plt.show()


def plot_confusion_matrix_for_snrs(model,snrs,x_test,y_test,classes,lbl,test_idx):
    # Plot confusion matrix
    acc = {}
    for snr in snrs:
        # extract classes @ SNR
        test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
        test_X_i = x_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = y_test[np.where(np.array(test_SNRs) == snr)]

        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes), len(classes)])
        confnorm = np.zeros([len(classes), len(classes)])
        for i in range(0, test_X_i.shape[0]):
            j = list(test_Y_i[i, :]).index(1)
            k = int(np.argmax(test_Y_i_hat[i, :]))
            conf[j, k] = conf[j, k] + 1
        for i in range(0, len(classes)):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
        plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)" % (snr))

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print("Overall Accuracy: ", cor / (cor + ncor))
        acc[snr] = 1.0 * cor / (cor + ncor)

    return acc


if __name__ == '__main__':

    filename = 'data/splits_data.npy'
    read_dictionary = prepare_data(filename)
    x_train =read_dictionary['X_train']
    x_test =read_dictionary['X_test']
    y_train =read_dictionary['Y_train']
    y_test =read_dictionary['Y_test']
    mods =read_dictionary['mods']
    snrs =read_dictionary['snrs']
    lbl =read_dictionary['lbl']
    test_idx =read_dictionary['test_idx']

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=11)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=11)

    in_shp = list(x_train.shape[1:])
    classes = mods
    print('x_train.shape, in_shp :', x_train.shape, in_shp)  # (17600, 2, 128)
    print('classes :',classes)
    # sys.exit(0)
    # in_shp=[2,128]
    # classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

    # init TVCNN2 model
    model = get_model()
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  optimizer='adam', metrics=['accuracy'])

    # Set up some params
    nb_epoch = 100  # number of epochs to train on
    batch_size = 1024  # training batch size
    filepath = 'models/convmodrecnets_CNN2_0.5.wts.h5'  # checkpoints saved

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10 * batch_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # train model and show loss curves
    # history = train_loop(model, x_train, y_train, x_test, y_test, batch_size, nb_epoch, filepath)
    # train_loop2(model, train_dataset, test_dataset, nb_epoch, filepath)
    # show_loss(history)
    # sys.exit(0)
    model.load_weights(filepath)
    score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
    print('model evaluate score:', score)
    # exit(0)

    # Plot confusion matrix
    test_Y_hat = model.predict(x_test, batch_size=batch_size)
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, x_test.shape[0]):
        j = list(y_test[i, :]).index(1)
        k = int(np.argmax(test_Y_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    plot_confusion_matrix(confnorm, labels=classes)

    # Plot confusion matrix
    accs = plot_confusion_matrix_for_snrs(model,snrs,x_test,y_test,classes,lbl,test_idx)

    if not os.path.exists('output/tvcnn2_classifer/accs.npy'):
        np.save('output/tvcnn2_classifer/accs.npy',accs)
    # accs = np.load('output/tvcnn2_classifer/accs.npy',allow_pickle=True).item()

    # Plot Classification Accuracy
    plt.plot(snrs, [accs[x] for x in snrs])
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
    plt.savefig('output/tvcnn2_classifer/CNN2 Classification Accuracy.png')
    plt.show()

