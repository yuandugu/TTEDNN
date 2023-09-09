from matplotlib import rcParams
from matplotlib import pyplot as plt
import numpy as np
import h5py
from sklearn.manifold import TSNE
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from utils.load_data import load_para
from trainer_keras import gen_adj
from spektral import GraphConv
from model.TCN import TempConvNet
from utils.load_data import load_data_one_IEEE_l


def draw_training_curve(exp_num):

    f = h5py.File('./result/%s/histroy.h5' % (exp_num), 'r')
    HISTORY = f['train_history'][()]
    f.close()
    del f

    acc_values = HISTORY[0, :]
    loss_values = HISTORY[2, :]
    config = {
        "font.family": 'Times New Roman',
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    fig = plt.figure(figsize=(4, 4), dpi=300)
    plt.rc('axes', lw=0.5)
    ax1 = fig.subplots()
    left = 0.15
    bottom = 0.15
    plt.subplots_adjust(
        left=left, bottom=bottom, right=1-2*bottom+left, top=1-bottom
    )
    ax2 = ax1.twinx()
    epochs = range(1, len(HISTORY[0, :]) + 1)
    ax1.plot(
        epochs, acc_values,     'r',
        linewidth=.5,
        label='ACC'
    )
    ax1.plot(
        epochs, loss_values,    'b',
        linewidth=.5,
        label='Loss'
    )
    ax1.set_ylabel('Training ACC')
    ax2.set_ylabel('Training Loss')

    ax1.set_ylim([0, 1])
    ax2.set_ylim([0, 1])
    ax1.tick_params(direction='in', width=0.5, length=2)
    ax2.tick_params(direction='in', width=0.5, length=2)
    ax1.set_xlabel('Epochs')
    legend = ax1.legend(fancybox=False, facecolor='none', edgecolor='k')
    plt.savefig('training.png')
    plt.show()


def get_layer_output(model, layer_index, inputs):

    layer = Model(inputs=model.input,
                  outputs=model.get_layer(index=layer_index).output)
    layer_out = layer.predict(inputs)
    return np.array(layer_out)


def get_middle_output(exp_num, N, data_number, timelength, TEST_SIZE, relative, normalize, standard, mode, move, WSZ, adj_mode, chosedlength, learning_rate):

    A, PY = load_para(
        N=N, adj_mode=adj_mode
    )

    X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test, a, b = load_data_one_IEEE_l(
        N=N,
        length=data_number,
        timelength=timelength,
        chosedlength=chosedlength,
        TEST_SIZE=TEST_SIZE,
        relative=relative,
        normalize=normalize,
        standard=standard,
        mode=mode,
        move=move,
        WSZ=WSZ
    )

    F = X_train.shape[-1]
    n_out = 1

    # process adj matrix, create filter for GCN and convert to sparse tensor
    if adj_mode != 3:
        # (N, N)
        adj = GraphConv.preprocess(A=A)
        del X_train_theta, X_val_theta, X_test_theta
        adj_train = np.repeat(
            a=np.expand_dims(adj, axis=0),
            repeats=X_train.shape[0],
            axis=0
        )
        adj_val = np.repeat(
            a=np.expand_dims(adj, axis=0),
            repeats=X_val.shape[0],
            axis=0
        )
        adj_test = np.repeat(
            a=np.expand_dims(adj, axis=0),
            repeats=X_test.shape[0],
            axis=0
        )
    else:
        # (length, N, N)
        adj_train = gen_adj(x_omega=X_train, x_theta=X_train_theta, PY=PY)
        adj_val = gen_adj(x_omega=X_val, x_theta=X_val_theta, PY=PY)
        adj_test = gen_adj(x_omega=X_test, x_theta=X_test_theta, PY=PY)

        adj_train = np.array([GraphConv.preprocess(adj_train[i, :, :]) for i in range(Y_train.shape[0])])
        adj_val = np.array([GraphConv.preprocess(adj_val[i, :, :]) for i in range(Y_val.shape[0])])
        adj_test = np.array([GraphConv.preprocess(adj_test[i, :, :]) for i in range(Y_test.shape[0])])

    path = './result/' + str(exp_num) + '/'

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr

    opt = Adam(learning_rate=learning_rate)
    lr_metric = get_lr_metric(opt)
    model = load_model(
        path + '_model.h5',
        custom_objects={'GraphConv': GraphConv, 'TempConvNet': TempConvNet, 'lr': lr_metric}
    )

    output_3_train = get_layer_output(
        model=model, layer_index=3, inputs=[X_train, adj_train]
    )  # second GCN output
    output_3_val = get_layer_output(
        model=model, layer_index=3, inputs=[X_val, adj_val]
    )  # second GCN output
    output_3_test = get_layer_output(
        model=model, layer_index=3, inputs=[X_test, adj_test]
    )  # second GCN output

    Y_predict_train = model.predict([X_train, adj_train])
    Y_predict_val = model.predict([X_val, adj_val])
    Y_predict_test = model.predict([X_test, adj_test])

    f = h5py.File(path + '_middle_output.h5', 'w')
    f.create_dataset('/true/train', data=Y_train)
    f.create_dataset('/true/val',   data=Y_val)
    f.create_dataset('/true/test',  data=Y_test)
    f.create_dataset('/output/train', data=Y_predict_train)
    f.create_dataset('/output/val',   data=Y_predict_val)
    f.create_dataset('/output/test',  data=Y_predict_test)
    f.create_dataset('/middle/3/train', data=output_3_train)
    f.create_dataset('/middle/3/val',   data=output_3_val)
    f.create_dataset('/middle/3/test',  data=output_3_test)
    f.close()
    del f


def hidden_layer(exp_num):

    f = h5py.File('./result/%s/middle_output.h5' % (exp_num), 'r')
    output_3_train = f['/middle/3/train'][()]
    Y_train = f['/true/train'][()]
    Y_predict_train = f['/output/train'][()]
    Y_predict_train_int = np.rint(Y_predict_train)
    f.close()
    del f

    color_train = []
    for i in Y_train:
        if i == 0:
            color_train.append('g')
        else:
            color_train.append('k')
    color_train_pre = []
    for i in Y_predict_train_int:
        if i == 0:
            color_train_pre.append('g')
        else:
            color_train_pre.append('k')
    color_train_error = []
    for i in range(Y_train.shape[0]):
        if Y_train[i] != Y_predict_train_int[i]:
            color_train_error.append('r')
        else:
            color_train_error.append('w')

    # true-pre-error
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne = tsne.fit_transform(output_3_train)
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 3, 1)
    plt.xlim(-80, 80)
    plt.ylim(-80, 80)
    plt.scatter(tsne[:, 0], tsne[:, 1], s=1, c=color_train)
    plt.subplot(1, 3, 2)
    plt.xlim(-80, 80)
    plt.ylim(-80, 80)
    plt.scatter(tsne[:, 0], tsne[:, 1], s=1, c=color_train_pre)
    plt.subplot(1, 3, 3)
    plt.xlim(-80, 80)
    plt.ylim(-80, 80)
    plt.scatter(tsne[:, 0], tsne[:, 1], s=1, cmap='RdYlBu', c=Y_predict_train)
    plt.savefig('hidden_layer.png')
    plt.show()


# training process visualization
draw_training_curve(exp_num=1)
# hidden layer activations visualization
get_middle_output(exp_num=1, N=14, data_number=1000, timelength=101, chosedlength=11, TEST_SIZE=0.2, relative=False, normalize=False, standard=False, mode=1, move=False, WSZ=5, adj_mode=2)
hidden_layer(exp_num=1)
