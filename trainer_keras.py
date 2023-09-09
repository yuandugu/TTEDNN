import numpy as np
import math
import time
import h5py
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, roc_curve, auc
from spektral.layers import GraphConv
from utils.load_data import load_para, load_data_one_IEEE_l
from nn import ttednn_keras
np.random.seed(1337)
SEED = 20000
tf.random.set_seed(seed=SEED)

#####################  set parameters  ####################

N = 14                     # number of node
omega_s = 100 * math.pi
theta = math.pi            # range of theta_0
omega = 20                 # range of omega_0
exp_num = 1
early_stop = False
interval = True
relative = False
normalize = False
standard = False
mode = 1
move = False
WSZ = 11

if interval:
    timelength = 100
else:
    timelength = 400

net = 'TTEDNN'
data_set = 'one'
adj_mode = 2               # adjacency matrix mode: 1、adj=Y
                           #                        2、adj=diag(P)+Y
                           #                        3、adj=P'+Y',P'=P·(1+ω_0/ω_s),Y'=Y_ij·sin(θ_i-θ_j)
chosedlength = 10          # length before cut out
data_number = 4000         # same in data/IEEE.py
TEST_SIZE = 0.2            # train:val_test = 6:2:2

F = chosedlength
n_out = 1
l2_reg_gcn = 5e-4
learning_rate = 1e-3       # Learning rate for Adam
BATCH_SIZE = 256           # Batch size
epochs = 500               # Number of training epochs
patience = 100             # Patience for early stopping

print('chosdlength:%s \n interval  : %s \n normalize : %s \n standard  : %s \n relative  : %s \n mode      : %s \n move      : %s'
    % (chosedlength, interval, normalize, standard, relative, mode, move)
)

#####################  load data & processing  ####################


def draw_training_curve(history):
    """
    save training curve
    """
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    lr_value = history_dict['lr']
    val_lr_value = history_dict['val_lr']
    HISTORY = np.zeros((6, len(loss_values)))
    HISTORY[0, :] = np.array(acc_values)
    HISTORY[1, :] = np.array(val_acc_values)
    HISTORY[2, :] = np.array(loss_values)
    HISTORY[3, :] = np.array(val_loss_values)
    HISTORY[4, :] = np.array(lr_value)
    HISTORY[5, :] = np.array(val_lr_value)

    return HISTORY


def gen_adj(x_omega, x_theta, PY):

    init_theta = x_theta[:, :, 0]
    Y = np.abs(np.sin(
            np.repeat(
                a=np.expand_dims(init_theta, axis=2),
                repeats=N,
                axis=2
            ) -
            np.repeat(
                a=np.expand_dims(init_theta, axis=1),
                repeats=N,
                axis=1
            )
        ) * np.repeat(
            a=np.expand_dims(PY[1:, :], axis=0),
            repeats=x_theta.shape[0],
            axis=0
        )
    )
    init_omega = x_omega[:, :, 0]
    P = np.abs(np.repeat(
            a=np.expand_dims(
                a=PY[0, :],
                axis=0
            ),
            repeats=x_omega.shape[0],
            axis=0
        )
    ) * (1 + 1 / omega_s * init_omega)
    P = np.array([np.diag(P[i, :]) for i in range(P.shape[0])])

    adj = np.array(P + Y)

    return adj


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

#####################  Network setup  ####################

model = ttednn_keras(
    N=N, F=F, n_out=n_out, l2_reg_gcn=l2_reg_gcn, filters_tcn=32, learning_rate=learning_rate
)

# Prepare data
validation_data = ([X_val, adj_val], Y_val)
del X_val, adj_val, Y_val
# Train model
print('Training ------------')

a = sum(Y_train)
b = len(Y_train)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=patience, mode='auto')
start = time.perf_counter()
history = model.fit(
    [X_train, adj_train],
    Y_train,
    epochs=epochs,
    validation_data=validation_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_weight={0: 1., 1: b/a-1},
    callbacks=[reduce_lr]
)
end = time.perf_counter()
print('training duration:%ss' % (end-start))
del X_train, Y_train

HISTORY = draw_training_curve(history=history)

###############################  test  ####################################

print('\nTesting ------------')
loss, accuracy, learning_rate = model.evaluate([X_test, adj_test], Y_test)
print('model test loss: ', loss)
print('model test accuracy: ', accuracy)
print('model test lr: ', learning_rate)

Y_predict = model.predict([X_test, adj_test])
Y_predict_int = np.rint(Y_predict)
con_mat = confusion_matrix(Y_test, Y_predict_int)
print(con_mat)
con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
con_mat_norm = np.around(con_mat_norm, decimals=2)
fpr, tpr, thresholds_keras = roc_curve(Y_test.astype(int), Y_predict)
auc = auc(fpr, tpr)
print("AUC : ", auc)

"""
save
"""
if not os.path.exists('./result/%s/histroy.h5' % (exp_num)):
    os.makedirs('./result/%s/histroy.h5' % (exp_num))
model.save('./result/%s/model.h5' % (exp_num))

f = h5py.File('./result/%s/histroy.h5' % (exp_num), 'w')
f.create_dataset('train_history', data=HISTORY)
f.create_dataset('test_loss', data=loss)
f.create_dataset('test_accuracy', data=accuracy)
f.create_dataset('test_matrix', data=con_mat)
f.create_dataset('test_fpr', data=fpr)
f.create_dataset('test_tpr', data=tpr)
f.create_dataset('test_AUC', data=auc)
f.create_dataset('pre', data=Y_predict)
f.close()
del f
