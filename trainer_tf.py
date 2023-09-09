import numpy as np
import math
import time
import h5py
import tensorflow as tf
from nn import ttednnn_tf
from spektral.layers import GraphConv
from spektral.utils import batch_iterator
from tensorflow.keras.metrics import BinaryAccuracy, BinaryCrossentropy, TruePositives
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, auc, roc_curve
from utils.load_data import load_data_one_IEEE_l, load_para
import os
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

n_critical = 100           # thresholds to change loss function
weight = False
n = 0                      # current epochs

if interval:
    timelength = 51
else:
    timelength = 101

net = 'TTEDNN'
data_set = 'one'
adj_mode = 2                # adjacency matrix mode: 1、adj=Y
                            #                        2、adj=diag(P)+Y
                            #                        3、adj=P'+Y',P'=P·(1+ω_0/ω_s),Y'=Y_ij·sin(θ_i-θ_j)
chosedlength = 10           # length before cut out
data_number = 4000          # same in data/IEEE.py
TEST_SIZE = 0.2             # train:val_test = 6:2:2

F = chosedlength
n_out = 1
l2_reg_gcn = 5e-4
learning_rate = 1e-3        # Learning rate for Adam
BATCH_SIZE = 256            # Batch size
epochs = 500                # Number of training epochs
patience = 100              # Patience for early stopping

print('choslength :%s \n interval  : %s \n normalize : %s \n standard  : %s \n relative  : %s \n mode      : %s \n move      : %s'
    % (chosedlength, interval, normalize, standard, relative, mode, move)
)

#####################  load data & processing  ####################


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
                axis=1)
        ) * np.repeat(
            a=np.expand_dims(PY[1:, :], axis=0),
            repeats=x_theta.shape[0],
            axis=0
        ))
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

Y_train = Y_train.reshape(-1, 1).astype('float32')
Y_val = Y_val.reshape(-1, 1).astype('float32')
Y_test = Y_test.reshape(-1, 1).astype('float32')

# process adj matrix, create filter for GCN and convert to sparse tensor
if adj_mode != 3:
    # (length, N, N)
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

model = ttednnn_tf(l2_reg_gcn=l2_reg_gcn, filters_tcn=32, n_out=n_out)
shape_1 = tf.TensorSpec(shape=(BATCH_SIZE, N, F), dtype=tf.dtypes.float32, name=None)
shape_2 = tf.TensorSpec(shape=(BATCH_SIZE, N, N), dtype=tf.dtypes.float32, name=None)
model._set_inputs(shape_1, shape_2)  # 设置模型的输入形状

optimizer = Adam(lr=learning_rate)

loss_object = tf.keras.losses.BinaryCrossentropy()  # for training
train_loss_fn = BinaryCrossentropy()                  # for model evaluate
train_acc_fn = BinaryAccuracy()
val_loss_fn = BinaryCrossentropy()
val_acc_fn = BinaryAccuracy()
test_loss_fn = BinaryCrossentropy()
test_acc_fn = BinaryAccuracy()
acc_fn_1 = TruePositives()

#####################  Functions  ####################


# Training step
def train_weight(x, fltr, y):
    """
    for class_balanced_loss
    """
    if weight:
        a = sum(y)
        b = len(y)
        if a > 0:
            pos_weight = y * b / a + np.ones_like(y) - y
        else:
            pos_weight = np.ones_like(y)
        with tf.GradientTape() as tape:
            predictions = model([x, fltr], training=True)
            loss = loss_object(y, predictions, sample_weight=pos_weight.tolist())
            loss += sum(model.losses)
    else:
        with tf.GradientTape() as tape:
            predictions = model([x, fltr], training=True)
            loss = loss_object(y, predictions)
            loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss_fn(y, predictions)
    train_acc_fn(y, predictions)


# Evaluation step
def evaluate_weight(x, fltr, y):
    if weight:
        a = sum(y)
        b = len(y)
        if a > 0:
            pos_weight = y * b / a + np.ones_like(y) - y
        else:
            pos_weight = np.ones_like(y)
        predictions = model([x, fltr], training=False)
        val_loss = val_loss_fn(y, predictions, sample_weight=pos_weight.tolist())
    else:
        predictions = model([x, fltr], training=False)
        val_loss = val_loss_fn(y, predictions)
    val_loss += sum(model.losses)
    val_acc = val_acc_fn(y, predictions)
    val_loss_fn.reset_states()
    val_acc_fn.reset_states()
    return val_loss, val_acc


# Testing step
@tf.function
def test_weight(x, fltr, y):
    predictions = model([x, fltr], training=False)
    te_loss = test_loss_fn(y, predictions)
    te_loss += sum(model.losses)
    te_acc = test_acc_fn(y, predictions)
    test_loss_fn.reset_states()
    test_acc_fn.reset_states()
    return te_loss, te_acc, predictions


# Setup training
best_val_loss = 99999
current_patience = patience
curent_batch = 0
batches_in_epoch = int(np.ceil(X_train.shape[0] / BATCH_SIZE))
batches_tr = batch_iterator([X_train, adj_train, Y_train], batch_size=BATCH_SIZE, epochs=epochs)

# Training loop
loss_train = []
acc_train = []
loss_val = []
acc_val = []
loss_test = []
acc_test = []
n = 0
print('\nTraining ------------')
start = time.perf_counter()
if n_critical == 0:
    weight = True
    print('Loss function=Weight BCE')
else:
    print('Loss function=BCE')
    pass
loss_1, acc_1 = evaluate_weight(x=X_train, fltr=adj_train, y=Y_train)
loss_2, acc_2 = evaluate_weight(x=X_val, fltr=adj_val, y=Y_val)
loss_3, acc_3 = evaluate_weight(x=X_test, fltr=adj_test, y=Y_test)
print(
    'Epochs: {:.0f} | ' 'Train loss: {:.4f}, acc: {:.4f} | ' 'Valid loss: {:.4f}, acc: {:.4f} | ' 'Test loss: {:.4f}, acc: {:.4f}'
    .format(n, loss_1, acc_1, loss_2, acc_2, loss_3, acc_3)
)

for batch in batches_tr:

    if n == n_critical and curent_batch == 0:
        weight = True
        print('Loss function=Weight BCE')
    else:
        pass
    curent_batch += 1
    train_weight(*batch)

    if curent_batch == batches_in_epoch:
        n = n + 1
        loss_va, acc_va = evaluate_weight(x=X_val, fltr=adj_val, y=Y_val)

        if loss_va < best_val_loss:
            best_val_loss = loss_va
            current_patience = patience
            loss_te, acc_te, _ = test_weight(x=X_test, fltr=adj_test, y=Y_test)
        else:
            current_patience -= 1
            if current_patience == 0:
                print('Early stopping')
                break

        # Print results
        print(
            'Epochs: {:.0f} | ' 'Train loss: {:.4f}, acc: {:.4f} | ' 'Valid loss: {:.4f}, acc: {:.4f} | ' 'Test loss: {:.4f}, acc: {:.4f}'
            .format(n, train_loss_fn.result(), train_acc_fn.result(), loss_va, acc_va, loss_te, acc_te)
        )
        loss_train.append(train_loss_fn.result().numpy())
        acc_train.append(train_acc_fn.result().numpy())
        loss_val.append(loss_va)
        acc_val.append(acc_va)
        loss_test.append(loss_te)
        acc_test.append(acc_te)
        # Reset epoch
        train_loss_fn.reset_states()
        train_acc_fn.reset_states()
        curent_batch = 0

end = time.perf_counter()
print('training duration:%ss' % (end-start))
del X_train, X_val, Y_train, Y_val

EPOCHS = loss_train.shape[0]
HISTORY = np.zeros((6, EPOCHS))
HISTORY[0, :] = np.array(acc_train)
HISTORY[1, :] = np.array(acc_val)
HISTORY[2, :] = np.array(loss_train)
HISTORY[3, :] = np.array(loss_val)
HISTORY[4, :] = np.array(acc_test)
HISTORY[5, :] = np.array(loss_test)

#####################  Testing  ####################

print('\nTesting ------------')
weight = False
loss, accuracy, Y_predict = test_weight(x=X_test, fltr=adj_test, y=Y_test)
acc_tpr = acc_fn_1(Y_test, Y_predict)
acc_fn_1.reset_states()

print('model test loss: ', loss)
print('model test accuracy: ', accuracy)
print('model test TPR: ', acc_tpr)

Y_predict_int = np.rint(Y_predict)  # output
con_mat = confusion_matrix(Y_test, Y_predict_int)  # confusion matrix
print(con_mat)
con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
con_mat_norm = np.around(con_mat_norm, decimals=2)
fpr, tpr, thresholds_keras = roc_curve(Y_test.astype(int), Y_predict)  # AUC   
auc = auc(fpr, tpr)
print("AUC : ", auc)

"""
save
"""
if not os.path.exists('./result/%s/histroy.h5' % (exp_num)):
    os.makedirs('./result/%s/histroy.h5' % (exp_num))
model.save_weights('./result/%s/model.h5' % (exp_num))

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
