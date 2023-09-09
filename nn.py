from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from spektral.layers import GraphConv
from model.TCN import TempConvNet


class ttednnn_tf(Model):

    def __init__(self, l2_reg_gcn, filters_tcn, n_out, **kwargs):

        self.filters_tcn = filters_tcn
        self.n_out = n_out
        self.l2_reg_gcn = l2_reg_gcn
        super().__init__(**kwargs)
        self.conv1 = GraphConv(
            16, kernel_regularizer=l2(l=self.l2_reg_gcn), activation='relu'
        )
        self.conv2 = GraphConv(
            16, kernel_regularizer=l2(l=self.l2_reg_gcn), activation='relu'
        )
        self.flatten = Flatten()
        self.fc1 = Dense(64, activation='relu')
        self.reshape = Reshape(target_shape=(64, 1))
        self.tcn = TempConvNet(
            nb_filters=self.filters_tcn, dilations=(1, 2, 4, 8, 16), dropout_rate=0.0, use_layer_norm=True
        )
        self.fc2 = Dense(32, activation='relu')
        if self.n_out == 1:
            self.fc3 = Dense(self.n_out, activation='sigmoid')
        elif self.n_out > 1:
            self.fc3 = Dense(self.n_out, activation='softmax')

    def call(self, inputs):
        x, fltr = inputs
        x = self.conv1([x, fltr])
        x = self.conv2([x, fltr])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.reshape(output)
        output = self.tcn(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


def ttednn_keras(N, F, n_out, l2_reg_gcn, filters_tcn, learning_rate):

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr

    opt = Adam(learning_rate=learning_rate)
    lr_metric = get_lr_metric(opt)
    X_in = Input(shape=(N, F))  # Ω_s
    A_in = Input((N, N))  # adjacency matrix
    gcn_1 = GraphConv(
        16, kernel_regularizer=l2(l=l2_reg_gcn), activation='relu'
    )([X_in, A_in])
    gcn_2 = GraphConv(
        16, kernel_regularizer=l2(l=l2_reg_gcn), activation='relu'
    )([gcn_1, A_in])
    flatten = Flatten()(gcn_2)
    fc_1 = Dense(64, activation='relu')(flatten)
    fc_1_1 = Reshape(target_shape=(64, 1))(fc_1)
    tcn = TempConvNet(
        nb_filters=filters_tcn, dilations=(1, 2, 4, 8, 16),
        dropout_rate=0.0, use_layer_norm=True
    )(fc_1_1)
    fc_2 = Dense(32, activation='relu')(tcn)
    if n_out == 1:
        fc_3 = Dense(n_out, activation='sigmoid')(fc_2)
    elif n_out > 1:
        fc_3 = Dense(n_out, activation='softmax')(fc_2)
    model = Model(inputs=[X_in, A_in], outputs=fc_3)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        weighted_metrics=['accuracy', lr_metric]
    )
    model.summary()  # show network structure
    return model


if __name__ == '__main__':

    # test
    N = 39
    F = 101
    l2_reg_gcn = 5e-4
    n_out = 1
    learning_rate = 1e-3
    model = ttednnn_tf(l2_reg_gcn=l2_reg_gcn, filters_tcn=32, n_out=n_out)
    model.build(input_shape=[(N, F), (N, N)])
    optimizer = Adam(lr=learning_rate)
    loss_fn = BinaryCrossentropy()
    acc_fn = BinaryAccuracy()
    model.summary()
    # tf.keras.utils.plot_model(model, to_file='model.png')
