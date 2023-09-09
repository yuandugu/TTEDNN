import h5py
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_para(N, adj_mode):
    """
    load electrical parameters from h5
    """
    f = h5py.File('data/parameter/parameter%s.h5' % (N), 'r')
    PY = f['PY'][()]
    initial = f['initial'][()]
    f.close()

    P = PY[0, :]
    Y = PY[1:, :]
    YY = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            if Y[i, j] != 0.:
                YY[i, j] = 1

    if adj_mode == 1:
        Y = YY + np.eye(N)
    elif adj_mode == 2:
        Y = Y + np.diag(abs(P))
        Y = Y / np.amax(Y)
    else:
        pass

    print('Electrical parameters loaded.')
    return Y, PY


def classify_random_5(N, x_theta, x_omega, y_data, test_size, chosedlength):
    """
    randomly classify data to train, val and test group
    """
    x_data = np.dstack((x_omega,x_theta))
    x_train, X_test, y_train, Y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0)
    del x_data, y_data, x_train, y_train
    X_train = np.reshape(X_train, (X_train.shape[0], N, chosedlength, 2))
    X_val = np.reshape(X_val, (X_val.shape[0], N, chosedlength, 2))
    X_test = np.reshape(X_test, (X_test.shape[0], N, chosedlength, 2))

    return X_train[:, :, :, 0], X_train[:, :, :, 1], X_val[:, :, :, 0], X_val[:, :, :, 1], X_test[:, :, :, 0], X_test[:, :, :, 1], Y_train, Y_val, Y_test


def normalization(N, x_theta, x_omega):

    scaler = StandardScaler()
    x_omega_norm = scaler.fit_transform(x_omega.astype(np.float32))
    x_theta_norm = scaler.fit_transform(x_theta.astype(np.float32))

    return x_omega_norm, x_theta_norm


def standardization(N, chosedlength, x_theta, x_omega, relative, mode):

    if relative:

        if mode == 0:

            omega_range = np.max(x_omega, axis=1) - np.min(x_omega, axis=1)
            theta_range = np.max(x_theta, axis=1) - np.min(x_theta, axis=1)

            return (x_omega - np.min(x_omega, axis=1).reshape(-1, 1)) / omega_range.reshape(-1, 1), (x_theta - np.min(x_theta, axis=1).reshape(-1, 1)) / theta_range.reshape(-1, 1)

        elif mode == 1:

            return x_omega / np.max(abs(x_omega), axis=1).reshape(-1, 1), x_theta / np.max(abs(x_theta), axis=1).reshape(-1, 1)

    else:
        x_omega_std = np.zeros((x_theta.shape[0], N * chosedlength))
        x_theta_std = np.zeros((x_theta.shape[0], N * chosedlength))
        if mode == 0:
            for i in range(N):
                omega_range = np.max(x_omega[:, i * chosedlength:(i + 1) * chosedlength], axis=1) - np.min(x_omega[:, i * chosedlength:(i+1)*chosedlength], axis=1)
                x_omega_std[:, i*chosedlength:(i + 1) * chosedlength] = (x_omega[:, i * chosedlength:(i + 1) * chosedlength] - np.min(x_omega[:,i*chosedlength:(i+1)*chosedlength], axis=0)) / omega_range

                theta_range = np.max(x_theta[:, i * chosedlength:(i+1)*chosedlength], axis=1) - np.min(x_theta[:, i * chosedlength:(i + 1) * chosedlength], axis=1)
                x_theta_std[:, i * chosedlength:(i + 1)*chosedlength] = (x_theta[:,i*chosedlength:(i + 1) * chosedlength] - np.min(x_theta[:, i * chosedlength:(i+1)*chosedlength], axis=0)) / theta_range

        elif mode == 1:

            for i in range(N):
                omega_max = np.max(abs(x_omega[:,i*chosedlength:(i+1)*chosedlength]), axis=1)
                x_omega_std[:, i*chosedlength:(i+1)*chosedlength] = x_omega[:,i*chosedlength:(i+1)*chosedlength] / omega_max.reshape(-1, 1)

                theta_max = np.max(abs(x_theta[:,i*chosedlength:(i+1)*chosedlength]), axis=1)
                x_theta_std[:, i*chosedlength:(i+1)*chosedlength] = x_theta[:,i*chosedlength:(i+1)*chosedlength] / theta_max.reshape(-1, 1)

        return x_omega_std, x_theta_std


def smooth(a, WSZ):
    """
    moving average
    """
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


def load_data_one_IEEE_l(N, length, timelength, chosedlength, TEST_SIZE, relative, normalize, standard, mode, move, WSZ):

    start = time.perf_counter()
    X_one_theta_2 = np.zeros((length, chosedlength * N))
    X_one_omega_2 = np.zeros((length, chosedlength * N))

    f = h5py.File('data/single/1.h5', 'r')

    if chosedlength != timelength:
        for i in range(N):
            X_one_theta_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_theta'][()][:, i*timelength:i*timelength+chosedlength]
            X_one_omega_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_omega'][()][:, i*timelength:i*timelength+chosedlength]
    else:
        X_one_theta_2 = f['data_theta'][()]
        X_one_omega_2 = f['data_omega'][()]

    Y_one_2 = f['Y'][()]
    f.close()
    del f

    for i in range(N):
        if i == 0:
            pass
        else:
            X_theta = np.zeros((length, chosedlength*N))
            X_omega = np.zeros((length, chosedlength*N))
            f = h5py.File('data/single/%s.h5' % (i+1), 'r')
            if chosedlength != timelength:
                for ii in range(N):
                    X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                    X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_omega'][()][:, ii*timelength:ii*timelength+chosedlength]
            else:
                X_theta = f['data_theta'][()]
                X_omega = f['data_omega'][()]
            Y = f['Y'][()]
            f.close()
            del f
            X_one_theta_2 = np.vstack((X_one_theta_2, X_theta))
            X_one_omega_2 = np.vstack((X_one_omega_2, X_omega))
            Y_one_2 = np.hstack((Y_one_2, Y))
            del X_theta, X_omega, Y

    end = time.perf_counter()
    print('所用时间为%ss' % (str(end - start)))

    X_one_theta_2 = np.float32(X_one_theta_2)
    X_one_omega_2 = np.float32(X_one_omega_2)

    if normalize:
        X_one_omega_norm, X_one_theta_norm = normalization(
            x_theta=X_one_theta_2, x_omega=X_one_omega_2
        )
        del X_one_theta_2, X_one_omega_2
        X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test = classify_random_5(
            x_theta=X_one_theta_norm,
            x_omega=X_one_omega_norm,
            y_data=Y_one_2,
            test_size=TEST_SIZE,
            chosedlength=chosedlength,
            N=N
        )

    elif standard:
        X_one_omega_std, X_one_theta_std = standardization(
            N=N, chosedlength=chosedlength, x_theta=X_one_theta_2, x_omega=X_one_omega_2, relative=relative, mode=mode
        )
        del X_one_theta_2, X_one_omega_2
        X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test = classify_random_5(
            x_theta=X_one_theta_std,
            x_omega=X_one_omega_std,
            y_data=Y_one_2,
            test_size=TEST_SIZE,
            chosedlength=chosedlength,
            N=N
        )

    else:
        X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test = classify_random_5(
            x_theta=X_one_theta_2,
            x_omega=X_one_omega_2,
            y_data=Y_one_2,
            test_size=TEST_SIZE,
            chosedlength=chosedlength,
            N=N
        )
        del X_one_theta_2, X_one_omega_2

    if move:
        for i in range(N):
            for j in range(X_train.shape[0]):
                X_train[j, :, i] = smooth(a=X_train[j, :, i], WSZ=WSZ)
            for j in range(X_val.shape[0]):
                X_val[j, :, i] = smooth(a=X_val[j, :, i], WSZ=WSZ)
            for j in range(X_test.shape[0]):
                X_test[j, :, i] = smooth(a=X_test[j, :, i], WSZ=WSZ)
        del i, j

    a = len(Y_train)+len(Y_val)+len(Y_test)
    print('Total number:%s' % (a))
    b = int(np.sum(Y_train)+np.sum(Y_val) + np.sum(Y_test))

    print('syn:non-syn=%s:%s' % (a-b, b))

    print('Training dataset：', X_train.shape)
    print('syn:non-syn=%s:%s' % (len(Y_train)-int(np.sum(Y_train)), int(np.sum(Y_train))))
    print('Validation dataset：', X_val.shape)
    print('syn:non-syn=%s:%s' % (len(Y_val)-int(np.sum(Y_val)), int(np.sum(Y_val))))
    print('Testing dataset：', X_test.shape)
    print('sys:bob-syn%s:%s' % (len(Y_test)-int(np.sum(Y_test)), int(np.sum(Y_test))))

    print('Dataset done, prepare to train.')

    return X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test, a, b


if __name__ == '__main__':

    Y, PY = load_para(N=14, adj_mode=1)

    X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test, a, b = load_data_one_IEEE_l(
        N=14,
        length=4000,
        timelength=100,
        chosedlength=10,
        TEST_SIZE=0.2,
        CHANNEL=1,
        relative=False,
        normalize=False,
        standard=False,
        mode=1,
        move=False,
        WSZ=11
    )

    print(X_train.shape)
