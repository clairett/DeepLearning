from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt


def binarize_data(input):
    threshold, upper, lower = 0.5, 1, 0
    input = np.where(input >= threshold, upper, lower)
    return input


def load_data(file):
    data = np.loadtxt(file, dtype='float', delimiter=',')
    X = data[:, :-1]
    X = binarize_data(X)  # 3000*784
    return X


def sigmoid(X):
    return 1 / (1+np.exp(-X))


def init_w(insize, outsize):
    a = math.sqrt(6.0 / (insize + outsize))
    return np.random.uniform(-a, a, size=(insize, outsize))


def init_b(size):
    return np.zeros((size, 1))


def gibbs_sampling(v, model, gs_steps):
    W, vbias, hbias = model['W'], model['vbias'], model['hbias']
    h = None
    for i in xrange(gs_steps):
        h = sigmoid(W.T.dot(v) + hbias)
        h = np.random.binomial(1, h)
        v = sigmoid(W.dot(h) + vbias)
        v = np.random.binomial(1, v)
    return [h, v]


def get_cross_entropy(X, model):
    W, vbias = model['W'], model['vbias']
    h, _ = gibbs_sampling(X, model, 1)
    v = sigmoid(W.dot(h) + vbias)
    cross_entropy = -np.mean(np.sum(X * np.log(v) + (1 - X) * np.log(1 - v), axis=0))
    return cross_entropy


def train_RBM(trainset=None, devset=None, seed=2, num_hidden=100, epochs=100, learning_rate=0.01,
              batch_size=10, K=100, gs_step=1):
    np.random.seed(seed)
    X_train = load_data(trainset)
    X_dev = load_data(devset)

    # transpose input
    X_train = X_train.T  # 784 * 3000
    X_dev = X_dev.T

    insize = X_train.shape[0]

    model = {}
    W = init_w(insize, num_hidden)       # 784 * 100
    vbias = init_b(insize).reshape(-1, 1)
    hbias = init_b(num_hidden)

    model['W'], model['vbias'], model['hbias'] = W, vbias, hbias

    # persistent chains
    v = np.random.binomial(1, 0.5, (insize, K))
    
    train_error, dev_error = [], []

    for epoch in xrange(epochs):
        train_cross_entropy = get_cross_entropy(X_train, model)
        dev_cross_entropy = get_cross_entropy(X_dev, model)
        print 'Epoch %d: train %f  dev %f' % (epoch, train_cross_entropy, dev_cross_entropy)
        train_error.append(train_cross_entropy)
        dev_error.append(dev_cross_entropy)

        for i in xrange(int(X_train.shape[1]/batch_size)):
            rows = np.random.permutation(X_train.shape[1])[:batch_size]
            X_batch = X_train[:, rows]  # mini-batch

            _, v = gibbs_sampling(v, model, gs_step)
            
            # CD-k
            # _, v = gibbs_sampling(X_batch, model, gs_step)

            h_X_batch = sigmoid(np.dot(W.T, X_batch) + hbias)
            h_v = sigmoid(hbias + np.dot(W.T, v) + hbias)

            # update model
            W += learning_rate * (X_batch.dot(h_X_batch.T)/X_batch.shape[1] - v.dot(h_v.T)/v.shape[1])
            vbias += learning_rate * (np.sum(X_batch, axis=1, keepdims=True)/X_batch.shape[1] - np.sum(v, axis=1, keepdims=True)/v.shape[1])
            hbias += learning_rate * (np.sum(h_X_batch, axis=1, keepdims=True)/h_X_batch.shape[1] - np.sum(h_v, axis=1, keepdims=True)/h_v.shape[1])

    # draw cross entropy of train and dev
    draw_error_fig(epochs, train_error, dev_error)
    draw_weights(W)
    draw_samples(insize, K, model)


def draw_error_fig(epochs, train_error, dev_error):
    fig, ax = plt.subplots()
    epochs = [i for i in xrange(epochs)]

    ax.plot(epochs, train_error, 'g', label='Train Cross Entropy')
    ax.plot(epochs, dev_error, 'b', label='Validation Cross Entropy')
    ax.set_xlabel("number of epoches")
    ax.set_ylabel("avg cross-entropy error")

    legend = ax.legend(loc='upper right', shadow=False)

    for label in legend.get_texts():
        label.set_fontsize('small')
    plt.show()


def draw_weights(weight):
    W = np.transpose(weight)
    W = np.reshape(W, (-1,28,28))

    fig = plt.figure()
    for i in xrange(W.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.axis('off')
        plt.imshow(W[i],cmap=plt.cm.binary)
    plt.show()


def draw_samples(insize, K, model):
    v = np.random.binomial(1, 0.5, (insize, K))
    h, v = gibbs_sampling(v, model, 1000)
    draw_weights(v)


if __name__ == "__main__":
    train_file = "digitstrain.txt"
    val_file = "digitsvalid.txt"
    test_file = "digitstest.txt"
    train_RBM(trainset=train_file, devset=val_file, batch_size=10, num_hidden=100)

