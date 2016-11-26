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


def mean_field(X, model, mf_steps):
    W1, W2 = model['W1'], model['W2']
    hbias1, hbias2 = model['hbias1'], model['hbias2']
    mu1 = np.random.rand(W2.shape[0], X.shape[1])
    mu2 = np.random.rand(W2.shape[1], X.shape[1])
    for i in xrange(mf_steps):
        mu1 = sigmoid(np.dot(W1.T, X) + np.dot(W2, mu2) + hbias1)
        mu2 = sigmoid(np.dot(W2.T, mu1) + hbias2)
    return mu1, mu2


def gibbs_sampling(h1, h2, v, model, gs_steps):
    W1, W2 = model['W1'], model['W2']
    vbias, hbias1, hbias2 = model['vbias'], model['hbias1'], model['hbias2']
    for i in xrange(gs_steps):
        h1 = sigmoid(np.dot(W1.T, v) + np.dot(W2, h2) + hbias1)
        h1 = np.random.binomial(1, h1)
        h2 = sigmoid(np.dot(W2.T, h1) + hbias2)
        h2 = np.random.binomial(1, h2)
        v = sigmoid(np.dot(W1, h1) + vbias)
        v = np.random.binomial(1, v)
    return h1, h2, v


def get_cross_entropy(X, model):
    W1, W2, vbias = model['W1'], model['W2'], model['vbias']
    h2 = np.random.rand(W2.shape[1], X.shape[1])
    h1, _, _ = gibbs_sampling(h2, h2, X, model, 1)
    v = sigmoid(np.dot(W1, h1) + vbias)
    cross_entropy = -np.mean(np.sum(X * np.log(v) + (1 - X) * np.log(1 - v), axis=0))
    return cross_entropy


def train_DBM(trainset=None, devset=None, seed=3, num_hidden1=100, num_hidden2=100, epochs=100, learning_rate=0.01,
              batch_size=20, K=100, mf_steps=10, gs_step=1):
    np.random.seed(seed)
    X_train = load_data(trainset)
    X_dev = load_data(devset)

    # transpose input
    X_train = X_train.T  # 784 * 3000
    X_dev = X_dev.T

    insize = X_train.shape[0]

    model = {}
    W1 = init_w(insize, num_hidden1)       # 784 * 100
    W2 = init_w(num_hidden1, num_hidden2)  # 100 * 100
    vbias = init_b(insize).reshape(-1, 1)
    hbias1 = init_b(num_hidden1)
    hbias2 = init_b(num_hidden2)
    model['W1'], model['W2'] = W1, W2
    model['vbias'], model['hbias1'], model['hbias2'] = vbias, hbias1, hbias2

    # persistent chains
    v = np.random.binomial(1, 0.5, (insize, K))
    h1 = np.random.binomial(1, 0.5, (num_hidden1, K))
    h2 = np.random.binomial(1, 0.5, (num_hidden2, K))

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

            mu1, mu2 = mean_field(X_batch, model, mf_steps)

            h1, h2, v = gibbs_sampling(h1, h2, v, model, gs_step)

            h1_X_batch = sigmoid(np.dot(W1.T, X_batch) + np.dot(W2, mu2) + hbias1)
            h1_v = sigmoid(np.dot(W1.T, v) + np.dot(W2, h2) + hbias1)
            h2_X_batch = sigmoid(np.dot(W2.T, mu1) + hbias2)
            h2_v = sigmoid(np.dot(W2.T, h1) + hbias2)

            # update model
            W1 += learning_rate * (X_batch.dot(mu1.T)/X_batch.shape[1] - v.dot(h1.T)/v.shape[1])
            W2 += learning_rate * (mu1.dot(mu2.T) / mu1.shape[1] - h1.dot(h2.T) / h1.shape[1])
            vbias += learning_rate * (np.sum(X_batch, axis=1, keepdims=True)/X_batch.shape[1] - np.sum(v, axis=1, keepdims=True)/v.shape[1])
            hbias1 += learning_rate * (np.sum(h1_X_batch, axis=1, keepdims=True)/h1_X_batch.shape[1] - np.sum(h1_v, axis=1, keepdims=True)/h1_v.shape[1])
            hbias2 += learning_rate * (np.sum(h2_X_batch, axis=1, keepdims=True)/h2_X_batch.shape[1] - np.sum(h2_v, axis=1, keepdims=True)/h2_v.shape[1])

    # # draw cross entropy of train and dev
    draw_error_fig(epochs, train_error, dev_error)
    draw_weights(W1)
    draw_samples(insize, num_hidden1, num_hidden2, K, model)
    return train_error, dev_error


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


def draw_weights(weight, num_hidden):
    W = np.transpose(weight)
    W = np.reshape(W, (-1,28,28))

    fig = plt.figure()
    for i in xrange(W.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.axis('off')
        plt.imshow(W[i],cmap=plt.cm.binary)
    # plt.show()
    fig.savefig(str(num_hidden) + '_samples.png')



def draw_multi_error_fig(train_error1, dev_error1, train_error5, dev_error5, train_error10, dev_error10):
    fig, ax = plt.subplots()

    ax.plot(train_error1, 'orange', label='Train h=100')
    ax.plot(dev_error1, 'brown', label='Val h=100')
    ax.plot(train_error5, 'g', label='Train h=200')
    ax.plot(dev_error5, 'b', label='Val h=200')
    ax.plot(train_error10, 'r', label='Train h=400')
    ax.plot(dev_error10, 'c', label='Val h=400')

    ax.set_xlabel("number of epoches")
    ax.set_ylabel("avg cross-entropy error")

    legend = ax.legend(loc='upper right', shadow=False)

    for label in legend.get_texts():
        label.set_fontsize('small')

    fig.savefig('errors.png')
    # plt.show()

def draw_samples(insize, num_hidden1, num_hidden2, K, model):
    v = np.random.binomial(1, 0.5, (insize, K))
    h1 = np.random.binomial(1, 0.5, (num_hidden1, K))
    h2 = np.random.binomial(1, 0.5, (num_hidden2, K))
    h1, h2, v = gibbs_sampling(h1, h2, v, model, 1000)
    draw_weights(v, num_hidden1)


if __name__ == "__main__":
    train_file = "digitstrain.txt"
    dev_file = "digitsvalid.txt"
    test_file = "digitstest.txt"
    train1, dev1 = train_DBM(trainset=train_file, devset=dev_file, batch_size=10, num_hidden1=100, num_hidden2=100, epochs=100)
    # train2, dev2 = train_DBM(trainset=train_file, devset=dev_file, batch_size=10, num_hidden1=200, num_hidden2=200, epochs=100)
    # train3, dev3 = train_DBM(trainset=train_file, devset=dev_file, batch_size=10, num_hidden1=400, num_hidden2=400, epochs=100)
    # draw_multi_error_fig(train1, dev1, train2, dev2, train3, dev3)
