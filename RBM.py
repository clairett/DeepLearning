import numpy as np
import math
import matplotlib.pyplot as plt

class RBM(object):
    def __init__(self, input, n_visible=784, n_hidden=100, W = None, hbias = None, vbias=None, rng=None):
        self.n_visible = n_visible
        self.hidden = n_hidden

        if W is None:
            W = np.random.uniform(-math.sqrt(6.0 / (n_visible + n_hidden)), math.sqrt(6.0 / (n_visible + n_hidden)), \
                                  [n_visible, n_hidden])

        if hbias is None:
            hbias = np.zeros(n_hidden)

        if vbias is None:
            vbias = np.zeros(n_visible)

        if rng is None:
            rng = np.random.RandomState(1234)

        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.rng = rng

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def propup(self, vis):
        pre_sigmoid_activation = np.dot(vis, self.W) + self.hbias
        return self.sigmoid(pre_sigmoid_activation)

    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)
        return [h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = np.dot(hid, self.W.T) + self.vbias
        return self.sigmoid(pre_sigmoid_activation)

    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.rng.binomial(size=v1_mean.shape, n=1, p=v1_mean)
        return [v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [v1_mean, v1_sample, h1_mean, h1_sample]

    def get_cost_update(self, lr=0.01, k=1, batch_size=10):
        indice = np.random.permutation(3000)
        for j in xrange(1, (3000 / batch_size) + 1):
            row = indice[(j - 1) * batch_size:j * batch_size]
            X_batch = self.input[row]

            indice = np.random.permutation(3000)
            ph_mean, ph_sample = self.sample_h_given_v(X_batch)
            chain_start = ph_sample

            # CD-k Algorithms
            for i in xrange(k):
                if i == 0:
                    nv_means, nv_samples, \
                    nh_means, nh_samples = self.gibbs_hvh(chain_start)
                else:
                    nv_means, nv_samples, \
                    nh_means, nh_samples = self.gibbs_hvh(nh_samples)

            # update model parameters
            self.W += lr * ((np.dot(X_batch.T, ph_mean) - np.dot(nv_samples.T, nh_means))/batch_size)
            self.hbias += lr * np.mean(ph_mean - nh_means, axis=0)
            self.vbias += lr * np.mean(X_batch - nv_samples, axis=0)

    def get_reconstruction_cost(self, X):
        pre_sigmoid_nh = np.dot(X, self.W)+self.hbias
        sigmoid_nh = self.sigmoid(pre_sigmoid_nh)
        pre_sigmoid_nv = np.dot(sigmoid_nh, self.W.T)+self.vbias
        sigmoid_nv = self.sigmoid(pre_sigmoid_nv)
        cross_entropy = -np.mean(np.sum(X * np.log(sigmoid_nv) + (1 - X) * np.log(1 - sigmoid_nv), axis=1))
        return cross_entropy

    def sample_analysis(self, k=1000):
        ph_mean, ph_sample = self.sample_h_given_v(self.input[:100])
        chain_start = ph_sample
        # CD-k Algorithms
        for i in xrange(k):
            if i == 0:
                nv_means, nv_samples, \
                nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples, \
                nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        print nv_samples.shape

        draw_weights(nv_samples.T)


def binarize_data(input):
    threshold, upper, lower = 0.5, 1, 0
    input = np.where(input>=threshold, upper, lower)
    return input

def load_data(file):
    data = np.loadtxt(file, dtype='float', delimiter=',')
    X = data[:, :-1]
    X = binarize_data(X)  # 3000*874
    return X

def draw_error_fig(epochs, train_error, dev_error):
    fig, ax = plt.subplots()

    ax.plot(epochs, train_error, 'g', label='Train Cross Entropy')
    ax.plot(epochs, dev_error, 'b', label='Validation Cross Entropy')
    ax.set_xlabel("number of epoches")
    ax.set_ylabel("avg cross-entropy error")

    legend = ax.legend(loc='upper right', shadow=False)

    for label in legend.get_texts():
        label.set_fontsize('small')
    plt.show()


def draw_multi_error_fig(epochs, train_error1, dev_error1, train_error5, dev_error5, train_error10, dev_error10, train_error20, dev_error20):
    fig, ax = plt.subplots()

    ax.plot(epochs, train_error1, 'orange', label='Train h=50')
    ax.plot(epochs, dev_error1, 'brown', label='Val h=50')
    ax.plot(epochs, train_error5, 'g', label='Train h=100')
    ax.plot(epochs, dev_error5, 'b', label='Val h=100')
    ax.plot(epochs, train_error10, 'r', label='Train h=200')
    ax.plot(epochs, dev_error10, 'c', label='Val h=200')
    ax.plot(epochs, train_error20, 'y', label='Train h=500')
    ax.plot(epochs, dev_error20, 'k', label='Val h=500')

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
        plt.subplot(10,10,i+1)
        plt.axis('off')
        plt.imshow(W[i],cmap=plt.cm.binary)
    plt.show()
    # fig.savefig(file_name+'_W.png')


def train_rbm(learning_rate=0.1, k=1, max_epochs=50, batch_size=10, n_hidden=100):
    X_train = load_data("digitstrain.txt")
    X_dev = load_data("digitsvalid.txt")
    rng = np.random.RandomState(123)
    epochs, train_error, dev_error = [], [], []

    rbm = RBM(X_train, rng=rng, n_hidden=n_hidden)

    for epoch in xrange(1, max_epochs + 1):
        rbm.get_cost_update(k=k)
        epochs.append(epoch)
        train_cost = rbm.get_reconstruction_cost(X_train)
        dev_cost = rbm.get_reconstruction_cost(X_dev)
        train_error.append(train_cost)
        dev_error.append(dev_cost)

        if epoch % 10 == 0:
            print "Train epoch %d, cross-entropy %f" % (epoch, train_cost)



    # np.savetxt('weight.txt', rbm.W)
    # np.savetxt('hbias.txt',rbm.hbias)
    # np.savetxt('vbias.txt', rbm.vbias)

    # W = np.loadtxt('weight.txt', dtype='float')
    # hbias = np.loadtxt('hbias.txt', dtype='float')
    # vbias = np.loadtxt('vbias.txt', dtype='float')
    # rbm = RBM(X_train, W=W, hbias=hbias, vbias=vbias, rng=rng)
    # print rbm.get_reconstruction_cost(X_train)
    # rbm.sample_analysis()

    return [epochs, train_error, dev_error]
    # draw_error_fig(epochs, train_error, dev_error)



if __name__ == "__main__":
    train_rbm()

    # epochs, train_error_k1, dev_error_k1 = train_rbm(k=1)
    # epochs, train_error_k5, dev_error_k5 = train_rbm(k=5)
    # epochs, train_error_k10, dev_error_k10 = train_rbm(k=10)
    # epochs, train_error_k20, dev_error_k20 = train_rbm(k=20)
    # draw_multi_error_fig(epochs, train_error_k1, dev_error_k1, train_error_k5, dev_error_k5, train_error_k10, \
    #                      dev_error_k10, train_error_k20, dev_error_k20)

    # epochs, train_error_h50, dev_error_h50 = train_rbm(n_hidden=50)
    # epochs, train_error_h100, dev_error_h100 = train_rbm(n_hidden=100)
    # epochs, train_error_h200, dev_error_h200 = train_rbm(n_hidden=200)
    # epochs, train_error_h500, dev_error_h500 = train_rbm(n_hidden=500)
    # draw_multi_error_fig(epochs, train_error_h50, dev_error_h50, train_error_h100, dev_error_h100, train_error_h200, \
    #                      dev_error_h200, train_error_h500, dev_error_h500)


