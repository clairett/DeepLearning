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

    def get_cost_update(self, lr=0.1, k=1, batch_size=10):
        for j in xrange(1, (3000 / batch_size) + 1):
            indice = np.random.permutation(3000)
            row = indice[:batch_size]
            # row = indice[(j - 1) * batch_size:j * batch_size]
            X_batch = self.input[row]

            ph_mean, ph_sample = self.sample_h_given_v(X_batch)
            chain_start = ph_sample

            # CD-k Algorithms
            for i in xrange(k):
                if i == 0:
                    nv_means, nv_samples, \
                    nh_means, nh_samples = self.gibbs_hvh(chain_start)
                else:
                    nv_means, nv_samples, \
                    nv_means, nv_samples = self.gibbs_hvh(nh_samples)

            # update model parameters
            self.W += lr * (np.dot(X_batch.T, ph_mean) - np.dot(nv_samples.T, nh_means))
            self.hbias += lr * np.mean(ph_mean - nh_means, axis=0)
            self.vbias += lr * np.mean(X_batch - nv_samples, axis=0)

    def get_reconstruction_cost(self, X):
        pre_sigmoid_nh = np.dot(X, self.W)+self.hbias
        sigmoid_nh = self.sigmoid(pre_sigmoid_nh)
        pre_sigmoid_nv = np.dot(sigmoid_nh, self.W.T)+self.vbias
        sigmoid_nv = self.sigmoid(pre_sigmoid_nv)
        cross_entropy = -np.mean(np.sum(X * np.log(sigmoid_nv) + (1 - X) * np.log(1 - sigmoid_nv), axis=1))
        return cross_entropy


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

    ax.plot(epochs, train_error, 'g', label='Training')
    ax.plot(epochs, dev_error, 'b', label='Validation')
    ax.set_xlabel("number of epoches")
    ax.set_ylabel("avg cross-entropy error")

    legend = ax.legend(loc='upper right', shadow=False)

    for label in legend.get_texts():
        label.set_fontsize('small')

    plt.show()

def train_rbm(learning_rate=0.1, k=1, max_epochs=100, batch_size=10):
    X_train = load_data("digitstrain.txt")
    X_dev = load_data("digitsvalid.txt")
    rng = np.random.RandomState(123)

    rbm = RBM(X_train, rng=rng)

    epochs = []
    train_error, dev_error = [], []

    for epoch in xrange(1, max_epochs + 1):
        rbm.get_cost_update()
        epochs.append(epoch)
        train_cost = rbm.get_reconstruction_cost(X_train)
        dev_cost = rbm.get_reconstruction_cost(X_dev)
        train_error.append(train_cost)
        dev_error.append(dev_cost)

        print "Train epoch %d, cross-entropy %f" % (epoch, train_cost)

    draw_error_fig(epochs, train_error, dev_error)
    # print rbm.W.size()


if __name__ == "__main__":
    train_rbm()


