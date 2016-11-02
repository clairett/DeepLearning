import numpy as np
import math

class Autoencoder(object):
    def __init__(self, input, n_visible=784, n_hidden=100, W=None, bhid=None, bvis=None, rng=None):
        self.n_visible = n_visible
        self.hidden = n_hidden

        np.random.seed(0)
        if W is None:
            W = np.random.uniform(-math.sqrt(6.0 / (n_visible + n_hidden)), math.sqrt(6.0 / (n_visible + n_hidden)), \
                                  [n_visible, n_hidden])

        if bhid is None:
            bhid = np.zeros((1, n_hidden))

        if bvis is None:
            bvis = np.zeros((1, n_visible))

        if rng is None:
            rng = np.random.RandomState(1234)

        self.X = input
        self.W1 = W
        self.W2 = self.W1.T
        self.b1= bhid
        self.b2 = bvis
        self.rng = rng

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def get_corrupted_input(self, input, dropout):
        return np.random.choice(2, input.shape, p=[dropout, 1-dropout])*input

    def get_cost_update(self, lr=0.01, batch_size=10, dropout=0.25):
        indice = np.random.permutation(3000)
        for j in xrange(1, (3000 / batch_size) + 1):
            row = indice[(j - 1) * batch_size:j * batch_size]
            X_batch = self.X[row]

            # forward propagation
            tilde_x = self.get_corrupted_input(X_batch, dropout)
            a1 = np.dot(tilde_x, self.W1) + self.b1
            h1 = self.sigmoid(a1)
            a2 = np.dot(h1, self.W2) + self.b2
            o = self.sigmoid(a2)

            # backpropagation
            a2_grad = -(self.sigmoid(1-a2)*X_batch - self.sigmoid(a2)*(1-X_batch))/batch_size
            W2_grad = np.dot(h1.T, a2_grad)
            b2_grad = np.sum(a2_grad, axis=0, keepdims=True)

            h1_grad = np.dot(a2_grad, self.W2.T)
            a1_grad = h1_grad * h1 * (1-h1)
            W_grad = np.dot(X_batch.T, a1_grad)
            b_grad = np.sum(a1_grad, axis=0, keepdims=True)

            self.W2 -= lr * W2_grad
            self.b2 -= lr * b2_grad
            self.W1 -= lr * W_grad
            self.b1 -= lr * b_grad


    def get_reconstruction_cost(self, input):
        a1 = np.dot(input, self.W1) + self.b1
        h1 = self.sigmoid(a1)
        a2 = np.dot(h1, self.W2) + self.b2
        o = self.sigmoid(a2)
        L = -(np.sum(input * np.log(o)) + np.sum((1 - input) * np.log(1 - o)))/input.shape[0]
        return L


def binarize_data(input):
    threshold, upper, lower = 0.5, 1, 0
    input = np.where(input>=threshold, upper, lower)
    return input


def load_data(file):
    data = np.loadtxt(file, dtype='float', delimiter=',')
    X = data[:, :-1]
    X = binarize_data(X)  # 3000*874
    return X


def draw_weights(weight):
    import matplotlib.pyplot as plt
    W = np.transpose(weight)
    W = np.reshape(W, (-1,28,28))

    fig = plt.figure()
    for i in xrange(W.shape[0]):
        plt.subplot(10,10,i+1)
        plt.axis('off')
        plt.imshow(W[i],cmap=plt.cm.binary)
    plt.show()


def draw_multi_error_fig(epochs, train_error1, dev_error1, train_error5, dev_error5, train_error10, dev_error10, train_error20, dev_error20):
    import matplotlib.pyplot as plt
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


def train_autoencoder(learning_rate=0.1, max_epochs=50, batch_size=10, n_hidden=100, dropout=0):
    X_train = load_data("digitstrain.txt")
    X_dev = load_data("digitsvalid.txt")
    rng = np.random.RandomState(123)
    epochs, train_error, dev_error = [], [], []

    autoencoder = Autoencoder(X_train, rng=rng, n_hidden=n_hidden)

    epochs = []
    train_error, dev_error = [], []

    for epoch in xrange(1, max_epochs + 1):
        autoencoder.get_cost_update(dropout=dropout)
        epochs.append(epoch)
        train_cost = autoencoder.get_reconstruction_cost(X_train)
        dev_cost = autoencoder.get_reconstruction_cost(X_dev)
        train_error.append(train_cost)
        dev_error.append(dev_cost)
        print "Train epoch %d, cross-entropy %f" % (epoch, train_cost)


    #draw_weights(autoencoder.W1)
    # np.savetxt('dauto_weight.txt', autoencoder.W1)

    return [epochs, train_error, dev_error]

if __name__ == "__main__":
    train_autoencoder(dropout=0)

    # epochs, train_error_h50, dev_error_h50 = train_autoencoder(n_hidden=50)
    # epochs, train_error_h100, dev_error_h100 = train_autoencoder(n_hidden=100)
    # epochs, train_error_h200, dev_error_h200 = train_autoencoder(n_hidden=200)
    # epochs, train_error_h500, dev_error_h500 = train_autoencoder(n_hidden=500)
    # draw_multi_error_fig(epochs, train_error_h50, dev_error_h50, train_error_h100, dev_error_h100, train_error_h200, \
    #                      dev_error_h200, train_error_h500, dev_error_h500)




