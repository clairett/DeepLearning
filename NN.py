import numpy as np
import math

nn_input_dim = 784  # input layer dimensionality
nn_output_dim = 10  # output layer dimensionality

def load_data(file):
    data = np.loadtxt(file, dtype='float', delimiter=',')
    X = data[:, :-1]
    y = np.int8(data[:, -1])  # a vector
    return X, y

X_train, y_train = load_data("digitstrain.txt")
X_dev, y_dev = load_data("digitsvalid.txt")
X_test, y_test = load_data("digitstest.txt")


def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def build_1Layer_model(regularizer=False, reg_lambda=0.1, random_seed=0, epoch=100, batch_size=10, print_loss=False, epsilon=0.1, alpha=0.5, dropout=False, nn_hdim=100):
    np.random.seed(random_seed)

    # W1 = np.loadtxt('dauto_weight.txt', dtype='float')
    # W1 = np.loadtxt('auto_weight.txt', dtype='float')
    # W1 = np.loadtxt('weight.txt', dtype='float')  # RBM initialization
    W1 = np.random.uniform(-math.sqrt(6.0/(nn_input_dim+nn_hdim)), math.sqrt(6.0/(nn_input_dim+nn_hdim)), [nn_input_dim, nn_hdim])
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.uniform(-math.sqrt(6.0/(nn_hdim+nn_output_dim)), math.sqrt(6.0/(nn_hdim+nn_output_dim)), [nn_hdim, nn_output_dim])
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    old_dW1, old_dW2, old_db1, old_db2 = 0, 0, 0, 0

    for i in xrange(1, epoch+1):
        indice = np.random.permutation(3000)
        for j in xrange(1, (3000/batch_size)+1):
            row = indice[(j - 1) * batch_size:j * batch_size]
            X_batch = X_train[row]
            y_batch = y_train[row]
            mask = np.random.choice([0, 1], (batch_size, nn_hdim))

            # forward propagation
            z1 = X_batch.dot(W1) + b1
            a1 = sigmoid(z1)
            if dropout:
                a1 *= mask
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # backpropagation
            delta3 = probs
            delta3[range(batch_size), y_batch] -= 1
            dW2 = (a1.T).dot(delta3) * (1./batch_size)
            db2 = np.sum(delta3, axis=0, keepdims=True) * (1./batch_size)  #sum on vertical axis
            delta2 = delta3.dot(W2.T) * a1 * (1-a1)
            dW1 = np.dot(X_batch.T, delta2)
            db1 = np.sum(delta2, axis=0)

            if regularizer:
                dW1 += reg_lambda * W1
                dW2 += reg_lambda * W2

            W1 -= epsilon * (dW1 + old_dW1 * alpha)
            b1 -= epsilon * (db1 + old_db1 * alpha)
            W2 -= epsilon * (dW2 + old_dW2 * alpha)
            b2 -= epsilon * (db2 + old_db2 * alpha)

            old_db1 = db1
            old_db2 = db2
            old_dW1 = dW1
            old_dW2 = dW2

            model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 10 == 0:
            print "Epoch %i: error %f loss %f" % (i, calculate_error(model, X_train, y_train, True), calculate_1Layer_loss(model, X_train, y_train, regularizer, reg_lambda))
    return model


def calculate_error(model, X, y, oneLayer):
    return np.mean(predict_1Layer(model, X) != y)*100


def calculate_1Layer_loss(model, X, y, regularizer, reg_lambda):
    num_examples = len(X)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # forward propagation
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # calculate the loss
    logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(logprobs)

    if regularizer:
        data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss


def predict_1Layer(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # forward propagation
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def print_error_message(X, y, model, message, regularizer, reg_lambda):
    print "1-layer Neural Network"
    error = np.mean(predict_1Layer(model, X) != y) * 100
    print message+" error: %%%0.2f" % (error)
    loss = calculate_1Layer_loss(model, X, y, regularizer, reg_lambda)
    print message + " loss: %0.2f" % (loss)

regular = False
lam = 0.1

model = build_1Layer_model(regularizer=regular, reg_lambda=lam, epoch=100, random_seed=0, print_loss=True, epsilon=0.1, alpha=0.5, dropout=False, nn_hdim=100)
print_error_message(X_train, y_train, model, "train", regularizer=regular, reg_lambda=lam)
print_error_message(X_dev, y_dev, model, "dev", regularizer=regular, reg_lambda=lam)
print_error_message(X_test, y_test, model, "test", regularizer=regular, reg_lambda=lam)