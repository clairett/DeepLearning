# Deep Boltzman Machine
# author: tian tian

class DRM(object):
    def __init__(self):
        pass



def binarize_data(input):
    threshold, upper, lower = 0.5, 1, 0
    input = np.where(input>=threshold, upper, lower)
    return input

def load_data(file):
    data = np.loadtxt(file, dtype='float', delimiter=',')
    X = data[:, :-1]
    X = binarize_data(X)  # 3000*874
    return X

if __name__ == '__main__':
    pass