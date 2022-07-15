import numpy as np


class Network:
    """Simple sequential neural network class"""

    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_d = None

    def add(self, layer):
        """Add layer to network
            :param layer: layer to add
        """
        self.layers.append(layer)

    def set_loss(self, loss, loss_d):
        """
        Set loss function
        :param loss: loss function
        :param loss_d: loss function derivative
        """
        self.loss = loss
        self.loss_d = loss_d

    def predict(self, input):
        """
        Run on given input
        :param input: input
        :return: output
        """
        result = []
        # run over all samples
        for s in input:
            out = s
            for l in self.layers:
                out = l.fwd(out)
            result.append(out)
        return np.array(result)

    def predict_sample(self, sample):
        """
        Run on given sample
        :param sample: sample
        :return: output
        """
        out = sample
        for l in self.layers:
            out = l.fwd(out)
        return out

    def train(self, x_train, y_train, epochs, lr=0.01):
        """Train network on given samples
        :param x_train, y_train: train data
        :param epochs: number of epochs to run
        :param lr: learning rate, 0.01 is default
        """

        for e in range(epochs):
            err = 0
            for x_batch, y_batch in zip(x_train, y_train):
                out = self.predict_sample(x_batch)
                err += self.loss(y_batch, out)
                error = self.loss_d(y_batch, out)
                for l in self.layers[::-1]:
                    error = l.backward(error, lr)
            #average error on all samples
            err/=len(x_train)
            print(f"Ep {e} of {epochs}; error: {err}")