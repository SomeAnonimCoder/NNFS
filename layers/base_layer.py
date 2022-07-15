class BaseLayer:
    """Base Layer class"""
    def __init__(self):
        self.input=None
        self.output=None

    def fwd(self,input):
        """Compute output of layer from its input
            :param input:  input numpy array
        """
        raise NotImplementedError

    def backward(self, out_error, lr):
        """Compute dE/dX for given dE/dY and update parameters if any
            :param out_error: dE/dY
            :param lr: learning rate
        """
        raise NotImplementedError