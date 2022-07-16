import numpy as np
from keras.datasets import mnist

from layers.activation import ActivationTanh, ActivationSigmoid
from layers.dense import Dense
from layers.loss import mse, mse_derivative
from network import Network


def onehot(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape(-1, 28*28)
test_X = test_X.reshape(-1, 28*28)

train_X = train_X/255

test_X = test_X/255

train_y = onehot(train_y)


nn=Network()
nn.add(layer=Dense(28*28,100))
nn.add(layer=ActivationTanh())
nn.add(layer=Dense(100,20))
nn.add(layer=ActivationTanh())
nn.add(layer=Dense(20,10))
nn.add(layer=ActivationSigmoid())


nn.set_loss(mse, mse_derivative)

for _ in range(5):
    nn.train(train_X,train_y,epochs=2)
    pred_test = np.argmax(nn.predict(test_X), axis=-1)[:,0]
    res = (pred_test==test_y)
    print(f"val acc = {sum(res)/len(res)}")

    """
    I got this result, your may vary because of random 
    weights initialisation
    Ep 0 of 2; error: 0.6944664824334034
    Ep 1 of 2; error: 0.37499309026531846
    val acc = 0.8238
    Ep 0 of 2; error: 0.24724034167466957
    Ep 1 of 2; error: 0.20626130271434956
    val acc = 0.8753
    Ep 0 of 2; error: 0.18338725538411205
    Ep 1 of 2; error: 0.1666053870993996
    val acc = 0.893
    Ep 0 of 2; error: 0.15359583233981286
    Ep 1 of 2; error: 0.14206901861495466
    val acc = 0.9007
    Ep 0 of 2; error: 0.13371918150928236
    Ep 1 of 2; error: 0.12721655005891658
    val acc = 0.9029
    Ep 0 of 2; error: 0.12180233112416103
    Ep 1 of 2; error: 0.11588422364911849
    val acc = 0.9095
    Ep 0 of 2; error: 0.1109457789010171
    Ep 1 of 2; error: 0.10650046572979154
    val acc = 0.9146
    Ep 0 of 2; error: 0.10314389401554755
    Ep 1 of 2; error: 0.09953824924847322
    val acc = 0.915
    Ep 0 of 2; error: 0.09685477252940422
    Ep 1 of 2; error: 0.09323156425803951
    val acc = 0.9165
    Ep 0 of 2; error: 0.09059662404436787
    Ep 1 of 2; error: 0.08861594888893341
    val acc = 0.919
    
    So we got 92% accuracy on mnist with our simple network
    """
