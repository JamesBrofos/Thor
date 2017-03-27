from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import regularizers

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

input_dim = 784 #28*28
X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



def mnist_trainer(batch_size, nb_epoch, lr, l2_weight):
    sgd = optimizers.SGD(lr=lr)
    model = Sequential()
    model.add(
        Dense(
            nb_classes,
            input_dim=input_dim,
            activation='softmax',
            kernel_regularizer=regularizers.l2(l2_weight),
        )
    )
    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        verbose=1,
        validation_data=(X_test, Y_test)
    )
    score = model.evaluate(X_test, Y_test, verbose=0)

    return score[1]

