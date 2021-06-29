import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D, BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model

class ResNet:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.shape = x_train[0].shape # for Input
        self.num_class = len(np.unique(y_train)) # num of classes for outputs
        self.x_train = x_train/255
        self.x_test = x_test/255
        self.y_train = y_train.flatten()
        self.y_test = y_test.flatten()

    def resModel(self, count):

        # should be the same value of the first Conv2D()
        filtersize = 32

        # first layer
        i = Input(shape=self.shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(i) # todo padding='valid'?
        x = BatchNormalization()(x)

        # key point of resnet is shortcut
        # here is finding the shortcut values for each period
        # previous outputs and next input must be the same -> do not change the filtersize in this loop
        # weight 1 - relu - weight 2 - concatenate[f(x) + x] - relu
        for itr in range (count):

            # for concatenation
            shortcut = x

            # weight layer 1
            x = Conv2D(filtersize, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)

            # relu
            x = Activation('relu')(x)

            # weight layer 2
            x = Conv2D(filtersize, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)

            x = Concatenate()([x, shortcut]) # f(x) + x

            x = Activation('relu')(x)

            # increase the filter size
            if itr != count - 1:
                x = MaxPool2D((2, 2))(x)
                filtersize = filtersize * 2
                # first: 32, second:64, third:128, fourth:256

        x = Flatten()(x)
        x = Dropout(0.2)(x)  # drop out 20% of nodes randomly for regularization
        # x = GlobalMaxPool2D()(x) # if the image sizes are different

        x = Dense(512, activation='relu')(x)  # first Dense layer
        x = Dense(self.num_class, activation='softmax')(x)  # output layer: set(y_train) collection of unique elements

        model = Model(i, x)  # model created here

        print(model.summary())  # check the model

        model, history = self.training(model)

        return model, history

    def training(self, model):

        model.compile(optimizer='adam',
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        # Model weights are saved at the end of every epoch, if it's the best seen
        checkpoint_filepath = '/temp/checkpoint' # need to change this path name

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        r = model.fit(self.x_train, self.y_train,
                      validation_data=(self.x_test, self.y_test),
                      epochs=50,
                      callbacks=[model_checkpoint_callback])
        # batch_size = 100 needs 12gb Ram or more

        # model save
        model.save('Cifar10.h5')
        np.save('my_history.npy', r.history)

        return r, r.history

    def plot(self, r, history):
        # step 4 Evaluate the model
        plt.plot(history['loss'], label="loss")
        plt.plot(history['val_loss'], label="val_loss")
        plt.title("Loss")
        plt.legend()
        plt.show()

        # if we want to have better val_accuracy, we need to get regularization well
        # 1) change the hyper parameters values (128 in this case)
        # 2) use random search for regularization

        plt.plot(history['accuracy'], label="accuracy")
        plt.plot(history['val_accuracy'], label="val_accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    data = tf.keras.datasets.cifar10
    (x_train, y_train), ( x_test, y_test) = data.load_data()
    resnet = ResNet(x_train, y_train, x_test, y_test) # preparing the dataset for the networks

    model, history = resnet.resModel(4) # if we need to change this value, we also need to change Dense values (currently 512)

    resnet.plot(model, history)
