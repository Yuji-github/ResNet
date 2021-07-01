import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPool2D, BatchNormalization, Activation, Concatenate, Add
from tensorflow.keras.models import Model

class ResNet:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.shape = x_train[0].shape # for Input (32, 32, 3)
        self.num_class = len(np.unique(y_train)) # num of classes for outputs
        self.x_train = x_train/255
        self.x_test = x_test/255
        self.y_train = y_train.flatten()
        self.y_test = y_test.flatten()

    def inputLayer(self):
        return Input(shape=self.shape)

    def identityBlock(self, x, filtersize):
        # key point of resnet is shortcut
        
        # identity resnet 
        # shortcut: inpput
        # regular: weight 1 - relu - weight 2  - add[f(x) + x] - relu

        # convBlock:
        # shortcut: weight 1
        # regular: weight 1 - relu - weight 2  - add[f(x) + x] - relu

        # for concatenation
        shortcut = x
        if x.shape[-1] != filtersize:
		        shortcut = Conv2D(filtersize, (1,1), padding='same', activation='relu')(x)

        # weight layer 1
        x = Conv2D(filtersize, (3, 3), padding='same')(x) 
        x = BatchNormalization()(x)

        # relu
        x = Activation('relu')(x)

        # weight layer 2
        x = Conv2D(filtersize, (3, 3), padding='same')(x) 
        x = BatchNormalization()(x)

        # f(x) + x
        x = Add()([x, shortcut])

        # relu
        x = Activation('relu')(x)

        return x

    def fullConnection(self, i, x):
        x = Flatten()(x)
        x = Dropout(0.2)(x)  # drop out 20% of nodes randomly for regularization
        # x = GlobalMaxPool2D()(x) # if the image sizes are different

        x = Dense(128, activation='relu')(x)  # first Dense layer
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
        checkpoint_filepath = '/temp/ResNet_tf2_Cifer10/checkpoint'

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        r = model.fit(self.x_train, self.y_train,
                      validation_data=(self.x_test, self.y_test),
                      epochs=100,
                      batch_size = 50,
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
    filtersize = 32 
    data = tf.keras.datasets.cifar10
    (x_train, y_train), ( x_test, y_test) = data.load_data()
    resnet = ResNet(x_train, y_train, x_test, y_test) # preparing the dataset for the networks

    inputLayer = resnet.inputLayer()
    
    # create te first layer
    model = Conv2D(filtersize, (1, 1), padding='same')(inputLayer)

    # 1st Identity block layer
    model = resnet.identityBlock(model, filtersize)

    model, history = resnet.fullConnection(inputLayer, model)

    resnet.plot(model, history)
