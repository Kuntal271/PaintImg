
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
import tensorflow as tf

from .References import References

class DecoderLayers(References):

    def __init__(self):

        self.model = Sequential()
        self.__setModelArch()

    def __setModelArch(self):
        """
        Setting up the model Architecture
        :return:
        """
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(7, 7, 512)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        self.model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

    def fit(self, vggfeatures, Y):
        """
        Fitting up the model for training
        :param vggfeatures:
        :param Y:
        :return:
        """
        self.model.fit(vggfeatures, Y, verbose=1, epochs=2000, batch_size=16)

        self.model.save(self.ROOT_DIR+self.SAVE_MODEL)

    def load_model(self):
        """
        Loading Trained Model
        :return:
        """
        self.model = tf.keras.models.load_model(self.ROOT_DIR+self.SAVE_MODEL,
                                           custom_objects=None,
                                           compile=True)
        return self.model

