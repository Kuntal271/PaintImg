
from keras.models import Sequential

from skimage.color import gray2rgb
import numpy as np

from keras.applications.vgg16 import VGG16
from .References import References

class EncoderLayers(References):

    def __init__(self):
        """ Initializing the Encoder"""
        self.vggmodel = VGG16()
        self.newmodel = Sequential()
        self.layers()

    def layers(self):
        "Replacing the encoder part with Feature Extraxtor of VGG"
        for i, layer in enumerate(self.vggmodel.layers):
            if i < 19:  # Only up to 19th layer to include feature extraction only
                self.newmodel.add(layer)
        self.newmodel.summary()
        for layer in self.newmodel.layers:
            layer.trainable = False  # We don't want to train these layers again, so False.
        return self.newmodel


    def getfeatures(self, X):

        # now we have one channel of L in each layer but, VGG16 is expecting 3 dimension,
        # so we repeated the L channel two times to get 3 dimensions of the same L channel
        vggfeatures = []
        for i, sample in enumerate(X):
            sample = gray2rgb(sample)
            sample = sample.reshape((1, 224, 224, 3))
            prediction = self.newmodel.predict(sample)
            prediction = prediction.reshape((7, 7, 512))
            vggfeatures.append(prediction)
        vggfeatures = np.array(vggfeatures)
        return vggfeatures
