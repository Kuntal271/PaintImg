
from .References import References
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab
import numpy as np

class LoadData(References):

    def __init__(self):
        """
        VGG16 is expecting an image of 3 dimension with size 224x224 as an input,
        in preprocessing we have to scale all images to 224 instead of 256
        """
        # Normalize images - divide by 255
        train_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train = train_datagen.flow_from_directory(self.path, target_size=(224, 224), batch_size=100, class_mode=None)


    def getData(self):
        "Convert from RGB to Lab: LAB image is a grey image in L channel and all color info stored in A and B channels"
        X = []
        Y = []
        for img in self.train[0]:
            try:
                lab = rgb2lab(img)
                X.append(lab[:, :, 0])
                Y.append(lab[:, :, 1:] / 128)  # A and B values range from -127 to 128,
                # so we divide the values by 128 to restrict values to between -1 and 1.
            except:
                print('error')
        X = np.array(X)
        Y = np.array(Y)
        X = X.reshape(X.shape + (1,))  # dimensions to be the same for X and Y
        return X, Y