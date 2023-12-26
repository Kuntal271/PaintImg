### "Image colorization using Autoencoders. Transfer learning using VGG."
### "Importing Necessary Libraries"
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np

from .References import References

class Inference(References):

    def processImg(self, idx, test, newmodel, model):
        """
        Processing the image and predicting the output
        :param idx:
        :param test:
        :param newmodel:
        :param model:
        :return:
        """
        test = resize(test, (224, 224), anti_aliasing=True)
        test *= 1.0 / 255
        lab = rgb2lab(test)
        l = lab[:, :, 0]
        L = gray2rgb(l)
        L = L.reshape((1, 224, 224, 3))
        vggpred = newmodel.predict(L)
        ab = model.predict(vggpred)
        ab = ab * 128
        cur = np.zeros((224, 224, 3))
        cur[:, :, 0] = l
        cur[:, :, 1:] = ab
        imsave(self.ROOT_DIR+ self.TEST_IMG+ str(idx) + ".jpg", lab2rgb(cur))