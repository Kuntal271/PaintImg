
### "Image colorization using Autoencoders. Transfer learning using VGG."
### "Importing Necessary Libraries"
from keras.preprocessing.image import img_to_array, load_img
import os

from ML_Pipeline.EncoderLayers import EncoderLayers
from ML_Pipeline.LoadData import LoadData
from ML_Pipeline.DecoderLayers import DecoderLayers
from ML_Pipeline.Inference import Inference

"""
Training Phase
"""
def trainingPhase():

    #Load Data
    datagen = LoadData()
    X, Y = datagen.getData()
    
    # Setup the Encoding Layer
    encoder_layer = EncoderLayers()
    encoded = encoder_layer.getfeatures(X)

    # Setup the Decoding Layer
    decoder_layer = DecoderLayers()
    decoder_layer.fit(encoded, Y)


"""
Inference Phase 
"""
def inferencePhase():
    # Setup the Encoding Layer
    encoder_layer = EncoderLayers()
    encode_model = encoder_layer.newmodel

    # Setup the Decoding Layer
    decoder_layer = DecoderLayers()
    model = decoder_layer.load_model()

    # taking Test images
    testpath = '../Input/dataset/test/test/'
    files = os.listdir(testpath)
    print(files)
    inf = Inference()

    # Inferencing and pasting the output
    for idx, file in enumerate(files):
        print(file)
        test = img_to_array(load_img(testpath+file))
        inf.processImg(idx, test, encode_model, model)

### Training the model ###
trainingPhase()

### Perform inference ###
inferencePhase()




