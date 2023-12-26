# Image Colorization

- Image colorization is the process of taking an input grayscale (black and white) image and then producing an output colorized image that represents the semantic colors and tones of the input.
- Image colorization assigns a color to each pixel of a target grayscale image
- Traditional colorization techniques requires significant user interaction.
- In this process, a fully automated data-driven approach “autoencoders” is used for colorization
- This method requires neither pre-processing nor post-processing.

## AutoEncoders

Autoencoders are a specific type of feedforward neural networks where the input is the same as the output. 
An autoencoder neural network is an Unsupervised Machine learning algorithm that applies backpropagation, setting the target values to be equal to the inputs.

## Backbone

-	Backbone is a term used in models/papers to refer to the feature extractor network.
-	This is a standard convolutional neural network (typically, VGG16 or VGG19) that serves as a feature extractor. The early layers detect low level features (edges and corners), and later layers successively detect higher level features (car, person, sky).
-	Backbone gives you a feature map representation of input
-	Multiple pre trained model used as a backbone for a variety of tasks 

## Code Description

    File Name : DecoderLayers.py
    File Description : Setting up the decoder layers architecture and loading the trained model



    File Name : Engine.py
    File Description : Main class for starting different parts and processes of training and inference lifecycle



    File Name : EncoderLayers.py
    File Description : Setting up the encoder layers architecture and loading the backbone VGG model



    File Name : Inference.py
    File Description : Inference cycle for setting up the end to end pipeline of inferencing and saving the image



    File Name : References.py
    File Description : References class for keeping the constants and reference path to dataset and weights



    File Name : LoadData.py
    File Description : LoadDataset class for loading the image data and converting it to LAB format


## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython (Google Colab)

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Modify `Engine.py` based on the mode that you are training on "Training" / "Inference"
- Run Code `python Engine.py`

### IPython Google Colab

Follow the instructions in the notebook `Image_Coloring_using_Transfer_Learning.ipynb`

