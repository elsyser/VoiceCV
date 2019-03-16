import json

import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model, model_from_json
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from PIL import Image

__all__ = [
    'IMAGE_ENCODER',
    'CAPTION_GENERATOR',
    
    'preprocess',
    'encode',
    'greedy_search_inference',
]


IMAGE_ENCODER = InceptionV3(weights='imagenet')
IMAGE_ENCODER = Model(IMAGE_ENCODER.inputs, IMAGE_ENCODER.layers[-2].output)
word2idx = json.load(open('./word2idx.json'))
idx2word = {val: key for key, val in word2idx.items()}

with open('./model.json', 'r') as json_file:
    architecture = json.load(json_file)
    architecture = json.dumps(architecture)

CAPTION_GENERATOR = model_from_json(architecture)
CAPTION_GENERATOR.load_weights('./model_30.h5')



def preprocess(path_to_img):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(path_to_img, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

def encode(path_to_img):
    image = preprocess(path_to_img) # preprocess the image
    fea_vec = IMAGE_ENCODER.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

def greedy_search_inference(path_to_img, max_length=34):
    encoded = encode(path_to_img).reshape((1, 2048))

    in_seq = 'startseq'

    for i in range(max_length):
        sequence = [word2idx[w] for w in in_seq.split() if w in word2idx]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = CAPTION_GENERATOR.predict([encoded, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = idx2word[yhat]
        in_seq += ' ' + word

        if word == 'endseq':
            break

    final = in_seq.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


if __name__ == '__main__':
    print(greedy_search_inference('./test.jpg'))
