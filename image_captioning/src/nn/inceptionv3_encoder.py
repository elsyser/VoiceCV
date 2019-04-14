import numpy as np
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.optimizers import RMSprop

from ..utils.config import IMAGE_SIZE

__all__ = [
    'InceptionV3',
]


class InceptionV3Encoder:
    """InceptionV3 CNN
    
    
    Parameters:
    -----------
    name : string, the name of the model, optional
    
    """

    def __init__(self, name='inceptionv3_encoder'):
        self.name = name
        
        # Inputs
        self.image_input = Input(IMAGE_SIZE, name='image_input')

        # Get the InceptionV3 model trained on imagenet data
        self._inceptionv3_model = InceptionV3(weights='imagenet', include_top=True, input_tensor=self.image_input)

        self.model = None
        self.build_model()

    def build_model(self):
        self.model = Model(self._inceptionv3_model.input, self._inceptionv3_model.layers[-2].output, name=self.name)
        self.model.compile(RMSprop(1e-4), loss='categorical_crossentropy')
        return self

    def encode_image(self, image):
        return self.model.predict(np.expand_dims(image, 0))[0]

    def encode_images(self, images):
        return self.model.predict(images)
