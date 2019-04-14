from keras import Model
from keras.optimizers import RMSprop

from .inceptionv3_encoder import InceptionV3Encoder
from .top_image_encoder import TopImageEncoder
from .sequence_decoder import SequenceDecoder

__all__ = [
    'NeuralImageCaptioning',
]


class NeuralImageCaptioning:
    """Neural Image Captioning
    
    The full model for generating code from sketch.
    
    Parameters:
    -----------
    embedding_dim : integer, the dimension in which to embed the sketch image and the tokens
    maxlen : integer, the maximum code length
    voc_size : integer, number of unique tokens in the vocabulary
    num_hidden_neurons : list with length of 2, specifying the number of hidden neurons in the LSTM decoders
    name : string, the name of the model, optional
    
    """

    def __init__(self, embedding_dim, maxlen, voc_size, num_hidden_neurons, word2idx, name='neural_image_captioning'):
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.voc_size = voc_size
        self.num_hidden_neurons = num_hidden_neurons
        self.word2idx = word2idx
        self.name = name

        # Encoder / Decoder
        self.inceptionv3_encoder = InceptionV3Encoder()
        self.top_image_encoder = TopImageEncoder(embedding_dim, self.inceptionv3_encoder.model.output_shape[1])
        self.sequence_decoder = SequenceDecoder(maxlen, embedding_dim, voc_size, num_hidden_neurons, word2idx)

        # Inputs
        self.image_embedding_input = self.top_image_encoder.image_embedding_input
        self.sequence_input = self.sequence_decoder.sequence_input

        self.model = None
        self.build_model()

    def build_model(self):
        """Builds a Keras Model to train/predict"""

        final_image_embedding = self.top_image_encoder.model(self.image_embedding_input)
        sequence_output = self.sequence_decoder.model([self.sequence_input, final_image_embedding])

        self.model = Model([self.image_embedding_input, self.sequence_input], sequence_output, name=self.name)
        self.model.compile(RMSprop(1e-4), loss='categorical_crossentropy')
        return self
