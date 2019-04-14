import os

from keras import Model
from keras.layers import (LSTM, Activation, Dense, Dropout, Embedding, Input,
                          TimeDistributed)
from keras.optimizers import RMSprop

from ..utils.sequence import init_word_embeddings_matrix

__all__ = [
    'SequenceDecoder',
]


class SequenceDecoder:
    """Sequence Deocder
    
    Code Generating Model.
    
    Parameters:
    -----------
    maxlen : integer, the maximum code length
    embedding_dim : integer, the dimension in which to embed the image and the tokens
    voc_size : integer, number of unique tokens in the vocabulary
    num_hidden_neurons : list with length of 2, specifying the number of hidden neurons in the LSTM decoders
    name : string, the name of the model, optional
    
    """

    def __init__(self, maxlen, embedding_dim, voc_size, num_hidden_neurons, word2idx, name='sequence_decoder'):
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.voc_size = voc_size
        self.num_hidden_neurons = num_hidden_neurons
        self.word2idx = word2idx
        self.name = name

        # Inputs
        self.sequence_input = Input((maxlen,), name='sequence_input')
        self.image_embedding_input = Input((1, embedding_dim), name='image_embedding_input')
        
        # Embeddings
        self.embeddings = Embedding(
            voc_size,
            embedding_dim,
            input_length=maxlen,
            mask_zero=True,
            name='glove_embeddings'
        )
        self.embeddings_dropout = Dropout(rate=0.5, name='embeddings_dropout')

        # LSTM decoders
        self.lstm_decoder_1 = LSTM(num_hidden_neurons[0], return_sequences=True, return_state=True, name='lstm_decoder_1')
        self.lstm_decoder_2 = LSTM(num_hidden_neurons[1], return_sequences=True, return_state=True, name='lstm_decoder_2')

        # Dense -> Softmax decoder
        self.dense_decoder = TimeDistributed(Dense(voc_size, name='dense_layer'), name='dense_decoder')
        self.softmax_decoder = TimeDistributed(Activation('softmax', name='softmax_layer'), name='softmax_decoder')

        self.model = None
        self.build_model()

    def build_model(self):
        """Builds a Keras Model to train/predict"""

        x = self.embeddings(self.sequence_input)
        x = self.embeddings_dropout(x)

        _, h_state, c_state = self.lstm_decoder_1(self.image_embedding_input)

        x, _, _ = self.lstm_decoder_1(x, initial_state=[h_state, c_state])
        x, _, _ = self.lstm_decoder_2(x)

        x = self.dense_decoder(x)
        x = self.softmax_decoder(x)

        self.model = Model([self.sequence_input, self.image_embedding_input], x, name=self.name)
        
        # Load Fixed GloVe Word Embeddings
        dirname = os.path.dirname(__file__)
        path_to_embeddings = os.path.join(dirname, '../../data/glove.6B/glove.6B.'+str(self.embedding_dim)+'d.txt')
        word_embeddings_matrix = init_word_embeddings_matrix(
            self.embedding_dim,
            self.voc_size,
            self.word2idx,
            path_to_embeddings=path_to_embeddings
        )
        self.model.get_layer('glove_embeddings').set_weights([word_embeddings_matrix])
        self.model.get_layer('glove_embeddings').trainable = False

        self.model.compile(RMSprop(1e-4), loss='categorical_crossentropy')
        return self
