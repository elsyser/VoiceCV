from keras import Model
from keras.layers import Activation, Dense, Dropout, Input, Reshape
from keras.optimizers import RMSprop

__all__ = [
    'TopImageEncoder',
]


class TopImageEncoder:
    """Top Image Encoder
    
    Top Image Encoder Model.
    
    Parameters:
    -----------
    embedding_dim : integer, the dimension in which to embed the image and the tokens
    name : string, the name of the model, optional
    
    """

    def __init__(self, embedding_dim, inceptionv3_output_dim, name='top_image_encoder'):
        self.embedding_dim = embedding_dim
        self.inceptionv3_output_dim = inceptionv3_output_dim
        self.name = name
        
        # Inputs
        self.image_embedding_input = Input((inceptionv3_output_dim,), name='image_embedding_input')

        # Top
        self.dropout_encoder_1 = Dropout(rate=0.5, name='dropout_encoder_1')
        self.dense_encoder_1 = Dense(inceptionv3_output_dim//2, name='dense_encoder_1')
        self.relu_encoder_1 = Activation('relu', name='relu_encoder_1')

        self.dropout_encoder_2 = Dropout(rate=0.5, name='dropout_encoder_2')
        self.dense_encoder_2 = Dense(self.embedding_dim, name='dense_encoder_2')
        self.relu_encoder_2 = Activation('relu', name='relu_encoder_2')

        self.reshape = Reshape((1, self.embedding_dim), name='reshape')

        self.model = None
        self.build_model()

    def build_model(self):
        x = self.dropout_encoder_1(self.image_embedding_input)
        x = self.dense_encoder_1(x)
        x = self.relu_encoder_1(x)

        x = self.dropout_encoder_2(x)
        x = self.dense_encoder_2(x)
        x = self.relu_encoder_2(x)

        x = self.reshape(x)

        self.model = Model(self.image_embedding_input, x, name=self.name)
        self.model.compile(RMSprop(1e-4), loss='categorical_crossentropy')
        return self
