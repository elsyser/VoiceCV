from string import punctuation

import numpy as np

from .config import PATH_TO_DATA

__all__ = [
    'clean',
    'init_idx_word_map',
    'init_word_idx_map',
    'load_word_embedding_map',
    'init_word_embeddings_matrix',
]

def clean(sentence):
    # Tokenize
    tokens = sentence.split()
    
    # Lower Case
    tokens = [token.lower() for token in tokens]
    
    # Remove punct
    for i in range(len(tokens)):
        tokens[i] = ''.join([ch for ch in tokens[i] if ch not in punctuation])
    
    # Remove hanging chars
    tokens = [token for token in tokens if len(token) > 1 or token == 'a']
    
    # Remove tokens with digits in it
    tokens = [token for token in tokens if token.isalpha()]
    
    return ' '.join(tokens)

def init_idx_word_map(vocabulary):
    return {key: val for key, val in enumerate(vocabulary)}

def init_word_idx_map(vocabulary):
    return {val: key for key, val in enumerate(vocabulary)}

def load_word_embedding_map(path_to_embeddings=PATH_TO_DATA+'glove.6B/glove.6B.300d.txt'):
    word2embedding = dict()

    with open(path_to_embeddings, encoding='utf-8') as f_embeddings:
        for line in f_embeddings:
            values = line.split()
            word2embedding[values[0]] = np.asarray(values[1:], dtype='float64')
    
    return word2embedding

def init_word_embeddings_matrix(dim, voc_size, word2idx, path_to_embeddings=PATH_TO_DATA+'glove.6B/glove.6B.300d.txt'):
    word2embedding = load_word_embedding_map(path_to_embeddings=path_to_embeddings)
    embedding_matrix = np.zeros((voc_size, dim))

    for word, idx in word2idx.items():
        try:
            embedding_matrix[idx, :] = word2embedding[word]
        except KeyError:
            pass
        
    return embedding_matrix
