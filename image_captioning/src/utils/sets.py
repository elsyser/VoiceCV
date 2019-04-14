import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from .config import TEXT_FILES_DIR

__all__ = [
    'load_raw_image_description_map',
    'init_image_descriptions_map',
    'load_set_images',
    'init_image_descriptions_map_set',
    'shuffled_dict_keys',
    'map2matrix',
    'data_generator',
]


def load_raw_image_description_map():
    with open(TEXT_FILES_DIR + 'Flickr8k.token.txt', 'r') as flickr8_token:
        raw_image2description = flickr8_token.read().split('\n')[:-1]

    return raw_image2description

def init_image_descriptions_map():
    raw_image2description = load_raw_image_description_map()
    image2descriptions = dict()
    
    i = 0
    while i < len(raw_image2description):
        img_name = raw_image2description[i].split('.')[0]
        image2descriptions[img_name] = []
        
        while i < len(raw_image2description) and img_name == raw_image2description[i].split('.')[0]:
            descr = raw_image2description[i].split('\t')[1]
            image2descriptions[img_name].append(descr)
            i+=1
            
    return image2descriptions

def load_set_images(type):
    if type == 'train':
        filename = TEXT_FILES_DIR + 'Flickr_8k.trainImages.txt'
    elif type == 'dev':
        filename = TEXT_FILES_DIR + 'Flickr_8k.devImages.txt'
    else:
        filename = TEXT_FILES_DIR + 'Flickr_8k.testImages.txt'

    with open(filename, 'r') as f:
        img_names = f.read().split('\n')[:-1]
        
    img_names = [name.split('.')[0] for name in img_names]
    return img_names

def init_image_descriptions_map_set(set_images, image2descriptions):
    image2descriptions_set = dict()
    
    for img_name in set_images:
        image2descriptions_set[img_name] = []
        descriptions = image2descriptions[img_name]
        
        for desc in descriptions:
            image2descriptions_set[img_name].append(
                desc,
            )
    
    return image2descriptions_set

def shuffled_dict_keys(x):
    keys = list(x.keys())
    np.random.shuffle(keys)
    return keys

def map2matrix(x):
    matrix = []
    for key, vals in x.items():
        for val in vals:
            matrix.append([key, val])

    return np.array(matrix)

def data_generator(image2descriptions, image2embedding, word2idx, batch_size, max_length, voc_size):
    X_img = []
    X_seq = []
    Y_seq = []
    n = 0

    image_description_matrix = map2matrix(image2descriptions)

    # loop for ever over images
    while True:
        
        # Shuffle the dataset
        np.random.shuffle(image_description_matrix)
        for example in image_description_matrix:
            img_id, desc = example
    
            # retrieve the image embedding
            image_emb = image2embedding[img_id]

            X_img.append(image_emb)
                
            # encode the sequence
            y_seq = [word2idx[word] for word in desc.split()] + [word2idx['<END>']]
            x_seq = [word2idx['<START>']] + y_seq[:-1]
                
            Y_seq.append(y_seq)
            X_seq.append(x_seq)
            
            n+=1
            if n == batch_size:
                X_seq = pad_sequences(X_seq, maxlen=max_length, padding='post')
                Y_seq = pad_sequences(Y_seq, maxlen=max_length, padding='post')

                # One-hot
                Y_seq = [[to_categorical(idx, voc_size) for idx in sent] for sent in Y_seq]

                yield [[np.array(X_img), X_seq], np.array(Y_seq)]

                X_img = []
                X_seq = []
                Y_seq = []
                n = 0
