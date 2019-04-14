import pickle

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

from ..nn.inceptionv3_encoder import InceptionV3Encoder
from .config import IMAGE_SIZE, IMAGES_DIR

__all__ = [
    'load_image',
    'load_image_embedding_map',
]


def load_image(x, target_size=IMAGE_SIZE[:-1], preprocess=True, from_set=True):
    if from_set:
        x = image.load_img(IMAGES_DIR+x+'.jpg', target_size=target_size)
    else:
        x = image.load_img(x, target_size=target_size)

    x = image.img_to_array(x)
    
    if preprocess:
        x = preprocess_input(x)

    return x

def load_image_embedding_map(set_type, image2descriptions):
    try:
        with open(set_type+'_image_embedding_map.bin', 'rb') as f:
            image2embedding = pickle.load(f)
        print('"{}" Image-Embedding Map loaded.'.format(set_type))        

    except FileNotFoundError:
        print('Creating "{}" Image-Embedding Map...'.format(set_type))

        encoder = InceptionV3Encoder()
        image2embedding = dict()
        for img_id, _ in image2descriptions.items():
            img = load_image(img_id, preprocess=True)
            image2embedding[img_id] = encoder.encode_image(img)

        with open(set_type+'_image_embedding_map.bin', 'wb') as f:
            pickle.dump(image2embedding, f)

        print('Done.')
    
    return image2embedding
