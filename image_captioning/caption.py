import json
import sys

from src.nn import NeuralImageCaptioning, NICInference
from src.utils.image import load_image

EMBEDDING_DIM = 300
NUM_HIDDEN_NEURONS = [256, 256]

MAXLEN = 37
word2idx = json.load(open('./word2idx.json'))
VOC_SIZE = len(word2idx)


if __name__ == '__main__':
    neural_image_captioning = NeuralImageCaptioning(
        EMBEDDING_DIM,
        MAXLEN,
        VOC_SIZE,
        NUM_HIDDEN_NEURONS,
        word2idx,
    )

    neural_image_captioning.model.load_weights('./src/weights/nic-weights.hdf5')

    nic_inference = NICInference(neural_image_captioning, word2idx)
    path_to_image = sys.argv[1]
    image = load_image(path_to_image, from_set=False)
    print(nic_inference.greedy_search(image))
