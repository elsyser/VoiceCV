import base64
import json

from flask import Flask, make_response, request
from PIL import Image
from src.nn import NeuralImageCaptioning, NICInference
from src.utils.image import load_image

app = Flask(__name__)

@app.route('/sendImage' , methods=['POST'])
def sendImage():
    req = request.get_json(force=True)
    img = base64.b64decode(req['imgData'])   
    image = Image.frombytes('RGB', (320, 240), img)
    image.save('./image.jpg')

    preprocessed_image = load_image('./image.jpg', from_set=False)
    caption = nic_inference.greedy_search(preprocessed_image)
    return make_response(caption , 200)


if __name__ == '__main__':
    EMBEDDING_DIM = 300
    NUM_HIDDEN_NEURONS = [256, 256]

    MAXLEN = 37
    word2idx = json.load(open('./word2idx.json'))
    VOC_SIZE = len(word2idx)

    neural_image_captioning = NeuralImageCaptioning(
        EMBEDDING_DIM,
        MAXLEN,
        VOC_SIZE,
        NUM_HIDDEN_NEURONS,
        word2idx,
    )
    neural_image_captioning.model.load_weights('./src/weights/nic-weights.hdf5')
    nic_inference = NICInference(neural_image_captioning, word2idx)


    app.run(host='0.0.0.0', port=4000)
