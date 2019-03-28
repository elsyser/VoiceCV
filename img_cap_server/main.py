import base64
import os

from PIL import Image
import os

from flask import Flask, make_response, request
from img_cap.main import greedy_search_inference

app = Flask(__name__)


@app.route('/sendImage' , methods=['POST'])
def sendImage():
    req = request.get_json(force=True)
    img = base64.b64decode(req['imgData'])   
    image = Image.frombytes('RGB', (320, 240), img)
    image.save('image.jpg')


    # res = greedy_search_inference('./image.jpg')
    os.system("python img_cap/main.py ./image.jpg")
    with open("./output.txt" , 'r') as f:
        res = f.read()

    return make_response(res , 200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
