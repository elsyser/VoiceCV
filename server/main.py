from flask import Flask , request
import base64
from PIL import Image
from vqa_pretrained import vqa

# from vqa_pretrained import vqa
app = Flask(__name__)

@app.route("/sendImage" , methods=["POST"])
def sendImage():
    req = request.get_json(force=True)
    img = base64.b64decode(req["imgData"])   
    image = Image.frombytes("RGB", (320, 240), img)
    image.save("image.jpg")
    print req["question"]
    vqa.main("image.jpg" , str(req["question"]))
    return "BACE"
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)