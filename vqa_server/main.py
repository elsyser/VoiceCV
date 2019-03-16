from flask import Flask , request, make_response
import base64
import os
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
    print "Question is " + req["question"]
    os.system("cd vqa_pretrained; python vqa.py -image_file_name ../image.jpg -question \"" + req["question"] +"\"")
    res = ''
    with open("vqa_pretrained/output.txt" , 'r') as f:
        res = f.read()
    print res
    return make_response(res , 200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)